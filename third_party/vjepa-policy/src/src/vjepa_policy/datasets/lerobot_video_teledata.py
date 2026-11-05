"""Reader for the episode-per-file TeleData LeRobot v2.1 layout.

The Tianji TeleData exports use the v2.1 metadata version but store one frame
parquet and one video per episode. This reader keeps that source layout
read-only and exposes the same sample contract as the v3 reader.
"""

import bisect
import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from vjepa_policy.datasets.lerobot_video import (
    DEFAULT_PROMPT,
    TextEmbeddingCache,
    _make_relative_action,
    _validate_feature_indices,
    make_video_offsets,
)
from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.datasets.transforms import combine_video_clips


class _TorchCodecDecoderCache:
    """Small per-worker cache for the episode video decoders.

    ``decode_video_frames`` constructs a new ``VideoDecoder`` for every view
    and sample.  TeleData samples revisit the same episode files frequently,
    so keeping a bounded LRU in each DataLoader worker avoids that setup and
    seek-table cost.  Workers are intentionally non-persistent in the trainer;
    their normal recycle cycle also releases the native decoder resources.
    """

    def __init__(self, max_size: int = 64) -> None:
        self.max_size = max_size
        self._decoders: OrderedDict[str, object] = OrderedDict()

    def decode(
        self,
        video_path: Path | str,
        timestamps: Sequence[float],
        tolerance_s: float,
    ) -> torch.Tensor:
        from torchcodec.decoders import VideoDecoder

        key = str(video_path)
        decoder = self._decoders.get(key)
        if decoder is None:
            decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
            self._decoders[key] = decoder
            if len(self._decoders) > self.max_size:
                self._decoders.popitem(last=False)
        else:
            self._decoders.move_to_end(key)

        average_fps = float(decoder.metadata.average_fps)
        frame_indices = [
            round(float(timestamp) * average_fps) for timestamp in timestamps
        ]
        frames = decoder.get_frames_at(indices=frame_indices)
        loaded_ts = frames.pts_seconds
        query_ts = torch.as_tensor(timestamps, dtype=torch.float32)
        distances = torch.cdist(query_ts[:, None], loaded_ts.float()[:, None], p=1)
        min_distance, nearest = distances.min(1)
        if not bool((min_distance < tolerance_s).all()):
            raise AssertionError(
                "One or several query timestamps unexpectedly violate the tolerance "
                f"({min_distance[~(min_distance < tolerance_s)]} > {tolerance_s=}) "
                f"for video {video_path}"
            )
        return frames.data.index_select(0, nearest).to(torch.float32).div_(255)


def read_teledata_tasks(dataset_dir: str | Path) -> dict[int, str]:
    """Read task text from the TeleData ``tasks.jsonl`` file."""
    path = Path(dataset_dir) / "meta" / "tasks.jsonl"
    result: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            index = int(record["task_index"])
            text = record.get("task")
            if not isinstance(text, str):
                values = record.get("tasks")
                text = values[0] if isinstance(values, list) and values else None
            if isinstance(text, str) and text.strip():
                result[index] = text.strip()
    if not result:
        raise ValueError(f"No non-empty task instructions found in {path}")
    return result


def _fixed_list_array(column, width: int, name: str) -> np.ndarray:
    """Convert an Arrow fixed-size list column to a two-dimensional ndarray."""
    values = column.combine_chunks().values.to_numpy(zero_copy_only=False)
    array = np.asarray(values, dtype=np.float32).reshape(-1, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Unexpected {name} shape {array.shape}")
    return array


class TeleDataEpisodeDataset(Dataset):
    """VJEPA-Policy samples from TeleData's episode-per-file export."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        video_keys: Sequence[str],
        num_frames: int = 33,
        past_frames: int = 4,
        video_stride: int = 4,
        action_key: str = "action",
        state_key: str = "observation.state",
        action_indices: Sequence[int] | None = None,
        state_indices: Sequence[int] | None = None,
        relative_action_indices: Sequence[int] | None = None,
        video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        view_combiner: str | Callable[[Sequence[torch.Tensor]], torch.Tensor] = "independent",
        text_cache_dir: str | Path | None = None,
        context_len: int = 128,
        prompt_template: str = DEFAULT_PROMPT,
        normalization_stats_path: str | Path | None = None,
        action_normalization: str = "QUANTILE",
        state_normalization: str = "QUANTILE",
        clip_normalized: bool = False,
        video_backend: str = "torchcodec",
        require_full_future_window: bool = False,
    ) -> None:
        self.root = Path(dataset_dir)
        with (self.root / "meta" / "info.json").open(encoding="utf-8") as handle:
            self.info = json.load(handle)
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(
                f"Expected TeleData v2.1 metadata at {self.root}, got "
                f"{self.info.get('codebase_version')!r}"
            )
        self.fps = int(self.info["fps"])
        features = self.info["features"]
        self.video_keys = list(video_keys)
        if not self.video_keys:
            raise ValueError("At least one TeleData video view is required")
        for key in [*self.video_keys, action_key, state_key]:
            if key not in features:
                raise KeyError(f"Feature {key!r} not found in {self.root}")

        self.action_key = action_key
        self.state_key = state_key
        self.action_indices = _validate_feature_indices(
            action_indices, int(features[action_key]["shape"][0]), action_key
        )
        self.state_indices = _validate_feature_indices(
            state_indices, int(features[state_key]["shape"][0]), state_key
        )
        self.relative_action_indices = (
            _validate_feature_indices(
                relative_action_indices, len(self.action_indices), "relative action"
            )
            if relative_action_indices is not None
            else ()
        )
        if self.relative_action_indices and self.action_indices != self.state_indices:
            raise ValueError("Relative actions require aligned action and state indices")
        self._action_index = torch.tensor(self.action_indices, dtype=torch.long)
        self._state_index = torch.tensor(self.state_indices, dtype=torch.long)
        self._relative_action_index = torch.tensor(
            self.relative_action_indices, dtype=torch.long
        )

        self._video_offsets = make_video_offsets(num_frames, past_frames, video_stride)
        self._action_offsets = tuple(range(num_frames - 1))
        self._state_offsets = (0,)
        self._sampling_num_frames = num_frames
        self._past_frames = past_frames
        self._video_stride = video_stride
        self.video_transform = video_transform
        self.view_combiner = view_combiner
        self.prompt_template = prompt_template
        self.video_backend = video_backend
        self.require_full_future_window = bool(require_full_future_window)
        # Created lazily so a decoder is never opened in the parent process
        # before DataLoader workers are forked.
        self._decoder_cache: _TorchCodecDecoderCache | None = None
        self.tasks = read_teledata_tasks(self.root)
        self.text_cache = (
            TextEmbeddingCache(text_cache_dir, context_len) if text_cache_dir else None
        )

        self.episodes = self._read_episodes()
        self._load_frame_columns()
        max_future = max(0, *self._video_offsets, *self._action_offsets)
        self._episode_sizes: list[int] = []
        self._cumulative_sizes: list[int] = []
        running = 0
        for episode in self.episodes:
            size = (
                max(0, int(episode["length"]) - max_future)
                if self.require_full_future_window
                else int(episode["length"])
            )
            self._episode_sizes.append(size)
            running += size
            self._cumulative_sizes.append(running)

        if normalization_stats_path is None:
            raise ValueError("TeleData training requires --dataset-stats")
        self.normalizer = ActionStateNormalizer.load(normalization_stats_path)
        requested_modes = {
            "action": action_normalization.upper(),
            "state": state_normalization.upper(),
        }
        loaded_modes = {
            "action": self.normalizer.action_mode,
            "state": self.normalizer.state_mode,
        }
        if loaded_modes != requested_modes:
            raise ValueError(
                f"Normalization modes in stats {loaded_modes} do not match {requested_modes}"
            )
        if self.normalizer.clip_values != bool(clip_normalized):
            raise ValueError(
                f"Stats clip_values={self.normalizer.clip_values} does not match "
                f"clip_normalized={bool(clip_normalized)}"
            )
        expected_shapes = {
            action_key: (len(self.action_indices),),
            state_key: (len(self.state_indices),),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in self.normalizer.features:
                raise KeyError(f"Normalization stats do not contain {key!r}")
            if tuple(self.normalizer.features[key].shape) != expected_shape:
                raise ValueError(
                    f"Normalization stats for {key!r} have shape "
                    f"{self.normalizer.features[key].shape}, expected {expected_shape}"
                )

    def _read_episodes(self) -> list[dict[str, int | list[str]]]:
        path = self.root / "meta" / "episodes.jsonl"
        records: list[dict[str, int | list[str]]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records.append(
                    {
                        "episode_index": int(record["episode_index"]),
                        "length": int(record["length"]),
                        "tasks": record.get("tasks", []),
                    }
                )
        records.sort(key=lambda item: int(item["episode_index"]))
        expected = list(range(len(records)))
        actual = [int(item["episode_index"]) for item in records]
        if actual != expected:
            raise ValueError(f"Episode indices must be contiguous from zero, got {actual[:8]}...")
        return records

    def _load_frame_columns(self) -> None:
        state_width = int(self.info["features"][self.state_key]["shape"][0])
        action_width = int(self.info["features"][self.action_key]["shape"][0])
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        task_indices: list[np.ndarray] = []
        expected_total = 0
        chunks_size = int(self.info.get("chunks_size", 1000))
        for episode in self.episodes:
            episode_index = int(episode["episode_index"])
            path = (
                self.root
                / "data"
                / f"chunk-{episode_index // chunks_size:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing episode parquet: {path}")
            table = pq.read_table(
                path,
                columns=[self.state_key, self.action_key, "timestamp", "task_index"],
            )
            if table.num_rows != int(episode["length"]):
                raise ValueError(
                    f"Episode {episode_index} length mismatch: metadata={episode['length']} "
                    f"parquet={table.num_rows}"
                )
            states.append(_fixed_list_array(table[self.state_key], state_width, self.state_key))
            actions.append(_fixed_list_array(table[self.action_key], action_width, self.action_key))
            timestamps.append(
                np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float32)
            )
            task_indices.append(
                np.asarray(table["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
            )
            expected_total += table.num_rows
        if expected_total != int(self.info["total_frames"]):
            raise ValueError(
                f"Frame count mismatch: metadata={self.info['total_frames']} loaded={expected_total}"
            )
        self._states = torch.from_numpy(np.concatenate(states, axis=0))
        self._actions = torch.from_numpy(np.concatenate(actions, axis=0))
        self._timestamps = np.concatenate(timestamps, axis=0)
        self._task_indices = np.concatenate(task_indices, axis=0)
        starts: list[int] = []
        running = 0
        for episode in self.episodes:
            starts.append(running)
            running += int(episode["length"])
        self._episode_starts = starts

        # Verify all expected video paths before model initialization so a
        # missing file cannot surface only inside a worker after a long wait.
        for episode in self.episodes:
            index = int(episode["episode_index"])
            for key in self.video_keys:
                path = self._video_path(key, index)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing episode video: {path}")

    @property
    def action_dim(self) -> int:
        return len(self.action_indices)

    @property
    def state_dim(self) -> int:
        return len(self.state_indices)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def video_offsets(self) -> tuple[int, ...]:
        return self._video_offsets

    @property
    def action_offsets(self) -> tuple[int, ...]:
        return self._action_offsets

    @property
    def state_offsets(self) -> tuple[int, ...]:
        return self._state_offsets

    @property
    def sampling_num_frames(self) -> int:
        return self._sampling_num_frames

    @property
    def past_frames(self) -> int:
        return self._past_frames

    @property
    def video_stride(self) -> int:
        return self._video_stride

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.normalizer.unnormalize_action(action, self.action_key)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def _anchor(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._cumulative_sizes, index)
        previous = 0 if episode_index == 0 else self._cumulative_sizes[episode_index - 1]
        return episode_index, self._episode_starts[episode_index] + index - previous

    @staticmethod
    def _clamped_indices(anchor: int, offsets: Sequence[int], start: int, end: int):
        indices = [max(start, min(end - 1, anchor + offset)) for offset in offsets]
        is_pad = torch.tensor(
            [anchor + offset < start or anchor + offset >= end for offset in offsets],
            dtype=torch.bool,
        )
        return indices, is_pad

    def _video_path(self, key: str, episode_index: int) -> Path:
        chunks_size = int(self.info.get("chunks_size", 1000))
        chunk = episode_index // chunks_size
        return self.root / self.info["video_path"].format(
            episode_chunk=chunk,
            chunk_index=chunk,
            file_index=episode_index,
            episode_index=episode_index,
            video_key=key,
        )

    def _decode_view(
        self, key: str, episode_index: int, timestamps: Sequence[float]
    ) -> torch.Tensor:
        path = self._video_path(key, episode_index)
        if self.video_backend == "torchcodec":
            if self._decoder_cache is None:
                self._decoder_cache = _TorchCodecDecoderCache()
            return self._decoder_cache.decode(path, timestamps, tolerance_s=1e-3)
        return decode_video_frames(
            path,
            [float(timestamp) for timestamp in timestamps],
            tolerance_s=1e-3,
            backend=self.video_backend,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        episode_index, anchor = self._anchor(index)
        episode = self.episodes[episode_index]
        start = self._episode_starts[episode_index]
        end = start + int(episode["length"])
        video_indices, image_is_pad = self._clamped_indices(
            anchor, self._video_offsets, start, end
        )
        action_indices, action_is_pad = self._clamped_indices(
            anchor, self._action_offsets, start, end
        )
        state_indices, proprio_is_pad = self._clamped_indices(
            anchor, self._state_offsets, start, end
        )

        action = self._actions[action_indices].index_select(-1, self._action_index)
        state = self._states[state_indices].index_select(-1, self._state_index)
        if self.relative_action_indices:
            action = _make_relative_action(action, state, self._relative_action_index)
        normalized = self.normalizer({self.action_key: action, self.state_key: state})
        state = normalized[self.state_key][0]

        timestamps = self._timestamps[video_indices].tolist()
        clips = []
        for key in self.video_keys:
            clip = self._decode_view(key, int(episode["episode_index"]), timestamps)
            if self.video_transform is not None:
                clip = self.video_transform(clip)
            clips.append(clip)
        video = combine_video_clips(clips, self.view_combiner)

        task_index = int(self._task_indices[anchor])
        try:
            instruction = self.tasks[task_index]
        except KeyError as error:
            raise KeyError(f"No instruction for task_index={task_index}") from error
        prompt = self.prompt_template.format(task=instruction)
        result = {
            "video": video,
            "action": normalized[self.action_key],
            "proprio": state,
            "prompt": prompt,
            "task": instruction,
            "task_id": instruction,
            "instruction": instruction,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "image_is_pad": image_is_pad,
            "index": index,
            "dataset_index": 0,
        }
        if self.text_cache is not None:
            result["context"], result["context_mask"] = self.text_cache.load(prompt)
        return result
