"""LeRobot v3 clip reader for shared parquet and video files.

LeRobot 0.3.x only understands the v2.1 layout used by the existing policy
datasets.  This reader handles the v3 layout without changing the source
dataset or the verified v2.1 loader.
"""

import bisect
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset as ArrowDataset
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from vjepa_policy.datasets.lerobot_video import (
    DEFAULT_PROMPT,
    TextEmbeddingCache,
    _make_relative_action,
    _validate_feature_indices,
    discover_video_keys,
    make_video_offsets,
)
from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.datasets.transforms import combine_video_clips


def read_v3_tasks(dataset_dir: str | Path) -> dict[int, str]:
    """Read the task-index-to-instruction mapping from ``tasks.parquet``."""
    path = Path(dataset_dir) / "meta" / "tasks.parquet"
    table = pq.read_table(path)
    if "task_index" not in table.column_names:
        raise ValueError(f"Missing task_index in {path}")
    text_columns = [name for name in table.column_names if name != "task_index"]
    if len(text_columns) != 1:
        raise ValueError(f"Expected one task text column in {path}, got {text_columns}")
    indices = table["task_index"].to_pylist()
    instructions = table[text_columns[0]].to_pylist()
    result = {
        int(index): instruction.strip()
        for index, instruction in zip(indices, instructions, strict=True)
        if isinstance(instruction, str) and instruction.strip()
    }
    if len(result) != len(indices):
        raise ValueError(f"Every task in {path} must have a non-empty instruction")
    return result


def _read_episode_metadata(root: Path, video_keys: Sequence[str]) -> list[dict]:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 episode metadata under {root / 'meta' / 'episodes'}")
    columns = ["episode_index", "dataset_from_index", "dataset_to_index", "length"]
    for key in video_keys:
        columns.extend(
            [
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
            ]
        )
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    payload = table.to_pydict()
    episodes = [
        {name: values[row] for name, values in payload.items()}
        for row in range(table.num_rows)
    ]
    episodes.sort(key=lambda episode: int(episode["episode_index"]))
    expected = list(range(len(episodes)))
    actual = [int(episode["episode_index"]) for episode in episodes]
    if actual != expected:
        raise ValueError(f"Episode indices must be contiguous from zero, got {actual[:8]}...")
    return episodes


class LeRobotVideoV3Dataset(Dataset):
    """Return VJEPA-Policy samples from one local LeRobot v3 dataset."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        video_keys: Sequence[str] | None = None,
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
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(
                f"Expected LeRobot v3.0 at {self.root}, got "
                f"{self.info.get('codebase_version')!r}"
            )
        self.fps = int(self.info["fps"])
        self.video_keys = list(video_keys or discover_video_keys([self.root]))
        features = self.info["features"]
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
        self.tasks = read_v3_tasks(self.root)
        self.text_cache = (
            TextEmbeddingCache(text_cache_dir, context_len) if text_cache_dir else None
        )

        data_paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not data_paths:
            raise FileNotFoundError(f"No v3 frame parquet files under {self.root / 'data'}")
        self.frames = ArrowDataset.from_parquet([str(path) for path in data_paths])
        self.episodes = _read_episode_metadata(self.root, self.video_keys)
        if len(self.frames) != int(self.info["total_frames"]):
            raise ValueError(
                f"Frame count mismatch: metadata={self.info['total_frames']} "
                f"parquet={len(self.frames)}"
            )

        max_future = max(0, *self._video_offsets, *self._action_offsets)
        self._episode_sizes = []
        self._cumulative_sizes = []
        running = 0
        for episode in self.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            length = int(episode["length"])
            if end - start != length:
                raise ValueError(
                    f"Episode {episode['episode_index']} has inconsistent bounds and length"
                )
            size = max(0, length - max_future) if require_full_future_window else length
            self._episode_sizes.append(size)
            running += size
            self._cumulative_sizes.append(running)

        if normalization_stats_path is None:
            raise ValueError("LeRobot v3 training requires --dataset-stats")
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
        start = int(self.episodes[episode_index]["dataset_from_index"])
        return episode_index, start + index - previous

    @staticmethod
    def _clamped_indices(anchor: int, offsets: Sequence[int], start: int, end: int):
        indices = [max(start, min(end - 1, anchor + offset)) for offset in offsets]
        is_pad = torch.tensor(
            [anchor + offset < start or anchor + offset >= end for offset in offsets],
            dtype=torch.bool,
        )
        return indices, is_pad

    def _decode_view(
        self,
        key: str,
        episode: dict,
        local_timestamps: Sequence[float],
    ) -> torch.Tensor:
        chunk = int(episode[f"videos/{key}/chunk_index"])
        file_index = int(episode[f"videos/{key}/file_index"])
        from_timestamp = float(episode[f"videos/{key}/from_timestamp"])
        relative_path = self.info["video_path"].format(
            video_key=key,
            chunk_index=chunk,
            file_index=file_index,
        )
        clip = decode_video_frames(
            self.root / relative_path,
            [from_timestamp + float(timestamp) for timestamp in local_timestamps],
            tolerance_s=1e-4,
            backend=self.video_backend,
        )
        return self.video_transform(clip) if self.video_transform is not None else clip

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        episode_index, anchor = self._anchor(index)
        episode = self.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        video_indices, image_is_pad = self._clamped_indices(
            anchor, self._video_offsets, start, end
        )
        action_indices, action_is_pad = self._clamped_indices(
            anchor, self._action_offsets, start, end
        )
        state_indices, proprio_is_pad = self._clamped_indices(
            anchor, self._state_offsets, start, end
        )

        action_rows = self.frames[action_indices][self.action_key]
        state_rows = self.frames[state_indices][self.state_key]
        action = torch.tensor(action_rows, dtype=torch.float32).index_select(
            -1, self._action_index
        )
        state = torch.tensor(state_rows, dtype=torch.float32).index_select(
            -1, self._state_index
        )
        if self.relative_action_indices:
            action = _make_relative_action(action, state, self._relative_action_index)
        normalized = self.normalizer({self.action_key: action, self.state_key: state})
        state = normalized[self.state_key][0]

        timestamps = self.frames[video_indices]["timestamp"]
        clips = [self._decode_view(key, episode, timestamps) for key in self.video_keys]
        video = combine_video_clips(clips, self.view_combiner)

        current = self.frames[anchor]
        task_index = int(current["task_index"])
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
