"""E01/E02 frozen frame grids, shared across windows; learned adapters stay online."""
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def frame_cache_identity(config, encoder_config, reader):
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(config, resolve=True)
    data.pop("frame_feature_cache_dir", None)
    data.pop("latent_cache_dir", None)
    data.pop("seed", None)
    name = encoder_config["name"]
    if (name not in ("dinov3", "ijepa") or data["type"] != "libero"
            or data.get("color_jitter", {}).get("enabled")
            or data.get("num_frames") != 33 or data.get("video_stride") != 4
            or data.get("window_stride") != 1 or data.get("history_num_tubelets") != 1
            or data.get("clean_condition_frame_offsets") is not None
            or not data.get("seperate_encode_view") or reader._head_camera_choices is not None
            or np.any(reader._ep_valid_start != 0)
            or len(reader) != int(reader._eps_df["length"].sum())):
        raise ValueError("Frame cache requires E01/E02 deterministic, untrimmed LIBERO stereo windows")
    if name == "dinov3" and encoder_config.get("use_avg_t_pooling", True):
        raise ValueError("E01 frame cache requires native unpooled features")
    model_root = Path(encoder_config["model_path"])
    weights = sorted(model_root.glob("*.safetensors")) if name == "dinov3" else [
        model_root / encoder_config.get("checkpoint", "checkpoint.pth.tar")]
    if not weights:
        raise FileNotFoundError(f"No encoder weights in {model_root}")
    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    root = Path(data["dataset_dir"])
    source = Path(__file__).resolve().parents[1]
    files = ["dataloader/image_frame_cache.py", "dataloader/libero.py", "dataloader/bases/lerobot_v3_reader.py",
             "dataloader/transforms/multiview.py", f"model/video_backbone/encoder/{name}.py"]
    return dict(schema=1, encoder=name, data=data, length=len(reader),
                shape=[1280 if name == "dinov3" else 1408, 14, 28], dtype="bfloat16",
                shard_size=256, stage="frozen_backbone_before_temporal_adapter_and_normalization",
                views="independent_front_wrist_width_concat", temporal_offsets=[-4, *range(0, 33, 4)],
                weights={str(p.name): sha(p) for p in weights},
                model_metadata={p.name: sha(p) for p in sorted(model_root.iterdir())
                                if p.is_file() and p.suffix in (".json", ".py")},
                torch_version=str(torch.__version__),
                backbone_config={key: encoder_config.get(key) for key in ("name", "model_name", "img_size", "patch_size")},
                code={p: sha(source / p) for p in files},
                metadata={str(p.relative_to(root)): sha(p) for p in sorted((root / "meta").rglob("*"))
                          if p.is_file() and p.suffix in (".json", ".jsonl", ".npy", ".parquet")},
                media={str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
                       for sub in ("data", "videos") for p in sorted((root / sub).rglob("*")) if p.is_file()})


class ImageFrameCache:
    """Read-only mmap shards; the LRU holds handles, not materialized whole shards."""
    def __init__(self, directory, identity=None, max_open_shards=8):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if identity is not None and self.manifest != identity:
            raise ValueError("Frame cache identity differs from data/encoder/processing")
        self.max_open_shards = max_open_shards
        self._handles = OrderedDict()
        self._pid = os.getpid()

    def close(self):
        for context, _ in self._handles.values():
            context.__exit__(None, None, None)
        self._handles.clear()

    def __getstate__(self):
        return {**self.__dict__, "_handles": OrderedDict(), "_pid": None}

    def read_many(self, indices):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        grouped = {}
        for index in dict.fromkeys(int(i) for i in indices):
            if not 0 <= index < self.manifest["length"]:
                raise IndexError(index)
            grouped.setdefault(index // self.manifest["shard_size"], []).append(index)
        result = {}
        for shard, keys in grouped.items():
            entry = self._handles.pop(shard, None)
            if entry is None:
                context = safe_open(self.directory / f"shard_{shard:06d}.safetensors", framework="pt", device="cpu")
                entry = (context, context.__enter__())
            self._handles[shard] = entry
            while len(self._handles) > self.max_open_shards:
                _, (old_context, _) = self._handles.popitem(last=False)
                old_context.__exit__(None, None, None)
            for key in keys:
                value = entry[1].get_tensor(str(key)).clone()
                if list(value.shape) != self.manifest["shape"] or value.dtype != torch.bfloat16:
                    raise ValueError(f"Invalid cached frame {key}: {value.shape}, {value.dtype}")
                result[key] = value
        return result

    def require_complete(self):
        size, count = self.manifest["shard_size"], self.manifest["length"]
        for start in range(0, count, size):
            with safe_open(self.directory / f"shard_{start // size:06d}.safetensors", framework="pt") as handle:
                if set(handle.keys()) != {str(i) for i in range(start, min(start + size, count))}:
                    raise ValueError("Incomplete frame cache; finish generation before formal training")


def window_frame_indices(reader, episode, offset):
    """Same episode-local history clamp and repeated-last-frame padding as RGB decoding."""
    start = int(reader._cum_n_starts[episode])
    last = int(reader._eps_df.iloc[episode]["length"]) - 1
    return [start + max(0, min(offset + delta, last)) for delta in [-4, *range(0, 33, 4)]]


class ImageFrameCacheCollator:
    def __init__(self, directory):
        self.directory = str(directory)
        self._reader = None

    def __call__(self, samples):
        if self._reader is None:
            self._reader = ImageFrameCache(self.directory)
        keys = [key for sample in samples for key in sample["image_frame_indices"]]
        values = self._reader.read_many(keys)
        for sample in samples:
            sample["image_frame_features"] = torch.stack(
                [values[key] for key in sample.pop("image_frame_indices")], dim=1)
        return samples

    def __getstate__(self):
        return {"directory": self.directory, "_reader": None}


class CachedImageDataset(torch.utils.data.Dataset):
    """Reuse the original action/text hooks while omitting the image decode stage."""
    def __init__(self, dataset, cache):
        self.dataset = dataset
        self._image_frame_cache = cache

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        ds = self.dataset
        episode = int(np.searchsorted(ds._cum_n_starts, index, side="right") - 1)
        offset = int(ds._ep_valid_start[episode]) + index - int(ds._cum_n_starts[episode])
        row = ds._eps_df.iloc[episode]
        actual_len = min(ds._num_frames, int(ds._ep_valid_end[episode]) - offset)
        table = ds._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        win = table.slice(int(ds._ep_data_row_offset[episode]) + offset, actual_len).to_pandas()
        action, action_mask = ds._finalize_action(ds._action_20d(win), actual_len)
        proprio, proprio_mask = ds._finalize_proprio(ds._proprio_20d(win))
        return dict(video=[], vace_video=None, clean_condition_images=None,
                    image_frame_indices=window_frame_indices(ds, episode, offset),
                    action=torch.from_numpy(action), action_mask=torch.from_numpy(action_mask),
                    video_mask=torch.from_numpy(ds._video_sample_indices < actual_len),
                    proprio=torch.from_numpy(proprio).float(), proprio_mask=torch.from_numpy(proprio_mask),
                    prompt=ds._resolve_prompt(row, win))


def attach_frame_cache(dataset, config):
    cache = ImageFrameCache(config["frame_feature_cache_dir"])
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(config, resolve=True)
    for key in ("frame_feature_cache_dir", "latent_cache_dir", "seed"):
        data.pop(key, None)
    if data != cache.manifest["data"] or len(dataset) != cache.manifest["length"]:
        raise ValueError("Frame cache dataset/config differs from current reader")
    return CachedImageDataset(dataset, cache)
