"""Read-only, mmap-backed native Wan window latents; projections are never cached."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def cache_identity(config, encoder_config, length):
    """Lock pixel preprocessing, temporal reset boundaries, source data and weights."""
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    data.pop("latent_cache_dir", None)
    data.pop("seed", None)
    if (data.get("type") != "libero" or data.get("color_jitter", {}).get("enabled")
            or data.get("video_stride") != 1 or data.get("num_frames") != 33
            or data.get("history_num_tubelets", 0) or data.get("clean_condition_frame_offsets") is not None
            or not data.get("seperate_encode_view")):
        raise ValueError("Wan cache requires deterministic LIBERO, 33 dense frames, independent views, no history/jitter")
    version = encoder_config.get("vae_version")
    if encoder_config.get("name") != "wan_vae" or version not in ("2.1", "2.2"):
        raise ValueError("This window cache supports Wan2.1/2.2 VAE only")
    root = Path(data["dataset_dir"])
    metadata = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "meta").rglob("*"))
                if p.is_file() and p.suffix in (".json", ".jsonl", ".npy", ".parquet")}
    files = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
             for directory in ("data", "videos") for p in sorted((root / directory).rglob("*")) if p.is_file()}
    source_root = Path(__file__).resolve().parents[1]
    code = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in (
        "dataloader/libero.py", "dataloader/bases/lerobot_v3_reader.py",
        "model/video_backbone/encoder/wan_vae.py", "model/video_backbone/wan/models/vae.py")}
    h = hashlib.sha256()
    weights = Path(encoder_config["model_path"]) / f"Wan{version}_VAE.pth"
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return dict(schema=1, data=data, length=length, metadata_sha256=metadata, media_inventory=files,
                code_sha256=code, weights_sha256=h.hexdigest(), vae_version=version,
                normalization="native_vae_mean_std", context="window_reset_33_frames",
                views="independent_then_width_concat", dtype="bfloat16",
                shape=[16, 9, 28, 56] if version == "2.1" else [48, 9, 14, 28], shard_size=256)


class WanLatentCache:
    def __init__(self, directory, identity=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if identity is not None and self.manifest != identity:
            keys = [k for k in identity if identity[k] != self.manifest.get(k)]
            raise ValueError("Wan latent cache identity mismatch: " + ", ".join(keys))

    def require_complete(self):
        size, count = self.manifest["shard_size"], self.manifest["length"]
        for start in range(0, count, size):
            path = self.directory / f"shard_{start // size:06d}.safetensors"
            with safe_open(path, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {str(i) for i in range(start, min(start + size, count))}:
                    raise ValueError(f"Incomplete cache shard {path}; complete the offline build before formal training")

    def read(self, index):
        path = self.directory / f"shard_{index // self.manifest['shard_size']:06d}.safetensors"
        # Independent mmap handle per call; safe across fork/spawn workers, no whole-shard load.
        with safe_open(path, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(str(index)).clone()
        if list(value.shape) != self.manifest["shape"] or value.dtype != torch.bfloat16:
            raise ValueError(f"Invalid cached latent at index {index}")
        return value


def attach_cache(dataset, config):
    from omegaconf import OmegaConf
    directory = config.get("latent_cache_dir")
    cache = WanLatentCache(directory)
    data = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    data.pop("latent_cache_dir", None)
    data.pop("seed", None)
    if data != cache.manifest["data"] or len(dataset) != cache.manifest["length"]:
        raise ValueError("Wan cache dataset/config differs from current reader")
    dataset._wan_latent_cache = cache
