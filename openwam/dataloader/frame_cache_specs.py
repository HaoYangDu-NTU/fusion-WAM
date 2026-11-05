"""Native DINO widths while preserving existing E01/E02 cache identities."""
import hashlib
import json
from pathlib import Path

from openwam.dataloader.image_frame_cache import frame_cache_identity as legacy_identity


def frame_cache_identity(config, encoder_config, reader):
    identity = legacy_identity(config, encoder_config, reader)
    if encoder_config["name"] != "dinov3":
        return identity
    metadata = json.loads((Path(encoder_config["model_path"])/"config.json").read_text())
    width = metadata["hidden_size"]
    if metadata["patch_size"] != 16 or not isinstance(width, int) or width <= 0:
        raise ValueError("Native DINO frame cache requires a positive width and patch_size=16")
    if width != 1280:
        identity["shape"] = [width, 14, 28]
        identity["code"]["dataloader/frame_cache_specs.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return identity
