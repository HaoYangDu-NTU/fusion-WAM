"""Add native SigLIP caches without changing existing DINO/I-JEPA identities."""
import hashlib
from pathlib import Path

from omegaconf import OmegaConf
from openwam.dataloader.frame_cache_specs import frame_cache_identity as existing_identity
from openwam.dataloader.image_frame_cache import frame_cache_identity as legacy_identity


def frame_cache_identity(config, encoder_config, reader):
    if encoder_config["name"] != "siglip":
        return existing_identity(config, encoder_config, reader)
    if config["type"] != "libero_native_stereo" or config.get("native_view_size") != 256:
        raise ValueError("SigLIP cache requires the native 256 stereo reader")
    # Reuse original dataset/media/weight hashing and deterministic-window guards.
    data = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    data.type = "libero"
    encoder = OmegaConf.create(OmegaConf.to_container(encoder_config, resolve=True))
    encoder.name = "dinov3"
    encoder.use_avg_t_pooling = False
    identity = legacy_identity(data, encoder, reader)
    identity["encoder"] = "siglip"
    identity["data"]["type"] = "libero_native_stereo"
    identity["shape"] = [1024, 16, 32]
    identity["backbone_config"] = dict(name="siglip", view_size=256, patch_size=16, pooling_head=False)
    identity["stage"] = "frozen_spatial_patches_after_native_post_layernorm_before_fixed_norm_and_projection"
    identity["code"].pop("model/video_backbone/encoder/dinov3.py")
    source = Path(__file__).resolve().parents[1]
    for name in ("dataloader/spatial_frame_cache_specs.py", "dataloader/libero_native_stereo.py",
                 "dataloader/transforms/native_stereo.py", "model/video_backbone/encoder/siglip.py"):
        identity["code"][name] = hashlib.sha256((source/name).read_bytes()).hexdigest()
    return identity
