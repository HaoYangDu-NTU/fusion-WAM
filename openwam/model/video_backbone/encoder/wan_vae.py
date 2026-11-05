"""Wan2.1/2.2 VAE encoder only: native causal video latents, no DiT/CLIP/T5."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import VideoEncoder, VideoEncoderProperties
from .bottleneck import BottleneckPatchEmbed
from .registry import register_video_encoder


def build_vae(version):
    from ..wan.models.vae import WanVideoVAE, WanVideoVAE38
    if version not in ("2.1", "2.2"):
        raise ValueError("vae_version must be '2.1' or '2.2'")
    vae = WanVideoVAE() if version == "2.1" else WanVideoVAE38()
    # The released checkpoint contains both directions. Only these two
    # encoder-side modules are loaded, saved and moved to the GPU.
    vae.model.decoder = nn.Identity()
    vae.model.conv2 = nn.Identity()
    return vae


@register_video_encoder("wan_vae")
class WanVAEFeatureEncoder(VideoEncoder):
    seperate_encode_view = True

    def __init__(self, vae, *, vae_version, microbatch_size=1, bottleneck_dim=256, spatial_patch_size=2):
        super().__init__()
        if vae_version not in ("2.1", "2.2"):
            raise ValueError("Unsupported VAE version")
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        self._m = vae
        self.vae_version = vae_version
        self.microbatch_size = microbatch_size
        self.bottleneck_dim = bottleneck_dim
        if spatial_patch_size not in (2, 4) or (224 // vae.upsampling_factor) % spatial_patch_size:
            raise ValueError("spatial_patch_size must be 2 or 4 and divide each view's latent grid")
        self.spatial_patch_size = spatial_patch_size
        self._spec = VideoEncoderProperties(z_dim=vae.z_dim, spatial_compression=vae.upsampling_factor,
                                            temporal_compression=4, causal_temporal=True,
                                            pixel_decode=False, dit_patch_size=(1, spatial_patch_size, spatial_patch_size))
        self.requires_grad_(False)
        self.train(False)

    @property
    def properties(self):
        return self._spec

    def train(self, mode=True):
        return super().train(False)

    @property
    def processing_config(self):
        return {"vae_version": self.vae_version, "microbatch_size": self.microbatch_size,
                "spatial_patch_size": self.spatial_patch_size,
                "bottleneck_dim": self.bottleneck_dim, "seperate_encode_view": True,
                "encoder_only": True, "normalization": "native_vae_mean_std", "temporal_compression": 4}

    def build_dit_input_proj(self, dit_dim):
        return BottleneckPatchEmbed(self.properties.z_dim, dit_dim, self.properties.dit_patch_size, self.bottleneck_dim)

    def preprocess_video(self, frames):
        parameter = next(self._m.parameters())
        images = []
        for image in frames:
            if image.size != (448, 224):
                raise ValueError("Wan VAE expects two 224x224 views packed as 224x448")
            images.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
        return torch.stack(images, dim=1).unsqueeze(0).to(parameter.device, parameter.dtype).div_(127.5).sub_(1)

    def preprocess_video_batch(self, videos):
        return torch.cat([self.preprocess_video(clip) for clip in videos])

    @torch.no_grad()
    def batch_encode(self, video):
        from openwam.train.utils.step_timing import phase
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2:] != (224, 448):
            raise ValueError("Expected [B,3,T,224,448] packed two-view video")
        if video.shape[2] < 1 or (video.shape[2] - 1) % 4:
            raise ValueError("Native Wan VAE requires 4k+1 consecutive frames")
        batch = video.shape[0]
        views = torch.stack(video.split(224, dim=-1), dim=1).flatten(0, 1)
        with phase("vae_encode"):
            features = torch.cat([self._m.batch_encode(chunk, device=video.device)
                                  for chunk in views.split(self.microbatch_size)])
        features = features.unflatten(0, (batch, 2))
        return torch.cat([features[:, 0], features[:, 1]], dim=-1)

    @classmethod
    def from_pretrained(cls, model_path, *, vae_version, microbatch_size=1, bottleneck_dim=256,
                        seperate_encode_view=True, add_input_bottleneck=True, spatial_patch_size=2):
        from accelerate import init_empty_weights
        if seperate_encode_view is not True or add_input_bottleneck is not True:
            raise ValueError("Wan VAE comparison requires independent views and clean/future projections")
        source = Path(model_path) / f"Wan{vae_version}_VAE.pth"
        with init_empty_weights():
            vae = build_vae(vae_version)
        raw = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        raw = raw.get("model_state", raw)
        state = {"model." + key: value.to(torch.bfloat16) for key, value in raw.items()
                 if key.startswith(("encoder.", "conv1."))}
        vae.load_state_dict(state, strict=True, assign=True)
        return cls(vae, vae_version=vae_version, microbatch_size=microbatch_size,
                   bottleneck_dim=bottleneck_dim, spatial_patch_size=spatial_patch_size)

    def save_deploy_assets(self, output_dir, cfg):
        directory = Path(output_dir) / "wan_vae"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "encoder.json").write_text(json.dumps({"processing": self.processing_config,
                                                           "z_dim": self.properties.z_dim,
                                                           "spatial_compression": self.properties.spatial_compression}, indent=2) + "\n")

    @classmethod
    def from_skeleton(cls, components_entry, *, device="cpu", encoder_cfg=None, ckpt_dir=None):
        if ckpt_dir is None:
            raise ValueError("Wan VAE requires checkpoint-local wan_vae/encoder.json")
        saved = json.loads((Path(ckpt_dir) / "wan_vae/encoder.json").read_text())
        processing = saved["processing"]
        processing.setdefault("spatial_patch_size", 2)
        if not processing.get("encoder_only") or processing.get("normalization") != "native_vae_mean_std":
            raise ValueError("Invalid Wan VAE encoder-only contract")
        for key in ("vae_version", "microbatch_size", "bottleneck_dim"):
            if encoder_cfg is not None and encoder_cfg.get(key) != processing[key]:
                raise ValueError(f"Wan VAE processing mismatch: {key}")
        if encoder_cfg is not None and encoder_cfg.get("spatial_patch_size", 2) != processing["spatial_patch_size"]:
            raise ValueError("Wan VAE processing mismatch: spatial_patch_size")
        with torch.device(device):
            encoder = cls(build_vae(processing["vae_version"]),
                          **{k: processing[k] for k in ("vae_version", "microbatch_size", "bottleneck_dim", "spatial_patch_size")})
        if saved["z_dim"] != encoder.properties.z_dim or saved["spatial_compression"] != encoder.properties.spatial_compression:
            raise ValueError("Wan VAE saved geometry differs from encoder version")
        return encoder.to(dtype=torch.bfloat16)
