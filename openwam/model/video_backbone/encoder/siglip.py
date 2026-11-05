"""Frozen SigLIP spatial patches, independently encoded stereo timestamps."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from einops import rearrange

from .base import VideoEncoder, VideoEncoderProperties
from .bottleneck import BottleneckPatchEmbed
from .registry import register_video_encoder


@register_video_encoder("siglip")
class SiglipVideoEncoder(VideoEncoder):
    def __init__(self, model, *, bottleneck_dim=256, feature_norm_eps=1e-5,
                 seperate_encode_view=True, add_one_history_context=True):
        super().__init__()
        config = model.config
        if ((config.hidden_size, config.image_size, config.patch_size) != (1024, 256, 16)
                or model.use_head or seperate_encode_view is not True or add_one_history_context is not True):
            raise ValueError("E08 requires headless SigLIP-L/16 at its native 256 resolution")
        self._m = model.requires_grad_(False).eval()
        self._raw_embed_dim = config.hidden_size
        self.view_size = config.image_size
        self._spec = VideoEncoderProperties(z_dim=config.hidden_size, spatial_compression=16,
            temporal_compression=1, causal_temporal=True, pixel_decode=False,
            dit_patch_size=(1, 2, 2), num_condition_latents=2)
        self._processing = dict(bottleneck_dim=bottleneck_dim, feature_norm_eps=feature_norm_eps,
            seperate_encode_view=True, add_one_history_context=True)
        self._out_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=feature_norm_eps)
        self._use_avg_t_pooling = False

    @property
    def properties(self):
        return self._spec

    @property
    def seperate_encode_view(self):
        return True

    @property
    def add_one_history_context(self):
        return True

    def train(self, mode=True):
        super().train(mode)
        self._m.eval()
        return self

    def preprocess_video(self, frames):
        anchor = next(self._m.parameters())
        tensors = []
        for image in frames:
            if image.size != (512, 256):
                raise ValueError("SigLIP transport must contain native 256 front/wrist images")
            pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32, copy=True))/255
            # Match the official processor in FP32, then cast for the frozen vision tower.
            tensors.append(((pixels-0.5)/0.5).permute(2, 0, 1))
        return torch.stack(tensors, dim=1).unsqueeze(0).to(device=anchor.device, dtype=anchor.dtype)

    def preprocess_video_batch(self, videos):
        return torch.cat([self.preprocess_video(v) for v in videos])

    @torch.no_grad()
    def _encode_frames(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2:] != (256, 512):
            raise ValueError("Expected B×3×T×256×512 SigLIP stereo pixels")
        b, _, t, _, _ = video.shape
        images = rearrange(video, "b c t h (v w) -> (b t v) c h w", v=2)
        self._m.eval()
        patches = self._m(pixel_values=images).last_hidden_state
        if patches.shape[1:] != (256, 1024):
            raise ValueError(f"Expected 256 spatial tokens without CLS/registers, got {patches.shape}")
        return rearrange(patches, "(b t v) (h w) d -> b d t h (v w)", b=b, t=t, v=2, h=16, w=16)

    def _apply_feature_norm(self, features):
        return self._out_norm(features.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()

    def batch_encode(self, video):
        return self._apply_feature_norm(self._encode_frames(video))

    def build_dit_input_proj(self, dit_dim):
        return BottleneckPatchEmbed(self._raw_embed_dim, dit_dim, (1, 2, 2), self._processing["bottleneck_dim"])

    @classmethod
    def from_pretrained(cls, model_path, *, bottleneck_dim=256, feature_norm_eps=1e-5,
                        seperate_encode_view=True, add_one_history_context=True):
        from transformers import SiglipVisionModel
        model, info = SiglipVisionModel.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"SigLIP vision weights did not load strictly: {info}")
        return cls(model, bottleneck_dim=bottleneck_dim, feature_norm_eps=feature_norm_eps,
                   seperate_encode_view=seperate_encode_view, add_one_history_context=add_one_history_context)

    def save_deploy_assets(self, output_dir, cfg):
        directory = Path(output_dir)/"siglip"
        directory.mkdir(parents=True, exist_ok=True)
        self._m.config.save_pretrained(directory)
        (directory/"encoder.json").write_text(json.dumps(self._processing, indent=2)+"\n")

    @classmethod
    def from_skeleton(cls, components_entry, *, device="cpu", encoder_cfg=None, ckpt_dir=None):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        directory = Path(ckpt_dir)/"siglip"
        config = SiglipVisionConfig.from_pretrained(directory, local_files_only=True)
        processing = json.loads((directory/"encoder.json").read_text())
        if any(encoder_cfg.get(k) != v for k, v in processing.items()):
            raise ValueError("SigLIP processing config differs from checkpoint sidecar")
        with torch.device(device):
            model = SiglipVisionModel(config)
        return cls(model.to(dtype=torch.bfloat16), **processing)
