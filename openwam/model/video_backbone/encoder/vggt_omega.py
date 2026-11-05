"""Frozen native VGGT-Omega spatial patches, joint stereo independently per time.

M29 spatial projection and target normalization reuse the existing mechanism.
History and current are two clean positions, followed by eight noisy future
positions. Registers/camera tokens stay inside the untouched native aggregator.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import VideoEncoder, VideoEncoderProperties
from .bottleneck import BottleneckPatchEmbed
from .registry import register_video_encoder


def _build_aggregator():
    vendor = Path(__file__).resolve().parents[4] / "third_party/vggt_omega"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from vggt_omega.models.aggregator import Aggregator
    return Aggregator()


@register_video_encoder("vggt_omega")
class VGGTOmegaSpatialEncoder(VideoEncoder):
    """No temporal fusion, learned feature adapters, or explicit register outputs."""

    def __init__(self, aggregator, *, pair_microbatch_size=1, bottleneck_dim=256,
                 feature_norm_eps=1e-5, image_size=224):
        super().__init__()
        if isinstance(pair_microbatch_size, bool) or pair_microbatch_size < 1:
            raise ValueError("pair_microbatch_size must be positive")
        if image_size < 32 or image_size % 32:
            raise ValueError("image_size must be divisible by native patch16 and spatial patch2")
        self.image_size = int(image_size)
        self._m = aggregator
        dim = 2 * int(aggregator.frame_blocks[0].norm1.normalized_shape[0])
        if aggregator.patch_size != 16 or aggregator.patch_token_start != 17:
            raise ValueError("E05 requires the native patch16, camera + 16-register layout")
        self.pair_microbatch_size = int(pair_microbatch_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.feature_norm_eps = float(feature_norm_eps)
        self.feature_cache = None
        self._out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=feature_norm_eps)
        self._spec = VideoEncoderProperties(
            z_dim=dim, spatial_compression=16, temporal_compression=1,
            causal_temporal=True, pixel_decode=False, dit_patch_size=(1, 2, 2),
            num_condition_latents=2,
        )
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @property
    def properties(self):
        return self._spec

    @property
    def add_one_history_context(self):
        return True

    @property
    def seperate_encode_view(self):
        # Host flag means two distinct spatial grids/view embeddings, not that
        # the native aggregator encodes views independently.
        return True

    @property
    def processing_config(self):
        return dict(joint_views_per_timestamp=True, cross_time_encoding=False,
                    add_one_history_context=True, spatial_patch_size=2, image_size=self.image_size,
                    view_order=["front", "wrist"], output_layout="width_concat",
                    pair_microbatch_size=self.pair_microbatch_size,
                    bottleneck_dim=self.bottleneck_dim, feature_norm_eps=self.feature_norm_eps,
                    output="final_spatial_patches_only", input_range="0_1_native_normalization")

    def build_dit_input_proj(self, dit_dim):
        return BottleneckPatchEmbed(self.properties.z_dim, dit_dim, (1, 2, 2), self.bottleneck_dim)

    def configure_feature_cache(self, directory, checkpoint_file):
        from .vggt_feature_cache import StereoFeatureCache, checkpoint_digest
        self.feature_cache = StereoFeatureCache(directory, dict(
            schema=1, checkpoint_sha256=checkpoint_digest(checkpoint_file),
            processing=self.processing_config,
            normalization="native_imagenet_once_then_frozen_nonaffine_layernorm",
            context="ordered_front_wrist_at_same_timestamp_only",
            key="exact_preprocessed_rgb_content_shape_dtype",
        ))

    def preprocess_video(self, frames):
        parameter = next(self._m.parameters())
        images = []
        for image in frames:
            if image.size != (2 * self.image_size, self.image_size):
                raise ValueError(f"E05 requires front/wrist views each {self.image_size} square")
            images.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
        # Native Aggregator performs its own ImageNet normalization exactly once.
        return torch.stack(images, dim=1).unsqueeze(0).to(parameter.device, parameter.dtype).div_(255)

    def preprocess_video_batch(self, videos):
        return torch.cat([self.preprocess_video(video) for video in videos])

    @torch.no_grad()
    def batch_encode(self, video):
        size = self.image_size
        grid = size // 16
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2:] != (size, 2 * size):
            raise ValueError(f"Expected [B,3,T,{size},{2 * size}]")
        b, _, t, _, _ = video.shape
        if t < 1:
            raise ValueError("E05 requires at least the current observation")
        # Each native sequence is precisely [front_t, wrist_t]. Time becomes
        # an independent batch axis, never the native inter-frame sequence axis.
        pairs = video.reshape(b, 3, t, size, 2, size).permute(0, 2, 4, 1, 3, 5)
        pairs = pairs.reshape(b * t, 2, 3, size, size)
        if self.feature_cache is None:
            patches = torch.cat([self._encode_pairs(chunk) for chunk in pairs.split(self.pair_microbatch_size)])
        else:
            # Cache units are independent timestamps, so overlapping windows reuse
            # the very same frozen stereo features. Deployment leaves this off.
            patches = torch.cat([self.feature_cache.get_or_encode(pair.unsqueeze(0), self._encode_pairs)
                                 for pair in pairs])
        # Keep the native row-major patch order; concatenate camera grids in width.
        return patches.reshape(b, t, 2, grid, grid, -1).permute(0, 5, 1, 3, 2, 4).reshape(
            b, self.properties.z_dim, t, grid, 2 * grid)

    def _encode_pairs(self, pairs):
        layers, start = self._m(pairs)
        if start != 17 or layers[-1] is None:
            raise ValueError("Native aggregator final layer/prefix contract changed")
        patches = layers[-1][:, :, start:]
        expected = (len(pairs), 2, (self.image_size // 16) ** 2, self.properties.z_dim)
        if tuple(patches.shape) != expected:
            raise ValueError(f"Spatial patch shape {patches.shape} differs from {expected}")
        return self._out_norm(patches)

    @classmethod
    def from_pretrained(cls, model_path, *, checkpoint="vggt_omega_1b_512.pt",
                        pair_microbatch_size=1, bottleneck_dim=256, feature_norm_eps=1e-5,
                        seperate_encode_view=True, add_one_history_context=True, add_input_bottleneck=True,
                        image_size=224, feature_cache_dir=None):
        from accelerate import init_empty_weights
        if (seperate_encode_view is not True or add_one_history_context is not True
                or add_input_bottleneck is not True):
            raise ValueError("E05 requires distinct camera grids and history/current clean layout")
        with init_empty_weights():
            aggregator = _build_aggregator()
        state = torch.load(Path(model_path) / checkpoint, weights_only=True, map_location="cpu", mmap=True)
        state = {key.removeprefix("aggregator."): value.to(torch.bfloat16)
                 for key, value in state.items() if key.startswith("aggregator.")}
        if not state:
            raise ValueError("Checkpoint has no native aggregator weights")
        aggregator.load_state_dict(state, strict=True, assign=True)
        # Non-persistent native normalization/RoPE buffers remain CPU-resident.
        instance = cls(aggregator.to(dtype=torch.bfloat16), pair_microbatch_size=pair_microbatch_size,
                   bottleneck_dim=bottleneck_dim, feature_norm_eps=feature_norm_eps, image_size=image_size)
        if feature_cache_dir:
            instance.configure_feature_cache(feature_cache_dir, Path(model_path) / checkpoint)
        return instance

    def save_deploy_assets(self, output_dir, cfg):
        directory = Path(output_dir) / "vggt_omega"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "encoder.json").write_text(json.dumps(dict(
            processing=self.processing_config, channels=self.properties.z_dim,
            native_patch_size=16, camera_tokens=1, register_tokens=16,
        ), indent=2) + "\n")

    @classmethod
    def from_skeleton(cls, components_entry, *, device="cpu", encoder_cfg=None, ckpt_dir=None):
        if ckpt_dir is None or encoder_cfg is None:
            raise ValueError("E05 requires checkpoint-local sidecar and encoder config")
        if any(encoder_cfg.get(key) is not True for key in (
                "add_one_history_context", "seperate_encode_view", "add_input_bottleneck")):
            raise ValueError("E05 deployment requires history/current clean and the original spatial projection")
        saved = json.loads((Path(ckpt_dir) / "vggt_omega/encoder.json").read_text())
        with torch.device(device):
            instance = cls(_build_aggregator(), **{key: encoder_cfg.get(key, default) for key, default in (
                ("pair_microbatch_size", 1), ("bottleneck_dim", 256), ("feature_norm_eps", 1e-5), ("image_size", 224))})
        if saved["processing"] != instance.processing_config or saved["channels"] != instance.properties.z_dim:
            raise ValueError("E05 processing differs from checkpoint sidecar")
        return instance.to(dtype=torch.bfloat16)
