"""E06: frozen native FLUX.2 image latents, independent views and timestamps."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .base import VideoEncoder, VideoEncoderProperties
from .bottleneck import BottleneckPatchEmbed
from .flux2_vae import _read_flux2_vae_config
from .flux2_vae_src import FluxVaeEncoderCore
from .registry import register_video_encoder
from .vggt_feature_cache import StereoFeatureCache, checkpoint_digest


@register_video_encoder("flux2_framewise")
class Flux2FramewiseEncoder(VideoEncoder):
    def __init__(self, core, *, bottleneck_dim=256):
        super().__init__()
        self._core = core
        self.bottleneck_dim = int(bottleneck_dim)
        self.feature_cache = None
        if core.z_dim != 128 or core.spatial_compression != 16:
            raise ValueError("E06 requires native FLUX.2 packed 128D, spatial stride16")
        self._spec = VideoEncoderProperties(
            z_dim=128, spatial_compression=16, temporal_compression=1,
            causal_temporal=True, pixel_decode=False, dit_patch_size=(1, 2, 2),
            num_condition_latents=2,
        )
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Native BN always uses its pretrained running statistics, including
        # when the parent policy recursively switches into training mode.
        return super().train(False)

    @property
    def properties(self):
        return self._spec

    @property
    def add_one_history_context(self):
        return True

    @property
    def seperate_encode_view(self):
        return True

    @property
    def processing_config(self):
        return dict(image_size=224, view_order=["front", "wrist"], output_layout="width_concat",
                    cross_time_encoding=False, joint_views_per_timestamp=False,
                    add_one_history_context=True, temporal_pooling=False,
                    posterior="mean", normalization="native_packing_then_fixed_bn",
                    input_range="minus1_plus1_bf16", encode_batch_size=16,
                    bottleneck_dim=self.bottleneck_dim)

    def build_dit_input_proj(self, dit_dim):
        return BottleneckPatchEmbed(128, dit_dim, (1, 2, 2), self.bottleneck_dim)

    def configure_feature_cache(self, directory, checkpoint_file):
        self.feature_cache = StereoFeatureCache(directory, dict(
            schema=1, checkpoint_sha256=checkpoint_digest(checkpoint_file),
            processing=self.processing_config,
            context="independent_front_and_wrist_images_at_one_timestamp",
            key="exact_preprocessed_rgb_content_shape_dtype",
        ))

    def preprocess_video(self, frames):
        parameter = next(self._core.parameters())
        images = []
        for image in frames:
            if image.size != (448, 224):
                raise ValueError("E06 requires two 224x224 views packed as 224x448")
            images.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
        return (torch.stack(images, dim=1).unsqueeze(0).to(parameter.device, parameter.dtype)
                .mul_(2.0 / 255.0).sub_(1))

    def preprocess_video_batch(self, videos):
        return torch.cat([self.preprocess_video(video) for video in videos])

    @torch.no_grad()
    def _encode_pairs(self, pairs):
        # Batch two images for efficiency. Encoder attention is spatial only;
        # fixed BN has no batch dependence or cross-view/context mixing.
        if pairs.ndim != 5 or pairs.shape[1:] != (2, 3, 224, 224) or not 1 <= len(pairs) <= 8:
            raise ValueError(f"Expected one ordered stereo timestamp, got {pairs.shape}")
        count = len(pairs)
        # Fixed batch geometry also makes offline batches and single cache misses
        # use the same CUDA convolution kernels. Padding has no statistical effect.
        padded = torch.cat([pairs, pairs[-1:].expand(8-count, -1, -1, -1, -1)]) if count < 8 else pairs
        z = self._core.encode(padded.flatten(0, 1))[:2*count]
        if z.shape != (2*count, 128, 14, 14):
            raise ValueError(f"Unexpected native FLUX.2 latent shape: {z.shape}")
        return z.reshape(count, 2, 128, 14, 14)

    @torch.no_grad()
    def batch_encode(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2:] != (224, 448):
            raise ValueError("Expected [B,3,T,224,448]")
        b, _, t, _, _ = video.shape
        pairs = video.reshape(b, 3, t, 224, 2, 224).permute(0, 2, 4, 1, 3, 5)
        pairs = pairs.reshape(b*t, 2, 3, 224, 224)
        encode = self._encode_pairs if self.feature_cache is None else lambda pair: self.feature_cache.get_or_encode(pair, self._encode_pairs)
        z = torch.cat([encode(pair.unsqueeze(0)) for pair in pairs])
        return z.reshape(b, t, 2, 128, 14, 14).permute(0, 3, 1, 4, 2, 5).reshape(b, 128, t, 14, 28)

    @classmethod
    def from_pretrained(cls, model_path, *, bottleneck_dim=256, feature_cache_dir=None,
                        seperate_encode_view=True, add_one_history_context=True,
                        add_input_bottleneck=True):
        from safetensors.torch import load_file
        if not all(value is True for value in (seperate_encode_view, add_one_history_context, add_input_bottleneck)):
            raise ValueError("E06 requires independent camera grids, two clean positions and M29 bottleneck")
        path = Path(model_path) / "encoder_only.safetensors"
        core = FluxVaeEncoderCore(**_read_flux2_vae_config(str(model_path))).to(torch.bfloat16)
        core.load_state_dict(load_file(path), strict=True)
        instance = cls(core, bottleneck_dim=bottleneck_dim)
        if feature_cache_dir:
            instance.configure_feature_cache(feature_cache_dir, path)
        return instance

    def save_deploy_assets(self, output_dir, cfg):
        directory = Path(output_dir) / "flux2_framewise"
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(cfg.model.video_backbone.encoder.model_path) / "config.json", directory / "config.json")
        (directory / "encoder.json").write_text(json.dumps(self.processing_config, indent=2) + "\n")

    @classmethod
    def from_skeleton(cls, components_entry, *, device="cpu", encoder_cfg=None, ckpt_dir=None):
        if ckpt_dir is None or encoder_cfg is None:
            raise ValueError("E06 requires checkpoint-local structural assets and encoder config")
        if any(encoder_cfg.get(key) is not True for key in ("add_one_history_context", "seperate_encode_view", "add_input_bottleneck")):
            raise ValueError("E06 deployment clean/projection contract differs")
        directory = Path(ckpt_dir) / "flux2_framewise"
        with torch.device(device):
            instance = cls(FluxVaeEncoderCore(**_read_flux2_vae_config(str(directory))),
                           bottleneck_dim=encoder_cfg.get("bottleneck_dim", 256))
        if json.loads((directory / "encoder.json").read_text()) != instance.processing_config:
            raise ValueError("E06 processing differs from checkpoint sidecar")
        return instance.to(dtype=torch.bfloat16)
