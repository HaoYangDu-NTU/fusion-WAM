"""Frozen Wan2.1 I2V video features with an independently encoded clean clip.

Inputs are five consecutive observations ending at t=0, optionally followed
by 4k future frames. The clean summary is the last latent of the five-frame
clip; future targets come from a separate full-video forward. No generated
pixels, diffusion sampling loop, or future-dependent clean tokens are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import VideoEncoder, VideoEncoderProperties
from .bottleneck import BottleneckPatchEmbed
from .registry import register_video_encoder


def dit_config(config):
    if config.get("model_type", "i2v") != "i2v" or config.get("in_dim", 36) != 36:
        raise ValueError("Wan feature extraction requires native 36-channel I2V weights")
    return {**{key: config[key] for key in ("dim", "ffn_dim", "freq_dim", "num_heads", "num_layers", "eps")},
            "in_dim": 36, "out_dim": 16, "text_dim": 4096,
            "patch_size": (1, 2, 2), "has_image_input": True}


def load_weights(module, paths, converter=lambda state: state):
    """Load shards directly as BF16, including into a meta-device skeleton."""
    from safetensors.torch import load_file
    expected = set(module.state_dict())
    loaded = set()
    for path in paths:
        path = Path(path)
        state = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = converter(state)
        if loaded.intersection(state):
            raise ValueError(f"Duplicate weight keys in {path}")
        unexpected = set(state) - expected
        if unexpected:
            raise ValueError(f"Unexpected weights in {path}: {sorted(unexpected)[:5]}")
        state = {key: value.to(torch.bfloat16) for key, value in state.items()}
        module.load_state_dict(state, strict=False, assign=True)
        loaded.update(state)
    if loaded != expected:
        raise ValueError(f"Missing weights: {sorted(expected - loaded)[:10]}")
    return module


def load_empty_context(path, model_path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        identity = json.loads((weights.metadata() or {}).get("identity", "{}"))
        source = Path(model_path) / "models_t5_umt5-xxl-enc-bf16.pth"
        expected = {"prompt": "", "encoder": "native_wan21_umt5", "padding": "zero_after_valid_tokens",
                    "source_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns}
        if identity != expected:
            raise ValueError("Wan empty-text cache differs from the native UMT5 source/prompt contract")
        context, lengths = weights.get_tensor("context"), weights.get_tensor("seq_lens")
    if (context.shape != (1, 512, 4096) or context.dtype != torch.bfloat16
            or lengths.shape != (1,) or not 0 < int(lengths[0]) <= 512
            or not torch.isfinite(context).all() or torch.count_nonzero(context[:, int(lengths[0]):])):
        raise ValueError("Invalid native Wan empty-text cache shape, dtype, length, or padding")
    return context


@register_video_encoder("wan21_features")
class Wan21VideoFeatureEncoder(VideoEncoder):
    clean_condition_frame_offsets = (-4, -3, -2, -1, 0)
    seperate_encode_view = True

    def __init__(self, dit, vae, image_encoder, empty_context, *, feature_layer=None,
                 microbatch_size=1, bottleneck_dim=256, feature_norm_eps=1e-5):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.image_encoder = image_encoder
        self.feature_layer = len(dit.blocks) if feature_layer is None else feature_layer
        if isinstance(self.feature_layer, bool) or not isinstance(self.feature_layer, int) or not 1 <= self.feature_layer <= len(dit.blocks):
            raise ValueError("feature_layer is a one-based block index within the loaded DiT")
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size <= 0:
            raise ValueError("microbatch_size must be a positive integer")
        if isinstance(bottleneck_dim, bool) or not isinstance(bottleneck_dim, int) or bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if empty_context.shape != (1, 512, 4096) or not torch.isfinite(empty_context).all():
            raise ValueError("Expected a finite, native zero-padded UMT5 empty context [1,512,4096]")
        self.register_buffer("empty_context", empty_context.to(dtype=next(dit.parameters()).dtype))
        self.microbatch_size = microbatch_size
        self.bottleneck_dim = bottleneck_dim
        self.feature_norm = nn.LayerNorm(dit.dim, eps=feature_norm_eps, elementwise_affine=False)
        self._spec = VideoEncoderProperties(z_dim=dit.dim, spatial_compression=16,
                                            temporal_compression=4, causal_temporal=True,
                                            pixel_decode=False, dit_patch_size=(1, 2, 2))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent model.train() must never reactivate CLIP/dropout in the teacher.
        return super().train(False)

    @property
    def properties(self):
        return self._spec

    @property
    def processing_config(self):
        return {"feature_layer": self.feature_layer, "microbatch_size": self.microbatch_size,
                "bottleneck_dim": self.bottleneck_dim, "feature_norm_eps": self.feature_norm.eps,
                "timestep": 0, "clean_condition_frame_offsets": list(self.clean_condition_frame_offsets),
                "seperate_encode_view": True, "temporal_design": "native_video_independent_clean"}

    def build_dit_input_proj(self, dit_dim):
        return BottleneckPatchEmbed(self.properties.z_dim, dit_dim, (1, 2, 2), self.bottleneck_dim)

    def preprocess_video(self, frames):
        parameter = next(self.dit.parameters())
        images = []
        for frame in frames:
            if frame.size != (448, 224):
                raise ValueError("Wan features expect two independent 224x224 views packed as 224x448")
            array = np.array(frame.convert("RGB"), copy=True)
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(images, dim=1).unsqueeze(0).to(parameter.device, parameter.dtype).div_(127.5).sub_(1)

    def preprocess_video_batch(self, videos):
        return torch.cat([self.preprocess_video(frames) for frames in videos])

    def _video_features(self, pixels):
        """Native VAE/CLIP/I2V input, then hidden states before the velocity head."""
        from ..wan.models.dit import sinusoidal_embedding_1d
        batch, _, frames, height, width = pixels.shape
        if frames < 1 or (frames - 1) % 4:
            raise ValueError("Wan video clips must contain 4k+1 frames")
        latents = self.vae.batch_encode(pixels, device=pixels.device)
        conditioning_pixels = torch.zeros_like(pixels)
        conditioning_pixels[:, :, :1] = pixels[:, :, :1]
        image_latents = self.vae.batch_encode(conditioning_pixels, device=pixels.device)
        mask = torch.zeros(batch, 4, *latents.shape[2:], device=pixels.device, dtype=pixels.dtype)
        mask[:, :, :1] = 1
        tokens = self.dit.patch_embedding(torch.cat([latents, mask, image_latents], dim=1))
        _, channels, temporal, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        # CLIP owns its native image normalization and penultimate-block output.
        clip = self.image_encoder.encode_image([pixels[:, :, 0].clone()])
        context = torch.cat([self.dit.img_emb(clip), self.dit.text_embedding(self.empty_context.expand(batch, -1, -1))], dim=1)
        timestep = torch.zeros(batch, device=pixels.device, dtype=pixels.dtype)
        time = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, timestep).to(pixels.dtype))
        modulation = self.dit.time_projection(time).unflatten(1, (6, channels))
        freqs = torch.cat([
            self.dit.freqs[0][:temporal].view(temporal, 1, 1, -1).expand(temporal, h, w, -1),
            self.dit.freqs[1][:h].view(1, h, 1, -1).expand(temporal, h, w, -1),
            self.dit.freqs[2][:w].view(1, 1, w, -1).expand(temporal, h, w, -1),
        ], dim=-1).reshape(temporal * h * w, 1, -1).to(pixels.device)
        for block in self.dit.blocks[:self.feature_layer]:
            tokens = block(tokens, context, modulation, freqs)
        tokens = self.feature_norm(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, temporal, h, w)

    @torch.no_grad()
    def batch_encode(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2:] != (224, 448):
            raise ValueError("Expected [B,3,T,224,448] packed two-view video")
        if video.shape[2] < 5 or (video.shape[2] - 5) % 4:
            raise ValueError("Expected five history/current frames plus 4k future frames")
        batch = video.shape[0]
        views = torch.stack(video.split(224, dim=-1), dim=1).flatten(0, 1)
        outputs = []
        for chunk in views.split(self.microbatch_size):
            clean = self._video_features(chunk[:, :, :5])[:, :, -1:]
            if chunk.shape[2] > 5:
                # Full-video attention is allowed only for detached future targets.
                future = self._video_features(chunk)[:, :, 2:]
                clean = torch.cat([clean, future], dim=2)
            outputs.append(clean)
        features = torch.cat(outputs).unflatten(0, (batch, 2))
        return torch.cat([features[:, 0], features[:, 1]], dim=-1)

    @classmethod
    def from_pretrained(cls, model_path, *, empty_context_path, feature_layer=40,
                        microbatch_size=1, bottleneck_dim=256, feature_norm_eps=1e-5,
                        seperate_encode_view=True, add_input_bottleneck=True):
        from accelerate import init_empty_weights
        from ..wan.models.dit import WanModel
        from ..wan.models.image_encoder import WanImageEncoder
        from ..wan.models.vae import WanVideoVAE
        from ..wan.shared.utils.state_dict_converters.wan_video_image_encoder import WanImageEncoderStateDictConverter
        from ..wan.shared.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter
        path = Path(model_path)
        if seperate_encode_view is not True:
            raise ValueError("Wan video features require independent camera views")
        if add_input_bottleneck is not True:
            raise ValueError("Wan video features require separate clean/future bottleneck projections")
        config = json.loads((path / "config.json").read_text())
        context = load_empty_context(empty_context_path, path)
        with init_empty_weights():
            dit = WanModel(**dit_config(config))
            vae = WanVideoVAE()
            image = WanImageEncoder()
        index = json.loads((path / "diffusion_pytorch_model.safetensors.index.json").read_text())
        shards = [path / name for name in sorted(set(index["weight_map"].values()))]
        load_weights(dit, shards)
        load_weights(vae, [path / "Wan2.1_VAE.pth"], WanVideoVAEStateDictConverter)
        load_weights(image, [path / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"], WanImageEncoderStateDictConverter)
        encoder = cls(dit, vae, image, context, feature_layer=feature_layer, microbatch_size=microbatch_size,
                      bottleneck_dim=bottleneck_dim, feature_norm_eps=feature_norm_eps)
        encoder._source_config = config
        return encoder

    def save_deploy_assets(self, output_dir, cfg):
        path = Path(output_dir) / "wan21_features"
        path.mkdir(parents=True, exist_ok=True)
        (path / "encoder.json").write_text(json.dumps({"model_config": self._source_config,
                                                      "processing": self.processing_config}, indent=2) + "\n")

    @classmethod
    def from_skeleton(cls, components_entry, *, device="cpu", encoder_cfg=None, ckpt_dir=None):
        from ..wan.models.dit import WanModel
        from ..wan.models.image_encoder import WanImageEncoder
        from ..wan.models.vae import WanVideoVAE
        if ckpt_dir is None:
            raise ValueError("Wan feature reconstruction requires checkpoint-local encoder.json")
        saved = json.loads((Path(ckpt_dir) / "wan21_features/encoder.json").read_text())
        processing = saved["processing"]
        if (processing.get("timestep") != 0 or processing.get("seperate_encode_view") is not True
                or processing.get("temporal_design") != "native_video_independent_clean"
                or processing.get("clean_condition_frame_offsets") != list(cls.clean_condition_frame_offsets)):
            raise ValueError("Wan video feature processing sidecar has an incompatible temporal contract")
        for key in ("feature_layer", "microbatch_size", "bottleneck_dim", "feature_norm_eps"):
            if encoder_cfg is not None and encoder_cfg.get(key) != processing[key]:
                raise ValueError(f"Wan feature checkpoint/config differs: {key}")
        with torch.device(device):
            encoder = cls(WanModel(**dit_config(saved["model_config"])), WanVideoVAE(), WanImageEncoder(),
                          torch.zeros(1, 512, 4096), **{k: processing[k] for k in ("feature_layer", "microbatch_size", "bottleneck_dim", "feature_norm_eps")})
        encoder._source_config = saved["model_config"]
        return encoder.to(dtype=torch.bfloat16)
