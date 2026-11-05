"""I-JEPA image encoder adapted to the M29 video-latent contract.

I-JEPA is an image JEPA: every image is encoded independently and produces a
spatial patch grid.  M29 still needs a causal clean prefix plus future latent
frames, so this module owns the small amount of temporal adaptation required
to compare it with V-JEPA:

``two_clean``
    ``[history, current, mean(future pairs)]``.  This preserves both clean
    images and advertises ``num_condition_latents=2``.

``gated_difference``
    ``[current + alpha * P(current-history), mean(future pairs)]``.  This keeps
    the V-JEPA/M29 five-latent geometry with one fused clean latent.  ``alpha``
    starts at zero, so the initial condition is exactly the current image.

The encoder itself is frozen by the host training configuration in the same
way as V-JEPA/DINOv3.  The difference adapter is a normal module so callers
can explicitly choose whether it is trainable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torch import Tensor

from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.bottleneck import BottleneckPatchEmbed
from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.svae import reducer

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MODES = {"gated_difference", "two_clean"}


def _preprocess_image(image: Image.Image, *, dtype, device) -> Tensor:
    arr = torch.tensor(np.array(image, dtype=np.float32), dtype=dtype, device=device)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"I-JEPA expects RGB images, got array shape {tuple(arr.shape)}")
    arr = arr / 255.0
    mean = torch.tensor(_IMAGENET_MEAN, dtype=dtype, device=device)
    std = torch.tensor(_IMAGENET_STD, dtype=dtype, device=device)
    arr = (arr - mean) / std
    return rearrange(arr, "H W C -> 1 C H W")


def _load_official_vit(model_name: str, *, img_size: int, patch_size: int) -> nn.Module:
    """Import the vendored official implementation without installing it."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/ijepa"))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.models import vision_transformer as vit  # type: ignore

    if model_name not in vit.VIT_EMBED_DIMS:
        raise ValueError(f"Unknown I-JEPA model_name={model_name!r}; choose {sorted(vit.VIT_EMBED_DIMS)}")
    return getattr(vit, model_name)(img_size=[img_size], patch_size=patch_size)


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict) and "encoder" in checkpoint:
        state = checkpoint["encoder"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError(f"I-JEPA checkpoint must be a mapping, got {type(checkpoint).__name__}")
    if not isinstance(state, dict) or not state:
        raise ValueError("I-JEPA checkpoint does not contain a non-empty encoder state dict")
    keys = list(state)
    if all(key.startswith("module.") for key in keys):
        state = {key[len("module."):]: value for key, value in state.items()}
    if any(key.startswith("encoder.") for key in state):
        state = {
            key[len("encoder."):]: value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
    return state


@register_video_encoder("ijepa")
class IJEPAImageVideoEncoder(VideoEncoder):
    """Frame-wise official I-JEPA ViT with M29 temporal adaptation."""

    def __init__(
        self,
        vit: nn.Module,
        *,
        embed_dim: int,
        patch_size: int = 16,
        temporal_group_size: int = 2,
        history_fusion_mode: str = "gated_difference",
        add_one_history_context: bool = True,
        seperate_encode_view: bool = False,
        add_input_bottleneck: bool = True,
        bottleneck_dim: int = 256,
        feature_norm_eps: float = 1e-5,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
        svae_config: dict | None = None,
    ):
        super().__init__()
        if history_fusion_mode not in _MODES:
            raise ValueError(f"history_fusion_mode must be one of {_MODES}, got {history_fusion_mode!r}")
        if temporal_group_size != 2:
            raise ValueError("M29 I-JEPA adaptation currently requires temporal_group_size=2")
        if not isinstance(add_one_history_context, bool):
            raise ValueError("add_one_history_context must be a boolean")
        if not add_one_history_context:
            raise ValueError("I-JEPA M29 comparison requires an explicit [history, current] pair")
        if seperate_encode_view and patch_size != 16:
            raise ValueError("Independent 224x224 views require patch_size=16")
        if add_input_bottleneck and (not isinstance(bottleneck_dim, int) or bottleneck_dim <= 0):
            raise ValueError("bottleneck_dim must be a positive integer")

        self._m = vit
        self._raw_embed_dim = int(embed_dim)
        self._patch_size = int(patch_size)
        self._temporal_group_size = int(temporal_group_size)
        self._history_fusion_mode = history_fusion_mode
        self._seperate_encode_view = bool(seperate_encode_view)
        self._add_input_bottleneck = bool(add_input_bottleneck)
        self._bottleneck_dim = int(bottleneck_dim)
        self._processing = {
            "patch_size": self._patch_size,
            "temporal_group_size": self._temporal_group_size,
            "history_fusion_mode": history_fusion_mode,
            "add_one_history_context": True,
            "seperate_encode_view": self._seperate_encode_view,
            "add_input_bottleneck": self._add_input_bottleneck,
            "bottleneck_dim": self._bottleneck_dim,
            "feature_norm_eps": float(feature_norm_eps),
        }
        self._svae = reducer.build(svae_path, svae_target_dim, svae_config)
        effective_z_dim = reducer.effective_z_dim(self._svae, self._raw_embed_dim)
        self._spec = VideoEncoderProperties(
            z_dim=int(effective_z_dim),
            spatial_compression=self._patch_size,
            temporal_compression=2,
            causal_temporal=True,
            pixel_decode=False,
            dit_patch_size=(1, 2, 2),
            num_condition_latents=2 if history_fusion_mode == "two_clean" else 1,
        )
        self.feature_norm = nn.LayerNorm(int(effective_z_dim), eps=feature_norm_eps)
        self._delta_proj = nn.Linear(self._raw_embed_dim, self._raw_embed_dim, bias=False)
        nn.init.eye_(self._delta_proj.weight)
        self.delta_scale = nn.Parameter(torch.zeros(()))

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    @property
    def history_fusion_mode(self) -> str:
        return self._history_fusion_mode

    @property
    def add_one_history_context(self) -> bool:
        return True

    @property
    def seperate_encode_view(self) -> bool:
        return self._seperate_encode_view

    @property
    def processing_config(self) -> dict[str, Any]:
        return dict(self._processing)

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "IJEPAImageVideoEncoder":
        """Rebuild from saved YAML geometry; checkpoint loading fills all weights."""
        if encoder_cfg is None:
            raise ValueError("I-JEPA checkpoint reconstruction requires the saved encoder config")
        get = encoder_cfg.get
        options = {key: get(key, default) for key, default in {
            "temporal_group_size": 2,
            "history_fusion_mode": "gated_difference",
            "add_one_history_context": True,
            "seperate_encode_view": False,
            "add_input_bottleneck": True,
            "bottleneck_dim": 256,
            "feature_norm_eps": 1e-5,
        }.items()}
        svae_config = reducer.read_sidecar(ckpt_dir)
        if get("svae_path") and svae_config is None:
            raise FileNotFoundError("I-JEPA checkpoint with S-VAE requires checkpoint-local svae_config.json")
        with torch.device(device):
            vit = _load_official_vit(
                get("model_name", "vit_giant"),
                img_size=int(get("img_size", 224)),
                patch_size=int(get("patch_size", 16)),
            ).eval()
            encoder = cls(
                vit,
                embed_dim=vit.embed_dim,
                patch_size=int(get("patch_size", 16)),
                svae_config=svae_config,
                svae_target_dim=reducer.read_target_dim_from_cfg(encoder_cfg),
                **options,
            )
        sidecar = os.path.join(ckpt_dir, "ijepa", "encoder.json") if ckpt_dir else None
        if sidecar and os.path.isfile(sidecar):
            with open(sidecar) as stream:
                saved = json.load(stream)
            if saved["processing"] != encoder.processing_config or saved["embed_dim"] != encoder._raw_embed_dim:
                raise ValueError("I-JEPA processing differs from saved encoder.json")
        return encoder

    def save_deploy_assets(self, output_dir: str, cfg: Any) -> None:
        directory = os.path.join(output_dir, "ijepa")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "encoder.json"), "w") as stream:
            json.dump({"embed_dim": self._raw_embed_dim, "processing": self.processing_config}, stream, indent=2)
        if self._svae is not None:
            reducer.write_sidecar(self._svae, output_dir, type(self).__name__)

    def build_dit_input_proj(self, dit_dim: int) -> nn.Module:
        if not self._add_input_bottleneck:
            return super().build_dit_input_proj(dit_dim)
        return BottleneckPatchEmbed(
            self.properties.z_dim, dit_dim, self.properties.dit_patch_size, self._bottleneck_dim
        )

    def preprocess_video(self, frames) -> Tensor:
        dtype = next(self._m.parameters()).dtype
        device = next(self._m.parameters()).device
        return torch.stack([_preprocess_image(frame, dtype=dtype, device=device) for frame in frames], dim=2)

    def preprocess_video_batch(self, videos) -> Tensor:
        return torch.cat([self.preprocess_video(frames) for frames in videos], dim=0)

    def _encode_frames(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"I-JEPA expects (B, 3, T, H, W), got {tuple(video.shape)}")
        b, _, t, h, w = video.shape
        if h % self._patch_size or w % self._patch_size:
            raise ValueError(
                f"I-JEPA requires H,W divisible by patch_size={self._patch_size}; got {(h, w)}"
            )
        if self._seperate_encode_view:
            if (h, w) != (224, 448):
                raise ValueError("seperate_encode_view requires two 224x224 views packed as 224x448")
            video = rearrange(video, "B C T H (V W) -> (B V) C T H W", V=2)
            b, _, t, h, w = video.shape
        flat = rearrange(video, "B C T H W -> (B T) C H W")
        output = self._m(flat)
        if isinstance(output, Tensor):
            tokens = output
        else:
            tokens = getattr(output, "last_hidden_state", output)
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise TypeError("I-JEPA forward must return a [B*T, N, D] tensor")
        expected = (h // self._patch_size) * (w // self._patch_size)
        if tokens.shape != (b * t, expected, self._raw_embed_dim):
            raise ValueError(
                f"I-JEPA output shape {tuple(tokens.shape)} does not match "
                f"expected {(b * t, expected, self._raw_embed_dim)}"
            )
        grid = rearrange(
            tokens, "(B T) (Hl Wl) D -> B D T Hl Wl",
            B=b, T=t, Hl=h // self._patch_size, Wl=w // self._patch_size,
        )
        if self._seperate_encode_view:
            grid = rearrange(grid, "(B V) D T H W -> B D T H (V W)", V=2)
        return grid

    def _batch_encode_raw(self, video: Tensor) -> Tensor:
        return self._adapt_frame_grid(self._encode_frames(video))

    def _adapt_frame_grid(self, grid: Tensor) -> Tensor:
        """Original I-JEPA I3 temporal path, also used after frozen extraction."""
        if grid.shape[2] < 2:
            raise ValueError("I-JEPA M29 adaptation requires at least history and current frames")
        history = grid[:, :, :1]
        current = grid[:, :, 1:2]
        future = grid[:, :, 2:]
        if future.shape[2] % self._temporal_group_size:
            raise ValueError(
                f"I-JEPA future frame count must be divisible by {self._temporal_group_size}; "
                f"got {future.shape[2]}"
            )
        future = future.reshape(
            *future.shape[:2], future.shape[2] // self._temporal_group_size,
            self._temporal_group_size, *future.shape[3:]
        ).mean(dim=3)
        if self._history_fusion_mode == "two_clean":
            return torch.cat([history, current, future], dim=2)
        delta = current - history
        delta = rearrange(delta, "B D T H W -> B T H W D")
        delta = self._delta_proj(delta)
        delta = rearrange(delta, "B T H W D -> B D T H W")
        condition = current + self.delta_scale.to(current.dtype) * delta
        return torch.cat([condition, future], dim=2)

    def _normalize(self, z: Tensor) -> Tensor:
        z = self._apply_svae(z)
        z = rearrange(z, "B D T H W -> B T H W D")
        z = self.feature_norm(z)
        return rearrange(z, "B T H W D -> B D T H W")

    def batch_encode(self, video: Tensor) -> Tensor:
        return self._normalize(self._batch_encode_raw(video))

    def batch_encode_pooled_for_svae_training(self, video: Tensor) -> Tensor:
        if self._svae is not None:
            raise RuntimeError("S-VAE feature collection requires an unreduced I-JEPA encoder")
        return self._batch_encode_raw(video)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        checkpoint: str | None = None,
        model_name: str = "vit_giant",
        img_size: int = 224,
        patch_size: int = 16,
        **kwargs: Any,
    ) -> "IJEPAImageVideoEncoder":
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"I-JEPA model_path is not a directory: {model_path}")
        vit = _load_official_vit(model_name, img_size=img_size, patch_size=patch_size)
        ckpt_path = checkpoint or "checkpoint.pth.tar"
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(model_path, ckpt_path)
        if not os.path.isfile(ckpt_path):
            candidates = [os.path.join(model_path, name) for name in ("checkpoint.pth", "checkpoint.pt", "encoder.pth")]
            ckpt_path = next((path for path in candidates if os.path.isfile(path)), ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"I-JEPA checkpoint not found under {model_path}")
        state = _checkpoint_state_dict(torch.load(ckpt_path, map_location="cpu"))
        missing, unexpected = vit.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"I-JEPA checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
        return cls(vit, embed_dim=vit.embed_dim, patch_size=patch_size, **kwargs)
