"""Frozen DINO image encoders adapted to the V-JEPA policy token geometry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoLatentEncoder(nn.Module):
    """Map DINOv2/v3 image tokens onto the policy's fixed 5x14x14 grid.

    Five endpoint frames are selected from the policy's ten-frame video clip.
    A 5D input returns per-view ``t-h-w`` tokens. A two-view 6D input returns
    tokens directly in ``t-view-h-w`` order.
    """

    embed_dim = 1024
    patch_size = 16
    tubelet_size = 2
    num_frames = 10
    num_views = 2
    grid_depth = 5
    grid_height = 14
    grid_width = 14
    patches_per_step = grid_height * grid_width
    num_patches_per_view = grid_depth * patches_per_step
    num_patches = num_views * num_patches_per_view

    _FAMILIES = {
        "dinov2": {"native_patch_size": 14, "prefix_tokens": 1, "register_tokens": 0},
        "dinov3": {"native_patch_size": 16, "prefix_tokens": 5, "register_tokens": 4},
    }

    def __init__(
        self,
        backbone: nn.Module,
        family: str,
        *,
        image_microbatch_size: int | None = None,
    ) -> None:
        super().__init__()
        if family not in self._FAMILIES:
            raise ValueError(f"unsupported DINO family {family!r}")
        if image_microbatch_size is not None and image_microbatch_size <= 0:
            raise ValueError("image_microbatch_size must be positive")
        self.family = family
        self.encoder_family = family
        self.image_microbatch_size = image_microbatch_size
        self.backbone = backbone
        self._vjepa_policy_family = family
        self._vjepa_policy_model_name = f"{family}_vitl"
        self._vjepa_policy_checkpoint_key = "local_pretrained"
        self.blocks = (
            SimpleNamespace(
                attn=SimpleNamespace(
                    interpolate_rope=False,
                    corrected_frequency_pairing=False,
                )
            ),
        )

        expected = self._FAMILIES[family]
        config = getattr(backbone, "config", None)
        if config is None:
            raise ValueError("DINO backbone must expose a Hugging Face-style config")
        required = {
            "hidden_size": self.embed_dim,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "patch_size": expected["native_patch_size"],
        }
        mismatches = {
            name: (getattr(config, name, None), value)
            for name, value in required.items()
            if getattr(config, name, None) != value
        }
        registers = int(getattr(config, "num_register_tokens", 0))
        if registers != expected["register_tokens"]:
            mismatches["num_register_tokens"] = (
                registers,
                expected["register_tokens"],
            )
        if mismatches:
            raise ValueError(f"DINO ViT-L configuration mismatch: {mismatches}")

        self.native_patch_size = expected["native_patch_size"]
        self.num_prefix_tokens = expected["prefix_tokens"]
        self.register_buffer(
            "frame_indices",
            torch.tensor([1, 3, 5, 7, 9], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> DinoLatentEncoder:
        """Keep the frozen backbone in eval mode, including under parent.train()."""
        super().train(False)
        return self

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        rgb = images.mul(0.5).add(0.5)
        mean = self.image_mean.to(dtype=rgb.dtype)
        std = self.image_std.to(dtype=rgb.dtype)
        return (rgb - mean) / std

    @staticmethod
    def _last_hidden_state(output: Any) -> torch.Tensor:
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, dict) and "last_hidden_state" in output:
            return output["last_hidden_state"]
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError("DINO backbone output has no last_hidden_state")

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        output = self.backbone(pixel_values=self._normalize_images(images))
        tokens = self._last_hidden_state(output)
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"DINO last_hidden_state must be [B,N,1024], got {tuple(tokens.shape)}"
            )

        native_grid = images.shape[-1] // self.native_patch_size
        expected_patches = native_grid * (images.shape[-2] // self.native_patch_size)
        expected_tokens = self.num_prefix_tokens + expected_patches
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"DINO returned {tokens.shape[1]} tokens, expected {expected_tokens}"
            )
        patches = tokens[:, self.num_prefix_tokens :]

        if self.family == "dinov2":
            if patches.shape[1] != 16 * 16:
                raise ValueError("DINOv2 at 224 must return a 16x16 patch grid")
            patches = patches.reshape(-1, 16, 16, self.embed_dim).permute(0, 3, 1, 2)
            patches = F.interpolate(
                patches,
                size=(self.grid_height, self.grid_width),
                mode="bilinear",
                align_corners=True,
            )
            patches = patches.permute(0, 2, 3, 1).reshape(
                -1, self.patches_per_step, self.embed_dim
            )
        elif patches.shape[1] != self.patches_per_step:
            raise ValueError("DINOv3 at 224 must return a 14x14 patch grid")
        return patches.contiguous()

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        microbatch = self.image_microbatch_size
        if microbatch is None or images.shape[0] <= microbatch:
            return self._encode_image_batch(images)
        return torch.cat(
            [
                self._encode_image_batch(images[start : start + microbatch])
                for start in range(0, images.shape[0], microbatch)
            ],
            dim=0,
        )

    def _validate_5d(self, clips: torch.Tensor) -> None:
        if clips.ndim != 5:
            raise ValueError(f"clips must be [B,C,T,H,W], got {tuple(clips.shape)}")
        _, channels, frames, height, width = clips.shape
        if (channels, frames, height, width) != (3, self.num_frames, 224, 224):
            raise ValueError(
                f"DINO clips must be [B,3,10,224,224], got {tuple(clips.shape)}"
            )
        if not torch.is_floating_point(clips):
            raise TypeError("DINO clips must be floating point")

    def _canonical_masks(
        self,
        masks: torch.Tensor | Sequence[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        mask_list = [masks] if torch.is_tensor(masks) else list(masks)
        if not mask_list:
            raise ValueError("masks must not be empty")
        token_counts = set()
        canonical = []
        for mask in mask_list:
            if not torch.is_tensor(mask) or mask.ndim != 2:
                raise ValueError("each mask must be a [B,K] tensor")
            if mask.shape[0] != batch_size:
                raise ValueError(
                    f"mask batch {mask.shape[0]} does not match clips batch {batch_size}"
                )
            if mask.dtype == torch.bool or torch.is_floating_point(mask):
                raise TypeError("mask indices must use an integer dtype")
            token_counts.add(mask.shape[1])
            canonical.append(mask.to(device=device, dtype=torch.long))
        if len(token_counts) != 1:
            raise ValueError("all masks must select the same number of tokens")
        return canonical

    def _forward_5d(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        self._validate_5d(clips)
        batch_size = clips.shape[0]

        if masks is None:
            temporal_ids = torch.arange(self.grid_depth, device=clips.device)
            mask_list = None
        else:
            mask_list = self._canonical_masks(masks, batch_size, clips.device)
            all_ids = torch.cat([mask.reshape(-1) for mask in mask_list])
            temporal_ids = torch.unique(
                torch.div(all_ids, self.patches_per_step, rounding_mode="floor"),
                sorted=True,
            )

        frame_ids = self.frame_indices.index_select(0, temporal_ids)
        images = clips.index_select(2, frame_ids)
        images = images.permute(0, 2, 1, 3, 4).reshape(
            batch_size * temporal_ids.numel(), 3, 224, 224
        )
        patches = self._encode_images(images).reshape(
            batch_size,
            temporal_ids.numel(),
            self.patches_per_step,
            self.embed_dim,
        )

        if mask_list is None:
            return patches.flatten(1, 2)

        batch_ids = torch.arange(batch_size, device=clips.device).unsqueeze(1)
        selected = []
        for mask in mask_list:
            mask_temporal = torch.div(
                mask, self.patches_per_step, rounding_mode="floor"
            )
            mask_spatial = mask.remainder(self.patches_per_step)
            encoded_temporal = torch.searchsorted(temporal_ids, mask_temporal)
            selected.append(patches[batch_ids, encoded_temporal, mask_spatial])
        return torch.cat(selected, dim=0)

    def forward(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if clips.ndim == 5:
            return self._forward_5d(clips, masks)
        if clips.ndim != 6:
            raise ValueError(
                f"clips must be [B,C,T,H,W] or [B,V,C,T,H,W], got {tuple(clips.shape)}"
            )
        if masks is not None:
            raise ValueError(
                "masked 6D clips must use vjepa_policy.models.vjepa_policy."
                "encode_video_views so global view masks are localized first"
            )

        batch_size, num_views, channels, frames, height, width = clips.shape
        if num_views != self.num_views:
            raise ValueError(f"DINO ablation requires exactly {self.num_views} views")
        per_view = self._forward_5d(
            clips.reshape(batch_size * num_views, channels, frames, height, width),
            None,
        )
        return (
            per_view.reshape(
                batch_size,
                num_views,
                self.grid_depth,
                self.patches_per_step,
                self.embed_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, self.num_patches, self.embed_dim)
        )

    @classmethod
    def from_pretrained(
        cls,
        family: str,
        pretrained_path: str | Path,
        *,
        image_microbatch_size: int | None = None,
    ) -> DinoLatentEncoder:
        return load_dino_latent_encoder(
            family,
            pretrained_path,
            image_microbatch_size=image_microbatch_size,
        )


def load_dino_latent_encoder(
    family: str,
    pretrained_path: str | Path,
    *,
    image_microbatch_size: int | None = None,
) -> DinoLatentEncoder:
    """Load an official DINO ViT-L from an explicit local HF directory."""
    if family not in DinoLatentEncoder._FAMILIES:
        raise ValueError(f"unsupported DINO family {family!r}")
    path = Path(pretrained_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"pretrained_path must be an existing local Hugging Face directory: {path}"
        )
    path = path.resolve()

    try:
        from transformers import DINOv3ViTModel, Dinov2Model
    except ImportError as error:
        raise ImportError(
            "loading DINO encoders requires a transformers build with DINOv3 support"
        ) from error

    model_class = Dinov2Model if family == "dinov2" else DINOv3ViTModel
    backbone = model_class.from_pretrained(str(path), local_files_only=True)
    return DinoLatentEncoder(
        backbone,
        family,
        image_microbatch_size=image_microbatch_size,
    )
