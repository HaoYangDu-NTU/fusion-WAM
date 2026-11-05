"""Frozen InternVideo3 vision tower adapted to the policy's 5x14x14 grid."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


INTERNVIDEO3_EMBED_DIM = 1152
INTERNVIDEO3_GRID_DEPTH = 5
INTERNVIDEO3_GRID_SIZE = 14
INTERNVIDEO3_NUM_FRAMES = 10
INTERNVIDEO3_PATCH_SIZE = 16
INTERNVIDEO3_TUBELET_SIZE = 2
INTERNVIDEO3_SPATIAL_MERGE_SIZE = 2
INTERNVIDEO3_PATCHES_PER_STEP = INTERNVIDEO3_GRID_SIZE**2

_VISION_SHARD = "model-vision-0001-others-save_rank0.safetensors"
_PROJECTOR_SHARD = "model-projector-0001-others-save_rank0.safetensors"
_VISION_PREFIX = "model.visual."


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _apply_vision_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype, key_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query.to(query_dtype), key.to(key_dtype)


def _fixed_grid_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    num_sequences: int,
    sequence_length: int,
) -> torch.Tensor:
    """Vectorize the official per-tubelet SDPA over equal 14x14 grids."""
    query, key, value = (
        attention.qkv(hidden_states)
        .reshape(num_sequences, sequence_length, 3, attention.num_heads, -1)
        .permute(2, 0, 1, 3, 4)
        .unbind(0)
    )
    cos, sin = (
        value.reshape(num_sequences, sequence_length, -1)
        for value in position_embeddings
    )
    query, key = _apply_vision_rope(query, key, cos, sin)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    outputs = []
    for index in range(num_sequences):
        output = F.scaled_dot_product_attention(
            query[index : index + 1],
            key[index : index + 1],
            value[index : index + 1],
            dropout_p=0.0,
            is_causal=False,
            scale=attention.scaling,
        )
        outputs.append(output.transpose(1, 2).contiguous().reshape(sequence_length, -1))
    return attention.proj(torch.cat(outputs, dim=0))


class InternVideo3LatentEncoder(nn.Module):
    """Expose final, normalized pre-merger InternVideo3 patch tokens."""

    embed_dim = INTERNVIDEO3_EMBED_DIM
    patch_size = INTERNVIDEO3_PATCH_SIZE
    tubelet_size = INTERNVIDEO3_TUBELET_SIZE
    num_frames = INTERNVIDEO3_NUM_FRAMES
    num_views = 2
    grid_depth = INTERNVIDEO3_GRID_DEPTH
    grid_height = INTERNVIDEO3_GRID_SIZE
    grid_width = INTERNVIDEO3_GRID_SIZE
    patches_per_step = INTERNVIDEO3_PATCHES_PER_STEP
    num_patches_per_view = grid_depth * patches_per_step
    num_patches = num_views * num_patches_per_view

    def __init__(
        self,
        backbone: nn.Module,
        final_norm: nn.Module,
        *,
        video_microbatch_size: int | None = None,
    ) -> None:
        super().__init__()
        if video_microbatch_size is not None and video_microbatch_size <= 0:
            raise ValueError("video_microbatch_size must be positive")
        config = getattr(backbone, "config", None)
        if config is None:
            raise ValueError("InternVideo3 backbone must expose its vision config")
        expected = {
            "hidden_size": self.embed_dim,
            "depth": 27,
            "num_heads": 16,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.tubelet_size,
            "spatial_merge_size": INTERNVIDEO3_SPATIAL_MERGE_SIZE,
        }
        mismatches = {
            name: (getattr(config, name, None), value)
            for name, value in expected.items()
            if getattr(config, name, None) != value
        }
        if mismatches:
            raise ValueError(f"InternVideo3 vision configuration mismatch: {mismatches}")

        self.backbone = backbone
        self.final_norm = final_norm
        self.video_microbatch_size = video_microbatch_size
        self.encoder_family = "internvideo3"
        self._vjepa_policy_family = "internvideo3"
        self._vjepa_policy_model_name = "internvideo3_8b_vision_premerger"
        self._vjepa_policy_checkpoint_key = "model.visual"
        self.backbone.blocks[0].attn.interpolate_rope = False
        self.backbone.blocks[0].attn.corrected_frequency_pairing = False

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    @property
    def blocks(self):
        return self.backbone.blocks

    def train(self, mode: bool = True) -> InternVideo3LatentEncoder:
        super().train(False)
        return self

    def _validate_clips(self, clips: torch.Tensor) -> None:
        if clips.ndim != 5:
            raise ValueError(f"clips must be [B,C,T,H,W], got {tuple(clips.shape)}")
        if tuple(clips.shape[1:]) != (3, self.num_frames, 224, 224):
            raise ValueError(
                "InternVideo3 clips must be [B,3,10,224,224], got "
                f"{tuple(clips.shape)}"
            )
        if not torch.is_floating_point(clips):
            raise TypeError("InternVideo3 clips must be floating point")

    @staticmethod
    def _canonical_masks(
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

    def _patchify(self, clips: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = clips.shape
        grid_t = frames // self.tubelet_size
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        merge = INTERNVIDEO3_SPATIAL_MERGE_SIZE
        patches = clips.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            grid_t,
            self.tubelet_size,
            channels,
            grid_h // merge,
            merge,
            self.patch_size,
            grid_w // merge,
            merge,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(
            batch_size * grid_t * grid_h * grid_w,
            channels * self.tubelet_size * self.patch_size**2,
        )

    def _restore_row_major(
        self,
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        temporal_steps: int,
    ) -> torch.Tensor:
        merge = INTERNVIDEO3_SPATIAL_MERGE_SIZE
        hidden_states = hidden_states.reshape(
            batch_size,
            temporal_steps,
            self.grid_height // merge,
            self.grid_width // merge,
            merge,
            merge,
            self.embed_dim,
        )
        return (
            hidden_states.permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(
                batch_size,
                temporal_steps,
                self.patches_per_step,
                self.embed_dim,
            )
            .contiguous()
        )

    def _encode_batch(self, clips: torch.Tensor) -> torch.Tensor:
        batch_size = clips.shape[0]
        temporal_steps = clips.shape[2] // self.tubelet_size
        grid_thw = torch.tensor(
            [[temporal_steps, self.grid_height, self.grid_width]],
            device=clips.device,
            dtype=torch.long,
        ).expand(batch_size, -1)

        hidden_states = self.backbone.patch_embed(self._patchify(clips))
        hidden_states = hidden_states + self.backbone.fast_pos_embed_interpolate(
            grid_thw
        )
        rotary = self.backbone.rot_pos_emb(grid_thw).reshape(
            hidden_states.shape[0], -1
        )
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())

        num_sequences = batch_size * temporal_steps
        for block in self.backbone.blocks:
            hidden_states = hidden_states + _fixed_grid_attention(
                block.attn,
                block.norm1(hidden_states),
                position_embeddings,
                num_sequences=num_sequences,
                sequence_length=self.patches_per_step,
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        hidden_states = self.final_norm(hidden_states)
        return self._restore_row_major(
            hidden_states,
            batch_size=batch_size,
            temporal_steps=temporal_steps,
        )

    @torch.no_grad()
    def _encode(self, clips: torch.Tensor) -> torch.Tensor:
        microbatch = self.video_microbatch_size
        if microbatch is None or clips.shape[0] <= microbatch:
            return self._encode_batch(clips)
        return torch.cat(
            [
                self._encode_batch(clips[start : start + microbatch])
                for start in range(0, clips.shape[0], microbatch)
            ],
            dim=0,
        )

    def forward(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._validate_clips(clips)
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
            if temporal_ids.numel() == 0 or int(temporal_ids[-1]) >= self.grid_depth:
                raise ValueError("mask contains an out-of-range InternVideo3 token index")

        frame_ids = torch.stack(
            (self.tubelet_size * temporal_ids, self.tubelet_size * temporal_ids + 1),
            dim=1,
        ).flatten()
        selected_clips = clips.index_select(2, frame_ids)
        patches = self._encode(selected_clips)
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


def load_internvideo3_latent_encoder(
    pretrained_path: str | Path,
    *,
    video_microbatch_size: int | None = None,
) -> InternVideo3LatentEncoder:
    """Load only the pinned local InternVideo3 visual tower and its final norm."""
    path = Path(pretrained_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            "pretrained_path must be an existing local InternVideo3 directory: "
            f"{path}"
        )
    path = path.resolve()
    required = (
        "config.json",
        "configuration_internvideo3.py",
        "modeling_internvideo3.py",
        _VISION_SHARD,
        _PROJECTOR_SHARD,
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete local InternVideo3 checkpoint: {missing}")

    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as error:
        raise ImportError(
            "loading InternVideo3 requires transformers and safetensors"
        ) from error

    config = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    vision_class = get_class_from_dynamic_module(
        "modeling_internvideo3.InternVideo3VisionModel",
        str(path),
        local_files_only=True,
    )
    backbone = vision_class._from_config(
        config.vision_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    final_norm = backbone.merger.norm
    backbone.merger = nn.Identity()

    raw_state = load_file(str(path / _VISION_SHARD), device="cpu")
    vision_state = {
        name.removeprefix(_VISION_PREFIX): value
        for name, value in raw_state.items()
        if name.startswith(_VISION_PREFIX)
    }
    backbone.load_state_dict(vision_state, strict=True)
    del raw_state, vision_state

    with safe_open(path / _PROJECTOR_SHARD, framework="pt", device="cpu") as shard:
        norm_state = {
            "weight": shard.get_tensor(f"{_VISION_PREFIX}merger.norm.weight"),
            "bias": shard.get_tensor(f"{_VISION_PREFIX}merger.norm.bias"),
        }
    final_norm.load_state_dict(norm_state, strict=True)
    return InternVideo3LatentEncoder(
        backbone,
        final_norm,
        video_microbatch_size=video_microbatch_size,
    )


__all__ = [
    "INTERNVIDEO3_EMBED_DIM",
    "InternVideo3LatentEncoder",
    "load_internvideo3_latent_encoder",
]
