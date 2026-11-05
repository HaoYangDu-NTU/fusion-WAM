"""Policy glue for latent encoders whose token geometry differs from the crop.

The production policy derives a multi-view mask grid from the raw image crop
and from an even RGB frame count.  The geometry experiments deliberately break
those assumptions: DINOv2 has a native 16x16 grid at 224px, while the Wan
stride-2 adapter consumes 17 RGB frames but exposes only five VAE time steps.
This module keeps the loss and Action Expert contract identical and changes
only the encoder-view dispatch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vjepa_policy.models.action_expert import ActionExpert
from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.vjepa_policy import (
    VJEPAPolicy,
    _local_view_masks,
    encode_proprio,
    encoded_proprio_dim,
)


class GeometryActionExpert(ActionExpert):
    """Action Expert with a small, experiment-local checkpointing budget."""

    def __init__(self, *args, activation_checkpointing_blocks=0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= activation_checkpointing_blocks <= self.num_layers:
            raise ValueError(
                "action activation_checkpointing_blocks must be between 0 "
                "and num_layers"
            )
        self.activation_checkpointing_blocks = activation_checkpointing_blocks

    def _run_blocks(
        self,
        tokens,
        predictor_kv,
        context_len,
        t_cond,
        condition_tokens,
        condition_valid,
    ):
        if len(predictor_kv) != self.num_layers:
            raise ValueError(
                f"expert has {self.num_layers} blocks, "
                f"predictor supplied {len(predictor_kv)}"
            )
        hidden = tokens
        for index, block in enumerate(self.blocks):
            ctx_k, ctx_v = predictor_kv[index]
            if context_len is not None:
                ctx_k = ctx_k[:, :, :context_len]
                ctx_v = ctx_v[:, :, :context_len]
            block_args = (
                hidden,
                ctx_k,
                ctx_v,
                t_cond,
                condition_tokens,
                condition_valid,
            )
            if index < self.activation_checkpointing_blocks and self.training:
                hidden = checkpoint(block, *block_args, use_reentrant=False)
            else:
                hidden = block(*block_args)
        hidden, _ = self.final_norm(hidden, t_cond)
        return self.out_proj(hidden)


def _encoder_patches_per_step(encoder) -> int:
    value = getattr(encoder, "patches_per_step", None)
    if value is None:
        height = getattr(encoder, "grid_height", None)
        width = getattr(encoder, "grid_width", None)
        if height is None or width is None:
            raise ValueError(
                "geometry-aware encoders must expose patches_per_step or "
                "grid_height/grid_width"
            )
        value = int(height) * int(width)
    value = int(value)
    if value <= 0:
        raise ValueError(f"encoder patches_per_step must be positive, got {value}")
    return value


def encode_geometry_video_views(encoder, clips, mask=None):
    """Encode a composite or independent-view clip using encoder geometry.

    Unlike ``models.vjepa_policy.encode_video_views``, this helper does not
    infer the temporal lattice from raw RGB frame count.  The encoder itself
    owns that mapping, which is required for Wan's 17-to-5 temporal VAE path.
    Global masks retain the usual ``t-view-h-w`` ordering and are localized
    before calling a 5D encoder.
    """

    def run_encoder(inputs, selection):
        if getattr(encoder, "_vjepa_policy_training_false", False):
            return encoder(inputs, selection, training=False)
        return encoder(inputs, selection)

    if clips.ndim == 5:
        return run_encoder(clips, mask)
    if clips.ndim != 6:
        raise ValueError(
            "clips must be [B,C,T,H,W] or [B,V,C,T,H,W], "
            f"got {tuple(clips.shape)}"
        )

    batch_size, num_views, channels, frames, height, width = clips.shape
    expected_views = getattr(encoder, "num_views", num_views)
    if expected_views != num_views:
        raise ValueError(
            f"clip view count {num_views} does not match encoder num_views={expected_views}"
        )
    patches_per_view = _encoder_patches_per_step(encoder)
    flat_clips = clips.reshape(
        batch_size * num_views, channels, frames, height, width
    )
    if mask is None:
        local_mask = None
    else:
        local_mask = _local_view_masks(
            mask,
            num_views=num_views,
            patches_per_view=patches_per_view,
        )

    latents = run_encoder(flat_clips, local_mask)
    if latents.ndim != 3 or latents.shape[0] != batch_size * num_views:
        raise ValueError(
            "geometry-aware encoder must return [B*V,K,D], "
            f"got {tuple(latents.shape)}"
        )
    tokens_per_view = latents.shape[1]
    if tokens_per_view % patches_per_view:
        raise ValueError(
            "geometry-aware encoder must return complete spatial grids, "
            f"got {tokens_per_view} tokens with {patches_per_view} per step"
        )
    selected_steps = tokens_per_view // patches_per_view
    feature_dim = latents.shape[-1]
    return (
        latents.reshape(
            batch_size,
            num_views,
            selected_steps,
            patches_per_view,
            feature_dim,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(batch_size, selected_steps * num_views * patches_per_view, feature_dim)
    )


class GeometryVJEPAPolicy(VJEPAPolicy):
    """The standard joint world/action loss with geometry-aware encoding."""

    def __init__(self, *args, action_microbatch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        if action_microbatch_size is not None and action_microbatch_size <= 0:
            raise ValueError("action_microbatch_size must be positive")
        self.action_microbatch_size = action_microbatch_size

    def _action_chunk_forward(
        self,
        predictor_kv,
        actions,
        language,
        language_mask,
        state,
        dof_mask,
    ):
        """Checkpoint one action batch chunk without changing its stochastic path."""
        kv_count = len(predictor_kv) * 2
        flat_inputs = [tensor for pair in predictor_kv for tensor in pair]
        flat_inputs.extend((actions, language, language_mask, state, dof_mask))

        def run(*inputs):
            kv = tuple(
                (inputs[index], inputs[index + 1])
                for index in range(0, kv_count, 2)
            )
            chunk_actions, chunk_language, chunk_mask, chunk_state, chunk_dof = (
                inputs[kv_count:]
            )
            return self.action_expert(
                kv,
                chunk_actions,
                chunk_language,
                chunk_mask,
                chunk_state,
                context_len=self.context_len,
                dof_mask=chunk_dof,
            )

        return checkpoint(run, *flat_inputs, use_reentrant=False)

    def forward(self, batch):
        (
            clips,
            language,
            language_mask,
            masks_enc,
            masks_pred,
            action,
            action_pad,
            state,
        ) = batch
        clips_dtype = clips.dtype
        if clips.ndim == 6 and clips.shape[1] != getattr(self.predictor, "num_views", 1):
            raise ValueError(
                f"clip view count {clips.shape[1]} does not match Predictor "
                f"num_views={getattr(self.predictor, 'num_views', 1)}"
            )
        if state.ndim != 2 or state.shape[-1] != self.raw_proprio_dim:
            raise ValueError(
                f"state must be [B,{self.raw_proprio_dim}], got {tuple(state.shape)}"
            )
        state = encode_proprio(state, self.proprio_encoding)

        with torch.no_grad():
            target_latents = encode_geometry_video_views(self.encoder, clips)
            target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))

        loss_world = clips.new_zeros(())
        cosine = clips.new_zeros(())
        predictor_kv = None
        for mask_enc, mask_pred in zip(masks_enc, masks_pred):
            with torch.no_grad():
                context_latents = encode_geometry_video_views(
                    self.encoder, clips, mask_enc
                )
            prediction, predictor_kv = self.predictor(
                context_latents,
                mask_enc,
                mask_pred,
                language=language,
                language_mask=language_mask,
                proprio=state,
            )
            target = apply_masks(target_latents, [mask_pred])
            per_token = (
                torch.mean(torch.abs(prediction - target) ** self.loss_exp, dim=-1)
                / self.loss_exp
            )
            # Cosine is a detached logging metric. Building its full graph keeps
            # float32 copies of the future lattice alive through Action Expert.
            with torch.no_grad():
                per_token_cosine = F.cosine_similarity(
                    prediction.float(), target.float(), dim=-1
                )
            if self.patch_valid is None:
                loss_world = loss_world + per_token.mean()
                cosine = cosine + per_token_cosine.mean()
            else:
                valid_tokens = self.patch_valid[mask_pred].to(per_token.dtype)
                denominator = valid_tokens.sum().clamp_min(1.0)
                loss_world = loss_world + (per_token * valid_tokens).sum() / denominator
                cosine = cosine + (per_token_cosine * valid_tokens).sum() / denominator
        mask_count = len(masks_enc)
        if mask_count <= 0:
            raise ValueError("at least one context/future mask pair is required")
        loss_world = loss_world / mask_count
        encoder_dtype = target_latents.dtype
        del (
            context_latents,
            per_token,
            per_token_cosine,
            prediction,
            target,
            target_latents,
        )
        if isinstance(batch, list):
            # The DataLoader batch is mutable only on this experimental path;
            # release the raw RGB storage before the action-only forward.
            batch[0] = None
        del clips

        dof_mask = (~action_pad).unsqueeze(-1).expand_as(action).float()
        action_chunk = self.action_microbatch_size or action.shape[0]
        action_numerator = action.new_zeros(())
        action_denominator = action.new_zeros(())
        for start in range(0, action.shape[0], action_chunk):
            stop = min(start + action_chunk, action.shape[0])
            chunk_dof = dof_mask[start:stop]
            if action_chunk < action.shape[0]:
                predicted_velocity, target_velocity = self._action_chunk_forward(
                    tuple(
                        (key[start:stop], value[start:stop])
                        for key, value in predictor_kv
                    ),
                    action[start:stop],
                    language[start:stop],
                    language_mask[start:stop],
                    state[start:stop],
                    chunk_dof,
                )
            else:
                predicted_velocity, target_velocity = self.action_expert(
                    predictor_kv,
                    action,
                    language,
                    language_mask,
                    state,
                    context_len=self.context_len,
                    dof_mask=dof_mask,
                )
            valid_actions = (~action_pad[start:stop]).float()
            per_action = F.mse_loss(
                predicted_velocity, target_velocity, reduction="none"
            ).mean(dim=-1)
            action_numerator = action_numerator + (
                per_action * valid_actions
            ).sum()
            action_denominator = action_denominator + valid_actions.sum()
        loss_action = action_numerator / action_denominator.clamp_min(1.0)
        loss = (
            self.world_loss_weight * loss_world
            + self.action_loss_weight * loss_action
        )
        self.forward_dtypes = {
            "clips": clips_dtype,
            "encoder": encoder_dtype,
            "predictor_k": predictor_kv[0][0].dtype,
            "predictor_v": predictor_kv[0][1].dtype,
            "action": predicted_velocity.dtype,
            "loss": loss.dtype,
        }
        return loss, {
            "loss": loss.detach(),
            "loss_world": loss_world.detach(),
            "loss_action": loss_action.detach(),
            "cos_sim": (cosine / mask_count).detach(),
        }


def build_geometry_policy(
    encoder,
    predictor,
    *,
    action_dim,
    action_chunk_size,
    context_len,
    proprio_dim,
    proprio_encoding="identity",
    action_hidden_size=512,
    action_num_layers=24,
    condition_num_heads=8,
    action_num_inference_steps=10,
    action_activation_checkpointing_blocks=0,
    action_microbatch_size=None,
    loss_exp=1.0,
    world_loss_weight=1.0,
    action_loss_weight=1.0,
    patch_valid=None,
):
    if len(predictor.predictor_blocks) != action_num_layers:
        raise ValueError("predictor depth must equal action_num_layers")
    predictor_attention = predictor.predictor_blocks[0].self_attn
    state_dim = encoded_proprio_dim(proprio_dim, proprio_encoding)
    if predictor.proprio_dim != state_dim:
        raise ValueError(
            f"predictor proprio_dim={predictor.proprio_dim} does not match encoded "
            f"state dimension {state_dim}"
        )
    action_expert = GeometryActionExpert(
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        num_layers=action_num_layers,
        predictor_num_heads=predictor_attention.num_heads,
        predictor_head_dim=predictor_attention.head_dim,
        state_dim=state_dim,
        hidden_size=action_hidden_size,
        lang_dim=predictor.lang_dim,
        condition_num_heads=condition_num_heads,
        num_inference_steps=action_num_inference_steps,
        activation_checkpointing_blocks=action_activation_checkpointing_blocks,
    )
    return GeometryVJEPAPolicy(
        encoder,
        predictor,
        action_expert,
        context_len=context_len,
        proprio_dim=proprio_dim,
        proprio_encoding=proprio_encoding,
        loss_exp=loss_exp,
        world_loss_weight=world_loss_weight,
        action_loss_weight=action_loss_weight,
        patch_valid=patch_valid,
        action_microbatch_size=action_microbatch_size,
    )


__all__ = [
    "GeometryActionExpert",
    "GeometryVJEPAPolicy",
    "build_geometry_policy",
    "encode_geometry_video_views",
]
