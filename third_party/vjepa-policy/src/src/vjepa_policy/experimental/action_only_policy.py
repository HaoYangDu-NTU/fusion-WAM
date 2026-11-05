"""Action-only V-JEPA policy that skips the unoptimized world target branch."""

import torch
import torch.nn.functional as F

from vjepa_policy.models.action_expert import ActionExpert
from vjepa_policy.models.vjepa_policy import (
    VJEPAPolicy,
    encode_proprio,
    encode_video_views,
    encoded_proprio_dim,
)


class ActionOnlyVJEPAPolicy(VJEPAPolicy):
    """Train Predictor K/V and the Action Expert without a world-model loss."""

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
        if clips.ndim == 6 and clips.shape[1] != getattr(
            self.predictor, "num_views", 1
        ):
            raise ValueError(
                f"clip view count {clips.shape[1]} does not match Predictor "
                f"num_views={getattr(self.predictor, 'num_views', 1)}"
            )
        if state.ndim != 2 or state.shape[-1] != self.raw_proprio_dim:
            raise ValueError(
                f"state must be [B,{self.raw_proprio_dim}], got {tuple(state.shape)}"
            )
        state = encode_proprio(state, self.proprio_encoding)

        predictor_kv = None
        context_latents = None
        for mask_enc, mask_pred in zip(masks_enc, masks_pred):
            with torch.no_grad():
                context_latents = encode_video_views(self.encoder, clips, mask_enc)
            predictor_kv = self._predictor_forward(
                context_latents,
                mask_enc,
                mask_pred,
                language=language,
                language_mask=language_mask,
                proprio=state,
                return_prediction=False,
            )

        dof_mask = (~action_pad).unsqueeze(-1).expand_as(action).float()
        predicted_velocity, target_velocity = self.action_expert(
            predictor_kv,
            action,
            language,
            language_mask,
            state,
            context_len=self.context_len,
            dof_mask=dof_mask,
        )
        valid_actions = (~action_pad).float()
        per_action = F.mse_loss(
            predicted_velocity, target_velocity, reduction="none"
        ).mean(dim=-1)
        loss_action = (
            per_action * valid_actions
        ).sum() / valid_actions.sum().clamp_min(1.0)
        loss = self.action_loss_weight * loss_action
        zero = loss.detach().new_zeros(())
        self.forward_dtypes = {
            "clips": clips.dtype,
            "encoder": context_latents.dtype,
            "predictor_k": predictor_kv[0][0].dtype,
            "predictor_v": predictor_kv[0][1].dtype,
            "action": predicted_velocity.dtype,
            "loss": loss.dtype,
        }
        return loss, {
            "loss": loss.detach(),
            "loss_world": zero,
            "loss_action": loss_action.detach(),
            "cos_sim": zero,
        }

    def _predictor_forward(
        self,
        context_latents,
        mask_enc,
        mask_pred,
        *,
        language,
        language_mask,
        proprio,
        return_prediction,
    ):
        _, predictor_kv = self.predictor(
            context_latents,
            mask_enc,
            mask_pred,
            language=language,
            language_mask=language_mask,
            proprio=proprio,
            return_prediction=return_prediction,
        )
        return predictor_kv


def build_action_only_policy(
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
    loss_exp=1.0,
    action_loss_weight=1.0,
    patch_valid=None,
    policy_class=ActionOnlyVJEPAPolicy,
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
    action_expert = ActionExpert(
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
    )
    return policy_class(
        encoder,
        predictor,
        action_expert,
        context_len=context_len,
        proprio_dim=proprio_dim,
        proprio_encoding=proprio_encoding,
        loss_exp=loss_exp,
        world_loss_weight=0.0,
        action_loss_weight=action_loss_weight,
        patch_valid=patch_valid,
    )
