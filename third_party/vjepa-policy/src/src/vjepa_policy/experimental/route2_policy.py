"""Policy integration for the original Route 2 Action Expert ablation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.experimental.legacy_action_expert import ActionExpert
from vjepa_policy.models.backbone.masks import apply_masks


def append_proprio_to_context(proprio_encoder, context, context_mask, proprio):
    """Append one projected proprio token to the language condition."""
    if proprio.ndim != 2:
        raise ValueError(f"proprio must be [B,D], got {tuple(proprio.shape)}")
    proprio_token = proprio_encoder(
        proprio.to(device=context.device, dtype=proprio_encoder.weight.dtype)
    ).unsqueeze(1)
    proprio_token = proprio_token.to(dtype=context.dtype)
    proprio_mask = torch.ones(
        (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
    )
    return (
        torch.cat((context, proprio_token), dim=1),
        torch.cat((context_mask, proprio_mask), dim=1),
    )


class Route2BaselinePolicy(nn.Module):
    """Frozen encoder, cross-conditioned Predictor, and FiLM Action Expert."""

    def __init__(
        self,
        encoder,
        predictor,
        action_expert,
        *,
        context_len,
        proprio_dim,
        loss_exp=1.0,
        world_loss_weight=1.0,
        action_loss_weight=1.0,
        patch_valid=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_expert = action_expert
        self.proprio_encoder = nn.Linear(proprio_dim, predictor.lang_dim)
        self.context_len = context_len
        self.loss_exp = loss_exp
        self.world_loss_weight = world_loss_weight
        self.action_loss_weight = action_loss_weight
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if patch_valid is None:
            self.patch_valid = None
        else:
            self.register_buffer("patch_valid", patch_valid.bool(), persistent=False)

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

        with torch.no_grad():
            target_latents = self.encoder(clips)
            target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))

        loss_world = clips.new_zeros(())
        cosine = clips.new_zeros(())
        predictor_kv = None
        for mask_enc, mask_pred in zip(masks_enc, masks_pred):
            with torch.no_grad():
                context_latents = self.encoder(clips, mask_enc)
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
            token_cosine = F.cosine_similarity(
                prediction.float(), target.float(), dim=-1
            )
            if self.patch_valid is None:
                loss_world = loss_world + per_token.mean()
                cosine = cosine + token_cosine.mean()
            else:
                valid = self.patch_valid[mask_pred].to(per_token.dtype)
                denominator = valid.sum().clamp_min(1.0)
                loss_world = loss_world + (per_token * valid).sum() / denominator
                cosine = cosine + (token_cosine * valid).sum() / denominator
        mask_count = len(masks_enc)
        loss_world = loss_world / mask_count

        action_language, action_language_mask = append_proprio_to_context(
            self.proprio_encoder, language, language_mask, state
        )
        dof_mask = (~action_pad).unsqueeze(-1).expand_as(action).float()
        predicted_velocity, target_velocity = self.action_expert(
            predictor_kv,
            action,
            action_language,
            action_language_mask,
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
        loss = (
            self.world_loss_weight * loss_world + self.action_loss_weight * loss_action
        )
        self.forward_dtypes = {
            "clips": clips.dtype,
            "encoder": target_latents.dtype,
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


def build_route2_baseline_policy(
    encoder,
    predictor,
    *,
    action_dim,
    action_chunk_size,
    context_len,
    proprio_dim,
    action_hidden_size=512,
    action_num_layers=24,
    action_num_inference_steps=10,
    loss_exp=1.0,
    world_loss_weight=1.0,
    action_loss_weight=1.0,
    patch_valid=None,
):
    if len(predictor.predictor_blocks) != action_num_layers:
        raise ValueError("predictor depth must equal action_num_layers")
    predictor_attention = predictor.predictor_blocks[0].self_attn
    action_expert = ActionExpert(
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        num_layers=action_num_layers,
        num_heads=predictor_attention.num_heads,
        head_dim=predictor_attention.head_dim,
        hidden_size=action_hidden_size,
        lang_dim=predictor.lang_dim,
        num_inference_steps=action_num_inference_steps,
    )
    return Route2BaselinePolicy(
        encoder,
        predictor,
        action_expert,
        context_len=context_len,
        proprio_dim=proprio_dim,
        loss_exp=loss_exp,
        world_loss_weight=world_loss_weight,
        action_loss_weight=action_loss_weight,
        patch_valid=patch_valid,
    )
