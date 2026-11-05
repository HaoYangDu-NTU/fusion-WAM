"""V-JEPA policy with latent world modeling and flow-matching actions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.action_expert import ActionExpert
from vjepa_policy.state import packed_state_dim, prepare_proprio_state


def encoded_proprio_dim(proprio_dim, encoding="identity"):
    if proprio_dim <= 0:
        raise ValueError("proprio_dim must be positive")
    if encoding == "identity":
        return proprio_dim
    if encoding == "sincos":
        return 2 * proprio_dim
    raise ValueError(f"unsupported proprio encoding {encoding!r}")


def encode_proprio(state, encoding="identity"):
    if encoding == "identity":
        return state
    if encoding == "sincos":
        return torch.cat((torch.sin(state), torch.cos(state)), dim=-1)
    raise ValueError(f"unsupported proprio encoding {encoding!r}")


def _local_view_masks(mask, *, num_views, patches_per_view):
    """Convert global t-view-patch indices to per-view t-patch indices."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be [B,K], got {tuple(mask.shape)}")
    tokens_per_step = num_views * patches_per_view
    temporal = mask // tokens_per_step
    within_step = mask % tokens_per_step
    view = within_step // patches_per_view
    spatial = within_step % patches_per_view
    local = temporal * patches_per_view + spatial

    if mask.shape[1] % num_views:
        raise ValueError("mask token count must be divisible by num_views")
    expected_count = mask.shape[1] // num_views
    view_counts = torch.stack(
        [(view == view_index).sum(dim=1) for view_index in range(num_views)],
        dim=1,
    )
    if not torch.all(view_counts == expected_count):
        raise ValueError("each sample and view must contribute the same number of mask tokens")

    token_positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    per_view = []
    for view_index in range(num_views):
        positions = torch.where(
            view == view_index,
            token_positions,
            mask.shape[1],
        )
        positions = positions.sort(dim=1).values[:, :expected_count]
        per_view.append(torch.gather(local, 1, positions))
    return torch.stack(per_view, dim=1).reshape(mask.shape[0] * num_views, -1)


def encode_video_views(encoder, clips, mask=None):
    """Encode one composite clip or independently encode a stack of camera views."""
    def run_encoder(inputs, selection):
        if getattr(encoder, "_vjepa_policy_training_false", False):
            return encoder(inputs, selection, training=False)
        return encoder(inputs, selection)

    if clips.ndim != 6:
        return run_encoder(clips, mask)

    batch_size, num_views, channels, frames, height, width = clips.shape
    if height % encoder.patch_size or width % encoder.patch_size:
        raise ValueError("view height and width must be divisible by encoder.patch_size")
    if frames % encoder.tubelet_size:
        raise ValueError("frame count must be divisible by encoder.tubelet_size")

    temporal_steps = frames // encoder.tubelet_size
    patches_per_view = (height // encoder.patch_size) * (width // encoder.patch_size)
    flat_clips = clips.reshape(
        batch_size * num_views, channels, frames, height, width
    )
    local_mask = None
    if mask is not None:
        local_mask = _local_view_masks(
            mask,
            num_views=num_views,
            patches_per_view=patches_per_view,
        )

    latents = run_encoder(flat_clips, local_mask)
    tokens_per_view = latents.shape[1]
    if mask is None:
        selected_steps = temporal_steps
        expected_tokens = temporal_steps * patches_per_view
    else:
        if tokens_per_view % patches_per_view:
            raise ValueError("multi-view masks must select complete spatial grids")
        selected_steps = tokens_per_view // patches_per_view
        expected_tokens = selected_steps * patches_per_view
    if tokens_per_view != expected_tokens:
        raise ValueError(
            f"encoder returned {tokens_per_view} tokens per view, expected {expected_tokens}"
        )

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


class VJEPAPolicy(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_expert,
        *,
        context_len,
        proprio_dim,
        proprio_encoding="identity",
        max_state_dim=None,
        loss_exp=1.0,
        world_loss_weight=1.0,
        action_loss_weight=1.0,
        patch_valid=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_expert = action_expert
        self.raw_proprio_dim = proprio_dim
        self.proprio_encoding = proprio_encoding
        self.encoded_proprio_dim = encoded_proprio_dim(
            proprio_dim, proprio_encoding
        )
        self.max_state_dim = (
            None if max_state_dim is None or int(max_state_dim) == 0 else int(max_state_dim)
        )
        if self.max_state_dim is not None and proprio_encoding != "identity":
            raise ValueError(
                "fixed-width state packing requires proprio_encoding='identity'; "
                "use max_state_dim=None for the legacy sincos representation"
            )
        if (
            self.max_state_dim is not None
            and self.encoded_proprio_dim > self.max_state_dim
        ):
            raise ValueError(
                f"encoded state dimension {self.encoded_proprio_dim} exceeds "
                f"max_state_dim={self.max_state_dim}"
            )
        self.packed_proprio_dim = (
            self.encoded_proprio_dim
            if self.max_state_dim is None
            else packed_state_dim(self.max_state_dim)
        )
        # Retain the published initialization sequence for exact seeded
        # reproducibility; state conditioning is implemented inside both experts.
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
        self.proprio_encoder = nn.Identity()

    def forward(self, batch):
        clips, language, language_mask, masks_enc, masks_pred, action, action_pad, state = batch
        if clips.ndim == 6 and clips.shape[1] != getattr(self.predictor, "num_views", 1):
            raise ValueError(
                f"clip view count {clips.shape[1]} does not match Predictor "
                f"num_views={getattr(self.predictor, 'num_views', 1)}"
            )
        if state.ndim != 2 or state.shape[-1] != self.raw_proprio_dim:
            raise ValueError(
                f"state must be [B,{self.raw_proprio_dim}], got {tuple(state.shape)}"
            )
        state = prepare_proprio_state(
            state,
            raw_state_dim=self.raw_proprio_dim,
            encoding=self.proprio_encoding,
            max_state_dim=self.max_state_dim,
        )

        with torch.no_grad():
            target_latents = encode_video_views(self.encoder, clips)
            target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))

        loss_world = clips.new_zeros(())
        cosine = clips.new_zeros(())
        predictor_kv = None
        for mask_enc, mask_pred in zip(masks_enc, masks_pred):
            with torch.no_grad():
                context_latents = encode_video_views(self.encoder, clips, mask_enc)
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
        loss_world = loss_world / mask_count

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
            (per_action * valid_actions).sum()
            / valid_actions.sum().clamp_min(1.0)
        )
        loss = (
            self.world_loss_weight * loss_world
            + self.action_loss_weight * loss_action
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


def build_vjepa_policy(
    encoder,
    predictor,
    *,
    action_dim,
    action_chunk_size,
    context_len,
    proprio_dim,
    proprio_encoding="identity",
    max_state_dim=None,
    action_hidden_size=512,
    action_num_layers=24,
    condition_num_heads=8,
    action_num_inference_steps=10,
    loss_exp=1.0,
    world_loss_weight=1.0,
    action_loss_weight=1.0,
    patch_valid=None,
):
    if len(predictor.predictor_blocks) != action_num_layers:
        raise ValueError("predictor depth must equal action_num_layers")
    predictor_attention = predictor.predictor_blocks[0].self_attn
    encoded_state_dim = encoded_proprio_dim(proprio_dim, proprio_encoding)
    if max_state_dim is not None and int(max_state_dim) != 0:
        if proprio_encoding != "identity":
            raise ValueError(
                "fixed-width state packing requires proprio_encoding='identity'; "
                "use max_state_dim=None for the legacy sincos representation"
            )
        if encoded_state_dim > int(max_state_dim):
            raise ValueError(
                f"encoded state dimension {encoded_state_dim} exceeds "
                f"max_state_dim={max_state_dim}"
            )
        state_dim = packed_state_dim(max_state_dim)
    else:
        max_state_dim = None
        state_dim = encoded_state_dim
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
    return VJEPAPolicy(
        encoder,
        predictor,
        action_expert,
        context_len=context_len,
        proprio_dim=proprio_dim,
        proprio_encoding=proprio_encoding,
        max_state_dim=max_state_dim,
        loss_exp=loss_exp,
        world_loss_weight=world_loss_weight,
        action_loss_weight=action_loss_weight,
        patch_valid=patch_valid,
    )
