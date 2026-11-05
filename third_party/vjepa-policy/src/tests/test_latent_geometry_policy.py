import torch
import torch.nn as nn

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.experimental.latent_space.geometry_policy import (
    build_geometry_policy,
    encode_geometry_video_views,
)
from vjepa_policy.models.wm_predictor import build_world_model_predictor


class TinyGeometryEncoder(nn.Module):
    embed_dim = 8
    patch_size = 16
    tubelet_size = 1
    num_views = 2
    patches_per_step = 4
    num_patches_per_view = 5 * patches_per_step
    num_patches = num_views * num_patches_per_view

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def forward(self, clips, mask=None):
        batch = clips.shape[0]
        values = torch.arange(self.num_patches_per_view, device=clips.device).view(1, -1, 1)
        values = values.expand(batch, -1, self.embed_dim).float() + self.anchor * 0
        if mask is None:
            return values
        return torch.stack([values[index, row] for index, row in enumerate(mask)], dim=0)


def make_predictor():
    return build_world_model_predictor(
        img_size=(4, 4),
        patch_size=2,
        num_frames=10,
        tubelet_size=2,
        embed_dim=8,
        predictor_embed_dim=12,
        out_embed_dim=8,
        depth=2,
        num_heads=3,
        num_mask_tokens=2,
        lang_dim=6,
        proprio_dim=3,
        interpolate_rope=False,
        num_views=2,
    )


def test_geometry_view_dispatch_does_not_derive_time_from_raw_frames():
    encoder = TinyGeometryEncoder()
    clips = torch.zeros(1, 2, 3, 17, 8, 8)
    target = encode_geometry_video_views(encoder, clips)
    mask = CausalPatchMask((4, 4), 2, 2, 10, 1, num_views=2)
    context = encode_geometry_video_views(encoder, clips, mask.ctx_idx.unsqueeze(0))

    assert target.shape == (1, 5 * 2 * 4, 8)
    assert context.shape == (1, 2 * 4, 8)


def test_geometry_policy_keeps_world_and_action_losses_joint():
    torch.manual_seed(7)
    encoder = TinyGeometryEncoder()
    predictor = make_predictor()
    policy = build_geometry_policy(
        encoder,
        predictor,
        action_dim=2,
        action_chunk_size=3,
        context_len=2 * 2 * 2,
        proprio_dim=3,
        action_hidden_size=12,
        action_num_layers=2,
        condition_num_heads=3,
        action_num_inference_steps=2,
        action_activation_checkpointing_blocks=1,
        action_microbatch_size=1,
    )
    mask = CausalPatchMask((4, 4), 2, 2, 10, 1, num_views=2)
    batch = (
        torch.randn(2, 2, 3, 17, 8, 8),
        torch.randn(2, 5, 6),
        torch.ones(2, 5, dtype=torch.bool),
        [mask.ctx_idx.unsqueeze(0).expand(2, -1)],
        [mask.tgt_idx.unsqueeze(0).expand(2, -1)],
        torch.randn(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.randn(2, 3),
    )
    loss, metrics = policy(batch)

    assert torch.isfinite(loss)
    assert metrics["loss_world"] > 0
    assert metrics["loss_action"] > 0
    loss.backward()
    assert policy.action_expert.activation_checkpointing_blocks == 1
    assert policy.action_microbatch_size == 1
    assert predictor.predictor_blocks[0].self_attn.qkv.weight.grad is not None
