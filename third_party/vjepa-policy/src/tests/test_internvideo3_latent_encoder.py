from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.experimental.latent_space.internvideo3_encoder import (
    INTERNVIDEO3_EMBED_DIM,
    InternVideo3LatentEncoder,
    _apply_vision_rope,
    _fixed_grid_attention,
)
from vjepa_policy.models.vjepa_policy import encode_video_views


class ZeroQKV(nn.Module):
    def forward(self, hidden_states):
        return hidden_states.new_zeros(hidden_states.shape[0], 3 * hidden_states.shape[1])


class ZeroMLP(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.scaling = (INTERNVIDEO3_EMBED_DIM // self.num_heads) ** -0.5
        self.qkv = ZeroQKV()
        self.proj = nn.Identity()


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn = FakeAttention()
        self.mlp = ZeroMLP()


class FakePatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, patches):
        self.calls.append(patches.shape[0])
        return patches[:, :1].expand(-1, INTERNVIDEO3_EMBED_DIM).contiguous()


class FakeInternVideo3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=1152,
            depth=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )
        self.patch_embed = FakePatchEmbed()
        self.blocks = nn.ModuleList([FakeBlock()])
        self.anchor = nn.Parameter(torch.ones(()))

    def fast_pos_embed_interpolate(self, grid_thw):
        count = int(grid_thw.prod(dim=1).sum())
        return self.anchor.new_zeros(count, INTERNVIDEO3_EMBED_DIM)

    def rot_pos_emb(self, grid_thw):
        count = int(grid_thw.prod(dim=1).sum())
        return self.anchor.new_zeros(count, 36)


def make_patch_ramp_clips(batch_size=1):
    temporal = 10_000 * torch.arange(5).view(5, 1, 1)
    rows = 100 * torch.arange(14).view(1, 14, 1)
    columns = torch.arange(14).view(1, 1, 14)
    values = temporal + rows + columns
    images = values.repeat_interleave(16, 1).repeat_interleave(16, 2)
    frames = images.repeat_interleave(2, 0)
    clips = frames.unsqueeze(0).unsqueeze(0).expand(batch_size, 3, -1, -1, -1)
    return clips.float(), values.float()


def make_encoder(*, video_microbatch_size=None):
    backbone = FakeInternVideo3Backbone()
    encoder = InternVideo3LatentEncoder(
        backbone,
        nn.Identity(),
        video_microbatch_size=video_microbatch_size,
    )
    return encoder, backbone


def test_premerger_tokens_are_restored_to_row_major_t_h_w():
    encoder, backbone = make_encoder()
    clips, expected = make_patch_ramp_clips()

    output = encoder(clips)

    assert output.shape == (1, 5 * 14 * 14, 1152)
    torch.testing.assert_close(output[0, :, 0].reshape(5, 14, 14), expected)
    assert backbone.patch_embed.calls == [5 * 14 * 14]
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False


def test_context_mask_encodes_only_first_tubelet_and_orders_views():
    encoder, backbone = make_encoder()
    clips, expected = make_patch_ramp_clips()
    two_views = torch.stack((clips, clips + 0.5), dim=1)
    context_mask = torch.arange(2 * 14 * 14).unsqueeze(0)

    output = encode_video_views(encoder, two_views, context_mask)

    assert output.shape == (1, 2 * 14 * 14, 1152)
    expected_views = torch.cat((expected[0].flatten(), expected[0].flatten() + 0.5))
    torch.testing.assert_close(output[0, :, 0], expected_views)
    assert backbone.patch_embed.calls == [2 * 14 * 14]


def test_video_microbatching_limits_backbone_batch_without_changing_tokens():
    clips, _ = make_patch_ramp_clips(batch_size=5)
    reference, _ = make_encoder()
    expected = reference(clips)
    encoder, backbone = make_encoder(video_microbatch_size=2)

    output = encoder(clips)

    torch.testing.assert_close(output, expected)
    assert backbone.patch_embed.calls == [2 * 5 * 14 * 14, 2 * 5 * 14 * 14, 5 * 14 * 14]


def test_vectorized_sdpa_matches_independent_sequence_attention():
    torch.manual_seed(7)
    hidden_size, num_heads = 16, 4
    attention = SimpleNamespace(
        qkv=nn.Linear(hidden_size, 3 * hidden_size),
        proj=nn.Linear(hidden_size, hidden_size),
        num_heads=num_heads,
        scaling=(hidden_size // num_heads) ** -0.5,
    )
    num_sequences, sequence_length = 3, 5
    hidden = torch.randn(num_sequences * sequence_length, hidden_size)
    rotary = torch.randn(num_sequences * sequence_length, hidden_size // num_heads)
    position_embeddings = (rotary.cos(), rotary.sin())

    actual = _fixed_grid_attention(
        attention,
        hidden,
        position_embeddings,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
    )

    query, key, value = (
        attention.qkv(hidden)
        .reshape(-1, 3, num_heads, hidden_size // num_heads)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    query, key = _apply_vision_rope(
        query,
        key,
        position_embeddings[0],
        position_embeddings[1],
    )
    outputs = []
    for index in range(num_sequences):
        start = index * sequence_length
        stop = start + sequence_length
        outputs.append(
            F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                key[start:stop].transpose(0, 1).unsqueeze(0),
                value[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                scale=attention.scaling,
            )
            .transpose(1, 2)
            .reshape(sequence_length, hidden_size)
        )
    expected = attention.proj(torch.cat(outputs))
    torch.testing.assert_close(actual, expected)
