from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from openwam.model.video_backbone.encoder.ijepa import IJEPAImageVideoEncoder


class _MockIJEPA(nn.Module):
    def __init__(self, dim=8, patch=16):
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Parameter(torch.ones(1))
        self.patch = patch

    def forward(self, x):
        b, _, h, w = x.shape
        hp, wp = h // self.patch, w // self.patch
        tokens = torch.zeros(b, hp * wp, self.embed_dim, device=x.device, dtype=x.dtype)
        tokens += x.mean(dim=(1, 2, 3), keepdim=False).view(b, 1, 1)
        tokens[:, :, 0] += torch.arange(hp * wp, device=x.device, dtype=x.dtype)
        return tokens


def _encoder(mode="gated_difference", views=False):
    return IJEPAImageVideoEncoder(
        _MockIJEPA(),
        embed_dim=8,
        patch_size=16,
        history_fusion_mode=mode,
        seperate_encode_view=views,
        add_input_bottleneck=False,
    )


@pytest.mark.parametrize(
    ("mode", "expected_t"),
    [("gated_difference", 5), ("two_clean", 6)],
)
def test_ijepa_m29_temporal_shapes(mode, expected_t):
    encoder = _encoder(mode)
    video = torch.zeros(1, 3, 10, 224, 224)
    z = encoder.batch_encode(video)
    assert z.shape == (1, 8, expected_t, 14, 14)
    assert encoder.properties.num_condition_latents == (2 if mode == "two_clean" else 1)


def test_ijepa_gated_difference_is_current_anchor_at_zero_scale():
    encoder = _encoder("gated_difference")
    video = torch.zeros(1, 3, 10, 32, 32)
    video[:, :, 0] = 0.0
    video[:, :, 1] = 1.0
    z = encoder.batch_encode(video)
    raw = encoder._encode_frames(video)
    expected = encoder._normalize(
        torch.cat([raw[:, :, 1:2], raw[:, :, 2:].reshape(1, 8, 4, 2, 2, 2).mean(dim=3)], dim=2)
    )
    torch.testing.assert_close(z, expected)


def test_ijepa_rejects_non_pair_future_count():
    with pytest.raises(ValueError, match="divisible"):
        _encoder().batch_encode(torch.zeros(1, 3, 9, 224, 224))


def test_ijepa_separate_views_pack_width():
    z = _encoder("two_clean", views=True).batch_encode(torch.zeros(1, 3, 10, 224, 448))
    assert z.shape == (1, 8, 6, 14, 28)


def test_ijepa_clean_context_is_causal_and_matches_inference():
    encoder = _encoder()
    with torch.no_grad():
        encoder.delta_scale.fill_(0.3)
    grid = torch.randn(2, 8, 10, 2, 4)
    training = encoder._normalize(encoder._adapt_frame_grid(grid))
    inference = encoder._normalize(encoder._adapt_frame_grid(grid[:, :, :2]))
    torch.testing.assert_close(training[:, :, :1], inference, rtol=0, atol=0)
    changed = grid.clone()
    changed[:, :, 2:] = torch.randn_like(changed[:, :, 2:]) * 100
    torch.testing.assert_close(
        encoder._adapt_frame_grid(grid)[:, :, :1],
        encoder._adapt_frame_grid(changed)[:, :, :1], rtol=0, atol=0,
    )
    changed[:, :, 0] += torch.randn_like(changed[:, :, 0])
    assert not torch.equal(
        encoder._adapt_frame_grid(grid)[:, :, :1],
        encoder._adapt_frame_grid(changed)[:, :, :1],
    )


def test_ijepa_zero_gate_can_start_learning_with_projection_initially_zero_grad():
    encoder = _encoder()
    grid = torch.randn(2, 8, 10, 2, 4)
    probe = torch.randn(2, 8, 1, 2, 4)
    loss = (encoder._normalize(encoder._adapt_frame_grid(grid))[:, :, :1] * probe).sum()
    loss.backward()
    assert torch.isfinite(encoder.delta_scale.grad) and encoder.delta_scale.grad.abs() > 0
    assert torch.count_nonzero(encoder._delta_proj.weight.grad) == 0
    with torch.no_grad():
        encoder.delta_scale.add_(-0.01 * encoder.delta_scale.grad)
    encoder.zero_grad(set_to_none=True)
    (encoder._normalize(encoder._adapt_frame_grid(grid))[:, :, :1] * probe).sum().backward()
    assert encoder._delta_proj.weight.grad.abs().sum() > 0


def test_ijepa_future_targets_follow_exact_native_pairs():
    encoder = _encoder()
    grid = torch.randn(2, 8, 10, 2, 4)
    output = encoder._adapt_frame_grid(grid)
    for index in range(4):
        expected = grid[:, :, 2 + 2 * index:4 + 2 * index].mean(dim=2)
        torch.testing.assert_close(output[:, :, index + 1], expected, rtol=0, atol=0)


def test_ijepa_skeleton_rebuilds_for_strict_checkpoint_load(monkeypatch, tmp_path):
    import openwam.model.video_backbone.encoder.ijepa as ijepa

    calls = []

    def make_vit(name, *, img_size, patch_size):
        calls.append((name, img_size, patch_size))
        return _MockIJEPA(patch=patch_size)

    monkeypatch.setattr(ijepa, "_load_official_vit", make_vit)
    original = _encoder("two_clean", views=True)
    with torch.no_grad():
        original.delta_scale.fill_(0.5)
        original._m.proj.fill_(7.0)
    skeleton = IJEPAImageVideoEncoder.from_skeleton(
        {},
        encoder_cfg={
            "model_name": "vit_giant", "img_size": 224, "patch_size": 16,
            "history_fusion_mode": "two_clean", "seperate_encode_view": True,
            "add_input_bottleneck": False, "model_path": "/unavailable",
        },
        ckpt_dir=str(tmp_path),
    )
    skeleton.load_state_dict(original.state_dict(), strict=True)
    assert calls == [("vit_giant", 224, 16)]
    assert skeleton.properties == original.properties
    assert skeleton.processing_config == original.processing_config
    for key, value in original.state_dict().items():
        torch.testing.assert_close(skeleton.state_dict()[key], value)


def test_ijepa_skeleton_requires_svae_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="svae_config.json"):
        IJEPAImageVideoEncoder.from_skeleton(
            {}, encoder_cfg={"svae_path": "/unavailable/svae.pt"}, ckpt_dir=str(tmp_path),
        )
