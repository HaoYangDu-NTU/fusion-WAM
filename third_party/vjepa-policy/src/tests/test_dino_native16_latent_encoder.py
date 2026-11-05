from types import SimpleNamespace

import torch
import torch.nn as nn

from vjepa_policy.experimental.latent_space.dino_native16 import (
    DinoV2Native16LatentEncoder,
    build_dinov2_native16_policy_mask,
)
from vjepa_policy.models.vjepa_policy import encode_video_views


class FakeNativeDinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = []
        self.config = SimpleNamespace(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            patch_size=14,
            num_register_tokens=0,
        )

    def forward(self, *, pixel_values):
        self.calls.append(pixel_values.detach().clone())
        batch_size = pixel_values.shape[0]
        rows = torch.arange(16, device=pixel_values.device).view(16, 1)
        columns = torch.arange(16, device=pixel_values.device).view(1, 16)
        values = (100 * rows + columns).flatten().float()
        values = values.unsqueeze(0).expand(batch_size, -1)
        patch_tokens = values.unsqueeze(-1).expand(-1, -1, 1024)
        cls = torch.full((batch_size, 1, 1024), -10_000.0, device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patch_tokens), dim=1))


def make_clips():
    return torch.zeros(1, 2, 3, 10, 224, 224)


def test_native16_returns_uninterpolated_two_view_grid():
    backbone = FakeNativeDinoBackbone()
    encoder = DinoV2Native16LatentEncoder(backbone)

    output = encoder(make_clips())

    assert output.shape == (1, 5 * 2 * 16 * 16, 1024)
    assert encoder.patch_size == 14
    assert encoder._vjepa_policy_latent_grid == (5, 16, 16)
    positions = torch.arange(16 * 16, dtype=torch.float32)
    expected = 100 * (positions // 16) + positions.remainder(16)
    torch.testing.assert_close(output[0, : 16 * 16, 0], expected, rtol=0, atol=0)


def test_native16_multiview_mask_localization_uses_native_grid():
    backbone = FakeNativeDinoBackbone()
    encoder = DinoV2Native16LatentEncoder(backbone)
    mask = build_dinov2_native16_policy_mask(num_views=2)

    context = mask.ctx_idx.unsqueeze(0)
    output = encode_video_views(encoder, make_clips(), context)

    assert mask.n_ctx == 2 * 16 * 16
    assert mask.tgt_idx.numel() == 4 * 2 * 16 * 16
    assert output.shape == (1, mask.n_ctx, 1024)
    # Only the first endpoint of each view is needed for the one-tubelet context.
    assert [call.shape[0] for call in backbone.calls] == [2]
    positions = torch.arange(16 * 16, dtype=torch.float32)
    expected_one_view = 100 * (positions // 16) + positions.remainder(16)
    expected = torch.cat((expected_one_view, expected_one_view))
    torch.testing.assert_close(output[0, :, 0], expected, rtol=0, atol=0)


def test_native16_mask_rejects_nonmatching_geometry():
    try:
        build_dinov2_native16_policy_mask(image_size=(224, 224))
    except ValueError as error:
        assert "256" in str(error)
    else:
        raise AssertionError("expected logical native16 geometry validation")
