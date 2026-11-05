import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vjepa_policy.experimental.latent_space.dino_encoder import (
    DinoLatentEncoder,
    load_dino_latent_encoder,
)
from vjepa_policy.models.vjepa_policy import encode_video_views


class FakeDinoBackbone(nn.Module):
    def __init__(self, family, *, spatial_ramp=False):
        super().__init__()
        self.family = family
        self.spatial_ramp = spatial_ramp
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = []
        if family == "dinov2":
            patch_size, registers = 14, 0
        else:
            patch_size, registers = 16, 4
        self.config = SimpleNamespace(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            patch_size=patch_size,
            num_register_tokens=registers,
        )

    def forward(self, *, pixel_values):
        self.calls.append(pixel_values.detach().clone())
        batch_size = pixel_values.shape[0]
        grid = 16 if self.family == "dinov2" else 14
        prefix = 1 if self.family == "dinov2" else 5
        if self.spatial_ramp:
            rows = torch.arange(grid, device=pixel_values.device).view(grid, 1)
            columns = torch.arange(grid, device=pixel_values.device).view(1, grid)
            values = (100 * rows + columns).flatten().float()
            values = values.unsqueeze(0).expand(batch_size, -1)
        else:
            image_ids = torch.arange(batch_size, device=pixel_values.device).unsqueeze(
                1
            )
            patch_ids = torch.arange(grid * grid, device=pixel_values.device).unsqueeze(
                0
            )
            values = image_ids * 1000 + patch_ids
        patch_tokens = values.unsqueeze(-1).expand(-1, -1, 1024).float()
        prefix_tokens = torch.full(
            (batch_size, prefix, 1024),
            -10_000.0,
            device=pixel_values.device,
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat((prefix_tokens, patch_tokens), dim=1)
        )


def make_clips():
    time = torch.arange(10).view(1, 1, 1, 10, 1, 1)
    view = torch.arange(2).view(1, 2, 1, 1, 1, 1)
    return (-1.0 + 0.1 * time + 0.01 * view).expand(1, 2, 3, 10, 224, 224)


def test_dinov3_selects_normalizes_and_orders_two_views():
    backbone = FakeDinoBackbone("dinov3")
    encoder = DinoLatentEncoder(backbone, "dinov3")

    output = encoder(make_clips())

    assert output.shape == (1, 1960, 1024)
    assert encoder.embed_dim == 1024
    assert encoder.patch_size == 16
    assert encoder.tubelet_size == 2
    assert encoder.training is False
    assert backbone.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False
    assert backbone.training is False

    selected = torch.tensor([1, 3, 5, 7, 9])
    source_values = torch.cat((-1.0 + 0.1 * selected, -1.0 + 0.1 * selected + 0.01))
    rgb = (source_values + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    expected_pixels = (rgb[:, None] - mean[None]) / std[None]
    torch.testing.assert_close(
        backbone.calls[0][:, :, 0, 0], expected_pixels, rtol=0, atol=1e-6
    )

    expected_token_ids = []
    for temporal in range(5):
        for view_index in range(2):
            image_index = view_index * 5 + temporal
            expected_token_ids.append(image_index * 1000 + torch.arange(14 * 14))
    expected_token_ids = torch.cat(expected_token_ids).float()
    torch.testing.assert_close(output[0, :, 0], expected_token_ids, rtol=0, atol=0)
    assert torch.all(output >= 0), "CLS/register tokens must not enter policy latents"


def test_dinov2_endpoint_aligned_interpolation_is_16_to_14():
    backbone = FakeDinoBackbone("dinov2", spatial_ramp=True)
    encoder = DinoLatentEncoder(backbone, "dinov2")
    clips = torch.zeros(1, 3, 10, 224, 224)

    output = encoder(clips)

    assert output.shape == (1, 5 * 14 * 14, 1024)
    coordinates = torch.arange(14).float() * 15.0 / 13.0
    expected_grid = 100 * coordinates[:, None] + coordinates[None, :]
    torch.testing.assert_close(
        output[0, : 14 * 14, 0].reshape(14, 14),
        expected_grid,
        rtol=0,
        atol=3e-4,
    )
    assert output[0, 0, 0] == 0
    assert output[0, 14 * 14 - 1, 0] == 1515


def test_masked_encode_video_views_only_encodes_context_endpoints():
    backbone = FakeDinoBackbone("dinov3")
    encoder = DinoLatentEncoder(backbone, "dinov3")
    clips = make_clips()
    context_mask = torch.arange(2 * 14 * 14).unsqueeze(0)

    output = encode_video_views(encoder, clips, context_mask)

    assert output.shape == (1, 2 * 14 * 14, 1024)
    assert backbone.calls[0].shape[0] == 2
    selected_values = torch.tensor([-0.9, -0.89])
    expected_red = ((selected_values + 1.0) / 2.0 - 0.485) / 0.229
    torch.testing.assert_close(
        backbone.calls[0][:, 0, 0, 0], expected_red, rtol=0, atol=1e-6
    )
    expected = torch.cat((torch.arange(196), 1000 + torch.arange(196))).float()
    torch.testing.assert_close(output[0, :, 0], expected, rtol=0, atol=0)


def test_image_microbatching_preserves_tokens_and_limits_backbone_batch():
    clips = torch.zeros(2, 3, 10, 224, 224)
    reference = DinoLatentEncoder(
        FakeDinoBackbone("dinov3", spatial_ramp=True),
        "dinov3",
    )(clips)
    backbone = FakeDinoBackbone("dinov3", spatial_ramp=True)
    encoder = DinoLatentEncoder(
        backbone,
        "dinov3",
        image_microbatch_size=3,
    )

    output = encoder(clips)

    assert output.shape == (2, 5 * 14 * 14, 1024)
    torch.testing.assert_close(output, reference)
    assert [call.shape[0] for call in backbone.calls] == [3, 3, 3, 1]
    assert len(encoder.blocks) == 1
    assert encoder.blocks[0].attn.interpolate_rope is False


def test_loader_uses_only_an_explicit_local_path(tmp_path, monkeypatch):
    calls = []

    class FakeDinov2Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return FakeDinoBackbone("dinov2")

    class FakeDinov3Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return FakeDinoBackbone("dinov3")

    transformers = ModuleType("transformers")
    transformers.Dinov2Model = FakeDinov2Model
    transformers.DINOv3ViTModel = FakeDinov3Model
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    encoder = load_dino_latent_encoder("dinov2", tmp_path)

    assert isinstance(encoder, DinoLatentEncoder)
    assert calls == [(str(tmp_path.resolve()), {"local_files_only": True})]
    with pytest.raises(FileNotFoundError, match="existing local"):
        load_dino_latent_encoder("dinov2", "facebook/dinov2-large")


def test_adapter_rejects_a_non_vitl_or_wrong_register_configuration():
    backbone = FakeDinoBackbone("dinov3")
    backbone.config.num_register_tokens = 0

    with pytest.raises(ValueError, match="num_register_tokens"):
        DinoLatentEncoder(backbone, "dinov3")
