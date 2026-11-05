from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vjepa_policy.experimental.latent_space import builders


def make_baseline_args(**overrides):
    values = {
        "encoder_family": "vjepa2",
        "model_name": "vit_large",
        "encoder_checkpoint_key": "target_encoder",
        "crop_size": 224,
        "patch_size": 16,
        "tubelet_size": 2,
        "context_tubelets": 1,
        "view_layout": "independent",
        "pred_depth": 24,
        "pred_embed_dim": 1024,
        "pred_num_heads": 16,
        "action_hidden_size": 512,
        "action_num_layers": 24,
        "condition_num_heads": 8,
        "predictor_rope_frequency_pairing": "corrected",
        "encoder_interpolate_rope": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validation_rejects_non_latent_baseline_drift(monkeypatch):
    monkeypatch.setattr(builders, "_model_image_size", lambda args: (224, 224))
    monkeypatch.setattr(builders, "_model_video_frames", lambda args: 10)
    monkeypatch.setattr(builders, "_camera_keys", lambda args: ["main", "wrist"])

    builders.validate_latent_ablation_config(make_baseline_args())
    with pytest.raises(ValueError, match="pred_depth"):
        builders.validate_latent_ablation_config(make_baseline_args(pred_depth=12))


def test_alternate_encoder_does_not_perturb_post_baseline_rng(monkeypatch):
    args = make_baseline_args(
        latent_encoder="dinov2",
        latent_checkpoint="/weights/dinov2",
        image_microbatch_size=8,
    )
    marker = object()
    monkeypatch.setattr(builders, "validate_latent_ablation_config", lambda args: None)
    monkeypatch.setattr(
        builders,
        "_consume_baseline_encoder_initialization",
        lambda args: torch.rand(7),
    )

    def fake_loader(*args, **kwargs):
        torch.rand(101)
        return marker

    monkeypatch.setattr(builders, "load_dino_latent_encoder", fake_loader)

    torch.manual_seed(17)
    torch.rand(7)
    expected_next = torch.rand(5)
    torch.manual_seed(17)

    assert builders.build_latent_encoder(args) is marker
    torch.testing.assert_close(torch.rand(5), expected_next)


class TinyPredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, 4)
        self.core = nn.Linear(4, 4)
        self.predictor_proj = nn.Linear(4, embed_dim)


@pytest.mark.parametrize(
    ("latent_encoder", "embed_dim"),
    (("wan2_2", 48), ("internvideo3", 1152)),
)
def test_shape_specific_predictor_copies_baseline_core_and_preserves_action_rng(
    monkeypatch, latent_encoder, embed_dim
):
    def fake_builder(**kwargs):
        return TinyPredictor(kwargs["embed_dim"])

    monkeypatch.setattr(builders, "build_world_model_predictor", fake_builder)
    torch.manual_seed(23)
    baseline = fake_builder(embed_dim=1024)
    baseline_state = {
        name: value.detach().clone() for name, value in baseline.state_dict().items()
    }
    expected_rng = torch.get_rng_state().clone()

    torch.manual_seed(23)
    predictor = builders.build_latent_predictor(
        latent_encoder=latent_encoder,
        embed_dim=embed_dim,
    )

    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in predictor.state_dict().items():
        if name not in builders._SHAPE_SPECIFIC_PREDICTOR_KEYS:
            torch.testing.assert_close(value, baseline_state[name], rtol=0, atol=0)
