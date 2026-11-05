"""Distribution statistics, legacy RNG equivalence and real training integration."""

import math
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.model.architectures.utils.timestep_sampling import (
    NOISE_SCHEDULE_DISTRIBUTIONS,
    sample_timestep_ids,
    validate_noise_schedule_distribution,
)
from openwam.train.utils.checkpointing import save_config
from tests.test_dual_system_idm import _CapturePrepareVideoBackbone, _make_idm_with_video
from tests.test_flow_prediction import TinyArchitecture


@pytest.mark.parametrize("low,high", [(0, 1000), (17, 851), (10, 11)])
def test_uniform_preserves_ids_and_rng(low, high):
    torch.manual_seed(42)
    expected = torch.randint(low, high, (64,))
    expected_state = torch.get_rng_state()
    torch.manual_seed(42)
    actual = sample_timestep_ids(1000, 64, low=low, high=high)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_state)


@pytest.mark.parametrize("distribution", NOISE_SCHEDULE_DISTRIBUTIONS)
@pytest.mark.parametrize("low,high", [(0, 1000), (250, 750)])
def test_quantiles_and_index_windows(distribution, low, high):
    torch.manual_seed(731)
    ids = sample_timestep_ids(1000, 65536, distribution=distribution, low=low, high=high)
    assert ids.dtype == torch.int64 and ids.device.type == "cpu"
    assert low <= ids.min() <= ids.max() < high
    # Recover the normalized pre-shift sample; quantization error is <= 1/window.
    b = 1 - (ids.double() - low) / (high - low)
    for q in (0.1, 0.5, 0.9):
        if distribution == "uniform":
            expected = q
        elif distribution == "logit_normal":
            expected = 1 / (1 + math.exp(-NormalDist().inv_cdf(q)))
        elif distribution == "logit_logistic":
            odds = (q / (1 - q)) ** 0.75
            expected = odds / (1 + odds)
        else:
            # Published mode quantile, with decreasing source transform reversed.
            expected = q - 1.29 * (math.sin(math.pi * q / 2) ** 2 - q)
        assert abs(torch.quantile(b, q).item() - expected) < 0.012


@pytest.mark.parametrize("distribution", NOISE_SCHEDULE_DISTRIBUTIONS[1:])
def test_extreme_draws_stay_finite_and_in_range(distribution):
    draws = torch.tensor([0.0, 1e-8, 0.5, 1 - 1e-8, 1.0])
    with patch("torch.rand", return_value=draws), patch("torch.randn", return_value=torch.tensor([-100., -20., 0., 20., 100.])):
        ids = sample_timestep_ids(1000, 5, distribution=distribution, low=200, high=800)
    assert torch.isfinite(ids).all()
    assert ((ids >= 200) & (ids < 800)).all()


@pytest.mark.parametrize("value", ["gaussian", "", None, "UNIFORM"])
def test_invalid_names_raise(value):
    with pytest.raises(ValueError, match="noise_schedule_distribution"):
        validate_noise_schedule_distribution(value)
    with pytest.raises(ValueError, match="noise_schedule_distribution"):
        TinyArchitecture().set_training_runtime(noise_schedule_distribution=value)


@pytest.mark.parametrize("low,high", [(-1, 1000), (0, 1001), (50, 50), (80, 20)])
def test_invalid_windows_raise(low, high):
    with pytest.raises(ValueError, match="window"):
        sample_timestep_ids(1000, 4, low=low, high=high)


def _legacy_ids(num_timesteps, batch_size, *, distribution="uniform", low=0, high=None):
    return torch.randint(low, num_timesteps if high is None else high, (batch_size,))


def _tiny_loss_case(video_mode="x", action_mode="x"):
    arch = TinyArchitecture({"video_prediction_type": video_mode, "action_prediction_type": action_mode})
    arch.video_backbone.scheduler.set_timesteps(1000, shift=42, training=True)
    arch.action_backbone.scheduler.set_timesteps(1000, shift=5, training=True)
    video = torch.linspace(-1, 1, 192).reshape(8, 1, 3, 2, 4)
    actions = torch.linspace(-0.5, 0.5, 48).reshape(8, 2, 3)
    return arch, dict(input_latents=video, actions=actions, clean_condition_latents=video[:, :, :1])


@pytest.mark.parametrize("action_enabled", [False, True])
def test_base_default_loss_and_rng_equal_legacy(action_enabled):
    arch, inputs = _tiny_loss_case()
    state = torch.get_rng_state()
    with patch("openwam.model.architectures.base.sample_timestep_ids", side_effect=_legacy_ids):
        expected = arch.compute_loss(**inputs, lambda_action=int(action_enabled)) if action_enabled else None
    # TinyArchitecture's forward assumes actions exist; use zero weight with a safe forward.
    if not action_enabled:
        arch.forward = lambda actions, t, *, latents, **kw: (arch.weight[0].expand_as(latents), None)
        torch.set_rng_state(state)
        with patch("openwam.model.architectures.base.sample_timestep_ids", side_effect=_legacy_ids):
            expected = arch.compute_loss(**inputs, lambda_action=0)
    expected_state = torch.get_rng_state()
    torch.set_rng_state(state)
    actual = arch.compute_loss(**inputs, lambda_action=int(action_enabled))
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    assert torch.equal(torch.get_rng_state(), expected_state)


@pytest.mark.parametrize("distribution", NOISE_SCHEDULE_DISTRIBUTIONS)
@pytest.mark.parametrize("video_mode,action_mode", [("x", "x"), ("x", "v")])
def test_real_loss_uses_independent_ids_and_existing_shift_weights(distribution, video_mode, action_mode):
    arch, inputs = _tiny_loss_case(video_mode, action_mode)
    arch.set_training_runtime(noise_schedule_distribution=distribution)
    seen = []

    def capture(n, b, **kw):
        assert kw["distribution"] == distribution
        ids = sample_timestep_ids(n, b, **kw)
        seen.append(ids.clone())
        return ids

    with patch("openwam.model.architectures.base.sample_timestep_ids", side_effect=capture):
        result = arch.compute_loss(**inputs)
    assert len(seen) == 2 and not torch.equal(*seen)
    # Check the actual scheduler lookup against an independent float64 shift oracle.
    for scheduler, ids, shift in zip((arch.video_scheduler, arch.action_scheduler), seen, (42, 5)):
        base = 1 - ids.double() / 1000
        expected_sigma = shift * base / (1 + (shift - 1) * base)
        torch.testing.assert_close(scheduler.sigmas[ids].double(), expected_sigma, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(scheduler.linear_timesteps_weights[ids]).all()
    assert all(torch.isfinite(x).all() for x in result.values())
    result["loss"].backward()
    assert torch.isfinite(arch.weight.grad).all()


@pytest.mark.parametrize("distribution", NOISE_SCHEDULE_DISTRIBUTIONS)
@pytest.mark.parametrize("cond_prob", [0.0, 1.0])
def test_idm_all_noisy_branches_and_clean_condition(distribution, cond_prob):
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.video_cond_noise_prob = cond_prob
    arch.init_training_schedulers(1000)
    arch.set_training_runtime(noise_schedule_distribution=distribution)
    inputs = dict(input_latents=torch.ones(4, 1, 1, 1, 1), context=torch.ones(4, 2, arch.action_backbone.text_dim),
                  context_mask=torch.ones(4, 2, dtype=torch.bool), actions=torch.ones(4, 3, 3))
    with patch("openwam.model.architectures.dual_system.idm.sample_timestep_ids", wraps=sample_timestep_ids) as sampler:
        state = torch.get_rng_state()
        result = arch.compute_loss(**inputs)
        after = torch.get_rng_state()
    assert sampler.call_count == (3 if cond_prob else 2)
    assert all(c.kwargs["distribution"] == distribution for c in sampler.call_args_list)
    assert torch.isfinite(result["loss"])
    if not cond_prob:
        assert torch.count_nonzero(vb.prepare_timesteps[1]) == 0
    if distribution == "uniform":
        torch.set_rng_state(state)
        with patch("openwam.model.architectures.dual_system.idm.sample_timestep_ids", side_effect=_legacy_ids):
            legacy = arch.compute_loss(**inputs)
        assert torch.equal(after, torch.get_rng_state())
        assert torch.equal(result["loss"], legacy["loss"])


@pytest.mark.parametrize("distribution", NOISE_SCHEDULE_DISTRIBUTIONS)
def test_hydra_config_roundtrip(tmp_path, distribution):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"training.noise_schedule_distribution={distribution}"])
    save_config(str(tmp_path), cfg)
    assert OmegaConf.load(tmp_path / "config.yaml").training.noise_schedule_distribution == distribution


@pytest.mark.parametrize("kind,live,saved,raises", [
    ("fresh", None, None, False), ("fresh", "mode", None, False),
    ("resume", None, None, False), ("resume", "uniform", None, False),
    ("resume", "mode", "mode", False), ("resume", "mode", None, True),
    ("resume", None, "logit_normal", True), ("finetune", "logit_logistic", "mode", False),
])
def test_trainer_runtime_and_resume_compatibility(monkeypatch, kind, live, saved, raises):
    import openwam.model as model
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils import ckpt_model_loader as loader

    arch = TinyArchitecture()
    arch.init_training_schedulers = lambda n: None
    arch.set_dtype_device = lambda *a: None
    arch.freeze_modules = lambda names: []
    resolved = SimpleNamespace(registry_name="test", params={}, canonical=SimpleNamespace(framework="dual_system", variant="joint_self_attn"))
    cfg = OmegaConf.create({"project": {"seed": None}, "model": {"freeze": []}, "training": {
        "lambda_video": 1, "lambda_action": 1, "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False, "max_timestep_boundary": 1, "min_timestep_boundary": 0,
    }})
    if live is not None:
        cfg.training.noise_schedule_distribution = live
    if kind != "fresh":
        cfg.training[f"{kind}_ckpt_path"] = "/unused/test-checkpoint"
    ckpt = OmegaConf.create({"training": {}})
    if saved is not None:
        ckpt.training.noise_schedule_distribution = saved
    monkeypatch.setattr(model, "resolve_architecture_config", lambda m: resolved)
    monkeypatch.setattr(model, "build_architecture", lambda *a: arch)
    monkeypatch.setattr(loader, "build_architecture_from_ckpt_dir", lambda *a, **kw: (resolved, arch, ckpt))
    monkeypatch.setattr(loader, "warn_live_model_cfg_ignored", lambda *a: None)
    monkeypatch.setattr(loader, "propagate_component_specs", lambda *a: None)
    monkeypatch.setattr("openwam.train.openwam_trainer.log_parameter_counts", lambda *a, **kw: None)
    if raises:
        with pytest.raises(ValueError, match="resume noise_schedule_distribution mismatch"):
            OpenWAMTrainer(cfg)
    else:
        trainer = OpenWAMTrainer(cfg)
        assert trainer.architecture._noise_schedule_distribution == (live or "uniform")
