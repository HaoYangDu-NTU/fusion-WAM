"""CPU checks for native geometry, learned IJEPA preprocessing and resume."""
import copy
import json
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from openwam.experiments import resolve_experiment, validate_resume
from openwam.train.utils.epoch_sampler import EpochRandomSampler


def test_baseline_resolved_encoder_produces_declared_geometry():
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
    from tests.test_vjepa_svae import _MockVJEPAViT
    cfg, profile, _ = resolve_experiment("E00")
    enc = cfg.model.video_backbone.encoder
    model = VJEPA21VideoEncoder(
        _MockVJEPAViT(embed_dim=8), embed_dim=8, variant="test",
        use_avg_t_pooling=enc.use_avg_t_pooling,
        history_num_tubelets=enc.history_num_tubelets,
        seperate_encode_view=enc.seperate_encode_view,
        use_wan_style_patch_grid=enc.use_wan_style_patch_grid,
    )
    with torch.no_grad():
        latent = model.batch_encode(torch.zeros(1, 3, 10, 224, 448))
    assert latent.shape[2] == profile["geometry"]["clean_latents"] + profile["geometry"]["future_latents"]
    assert latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2) == profile["geometry"]["tokens"]


def test_resume_identity_ignores_launch_timestamp_but_locks_learning_rate(tmp_path, monkeypatch):
    from openwam import experiments as exp
    monkeypatch.setattr(exp, "source_fingerprint", lambda: "unchanged-source")
    first, profile, _ = resolve_experiment("E02")
    second, _, _ = resolve_experiment("E02", resume=str(tmp_path))
    first.project.output_dir = "outputs/first-launch"
    second.project.output_dir = "outputs/resumed-launch"
    original = exp.run_identity(first, profile, 4, {})
    (tmp_path / "experiment_identity.json").write_text(json.dumps(original))
    validate_resume(tmp_path, exp.run_identity(second, profile, 4, {}))
    second.training.learning_rate *= 2
    with pytest.raises(ValueError, match="identity changed: config"):
        validate_resume(tmp_path, exp.run_identity(second, profile, 4, {}))


def test_formal_batch_override_is_rejected():
    with pytest.raises(ValueError, match="protocol batch"):
        resolve_experiment("E02", batch_size=2)
    assert resolve_experiment("E02", batch_size=2, smoke_steps=35)[0].training.batch_size == 2


def test_protocol_training_budget_and_policy_width_are_applied(monkeypatch):
    from openwam import experiments as exp
    load = exp.load_yaml
    def modified(path):
        value = load(path)
        if path.name == "protocol.yaml":
            value["training"]["max_steps"] = 1234
            value["policy"]["video_dit"]["dim"] = 768
        return value
    monkeypatch.setattr(exp, "load_yaml", modified)
    cfg, _, _ = resolve_experiment("E02")
    assert cfg.training.max_steps == 1234
    assert cfg.model.video_backbone.dit_config.dim == 768


@pytest.mark.parametrize("experiment,clean,future,tokens", [
    ("E00", 1, 4, 490), ("E01", 2, 8, 980), ("E02", 1, 4, 490),
])
def test_profiles_preserve_native_geometry(experiment, clean, future, tokens):
    cfg, profile, _ = resolve_experiment(experiment)
    assert profile["geometry"] == {
        **profile["geometry"], "clean_latents": clean, "future_latents": future, "tokens": tokens,
    }
    assert cfg.training.batch_size == 64 and cfg.training.save_full_states_for_resume
    assert cfg.dataloader.history_num_tubelets == 1
    assert cfg.model.video_backbone.text_encoder_type == "rynnbrain1p1-9b"
    if experiment == "E01":
        assert not cfg.model.video_backbone.encoder.use_avg_t_pooling
        assert "history_fusion_mode" not in cfg.model.video_backbone.encoder
    if experiment == "E02":
        assert cfg.model.freeze[-1] == "video_backbone.video_encoder._m"


def test_resume_refuses_different_model_or_training_budget(tmp_path):
    identity = {"experiment": "E01", "world_size": 4, "batch_size": 64}
    (tmp_path / "experiment_identity.json").write_text(json.dumps(identity))
    validate_resume(tmp_path, identity)
    with pytest.raises(ValueError, match="identity changed"):
        validate_resume(tmp_path, {**identity, "world_size": 2})


def test_interrupted_fullstate_overwrite_is_not_discoverable(tmp_path):
    from types import SimpleNamespace
    from openwam.train.utils.checkpointing import save_full_state, find_latest_accel_state

    state = tmp_path / "accel_state_step_3"
    state.mkdir()
    (state / "trainer_state.json").write_text('{"global_step":3}')
    def interrupted(directory):
        assert not (state / "trainer_state.json").exists()
        raise OSError("simulated interrupted write")
    accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None,
                                  save_state=interrupted)
    with pytest.raises(OSError, match="interrupted"):
        save_full_state(accelerator, str(tmp_path), 3, 3, 0)
    assert find_latest_accel_state(str(tmp_path)) is None


@pytest.mark.parametrize("rank", range(4))
def test_epoch_sampler_four_rank_shards_resume_at_same_samples(rank):
    from accelerate.data_loader import prepare_data_loader, skip_first_batches

    dataset = list(range(103))
    def loader():
        dl = torch.utils.data.DataLoader(dataset, batch_size=4,
                                        sampler=EpochRandomSampler(dataset, 42))
        return prepare_data_loader(dl, num_processes=4, process_index=rank,
                                   put_on_device=False, rng_types=[])
    full = loader()
    full.set_epoch(3)
    expected = [batch.tolist() for batch in full]
    resumed = loader()
    resumed.set_epoch(3)
    actual = [batch.tolist() for batch in skip_first_batches(resumed, 2)]
    assert actual == expected[2:]
    other = loader()
    other.set_epoch(4)
    assert [batch.tolist() for batch in other] != expected


def test_accelerate_fullstate_restores_optimizer_scheduler_and_rng(tmp_path):
    from accelerate import Accelerator
    from openwam.train.utils.checkpointing import save_full_state, load_full_state

    accelerator = Accelerator(cpu=True)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    def step():
        optimizer.zero_grad()
        loss = model(torch.randn(5, 3)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

    for _ in range(3):
        step()
    save_full_state(accelerator, str(tmp_path), 3, 3, 0)
    for _ in range(3):
        step()
    expected = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_lr = scheduler.get_last_lr()
    expected_rng = (torch.rand(3), random.random(), np.random.rand())
    meta = load_full_state(accelerator, str(tmp_path / "accel_state_step_3"))
    assert meta["global_step"] == 3
    for _ in range(3):
        step()
    for key, tensor in expected.items():
        torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
    for key, state in expected_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)
    assert scheduler.get_last_lr() == expected_lr
    actual_rng = (torch.rand(3), random.random(), np.random.rand())
    torch.testing.assert_close(actual_rng[0], expected_rng[0], rtol=0, atol=0)
    assert actual_rng[1:] == expected_rng[1:]


def test_ijepa_native_adapter_has_grad_after_frozen_preparation(tmp_path, monkeypatch):
    from openwam.model import build_architecture, resolve_architecture_config
    from openwam.model.video_backbone.encoder.ijepa import IJEPAImageVideoEncoder
    from tests.test_ijepa_encoder import _MockIJEPA
    from tests.test_exp_dinov3_native import frames
    from tests.test_wan_text_embedding_cache import make_cache

    cfg, _, _ = resolve_experiment("E02")
    vb = cfg.model.video_backbone
    vb.text_encoder_type = "umt5-xxl"
    vb.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        vb.dit_config[key] = value
    cfg.model.action_backbone.dim = 32
    cfg.model.action_backbone.ffn_dim = 64
    cfg.dataloader.normalize_mode = None
    class SpatialIJEPA(_MockIJEPA):
        def forward(self, x):
            tokens = super().forward(x)
            pixels = x[:, 0, ::self.patch, ::self.patch].flatten(1).unsqueeze(-1)
            return tokens + pixels * torch.arange(self.embed_dim, device=x.device, dtype=x.dtype)

    monkeypatch.setattr(
        IJEPAImageVideoEncoder, "from_pretrained",
        classmethod(lambda cls, path, **kw: cls(SpatialIJEPA(), embed_dim=8, **{
            k: v for k, v in kw.items() if k not in ("checkpoint", "model_name", "img_size", "patch_size")
        })),
    )
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.freeze_modules(list(cfg.model.freeze))
    model.init_training_schedulers()
    encoder = model.video_backbone.video_encoder
    # Nonzero scale exercises projection gradients too; native initialization remains zero.
    with torch.no_grad():
        encoder.delta_scale.fill_(0.2)
    clip = frames()
    sample = dict(video=clip[1:], clean_condition_images=clip[:2], prompt="open drawer",
                  action=torch.zeros(32, 80), proprio=torch.zeros(80),
                  video_mask=torch.ones(9, dtype=torch.bool))
    inputs = model.prepare_inputs([sample])
    assert inputs["input_latents"] is None
    assert not inputs["ijepa_frame_features"].requires_grad
    assert inputs["num_clean_condition_latents"] == 1
    assert inputs["video_is_pad"].shape == (1, 4)
    losses = model.compute_loss(**inputs)
    losses["loss"].backward()
    for parameter in (encoder.delta_scale, encoder._delta_proj.weight, encoder.feature_norm.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder._m.parameters())
    # Rebuild strictly from checkpoint-local assets, with the original weights unavailable.
    from safetensors.torch import save_file
    from openwam.train.utils.checkpointing import save_config
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
    from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint
    from types import SimpleNamespace

    monkeypatch.setattr("openwam.model.video_backbone.encoder.ijepa._load_official_vit",
                        lambda *a, **kw: SpatialIJEPA())
    run = tmp_path / "run"
    model.video_backbone.save_deploy_assets(str(run), cfg)
    cfg.model.video_backbone.encoder.model_path = "/unavailable"
    save_config(str(run), cfg)
    save_file(model.state_dict(), str(run / "checkpoint_step_1.safetensors"))
    np.save(run / "normalization_stats.npy", {})
    _, deployed = load_from_checkpoint_dir(str(run), "cpu", "checkpoint_step_1.safetensors")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(deployed.state_dict()[name], tensor, rtol=0, atol=0)
    native = SimpleNamespace(
        architecture=deployed, _cfg_scale=1., _decode_video=False, _dit_cache=None, _prompt_embed_cache={},
        cfg=OmegaConf.create({"inference": {"denoise_mode": "sync", "denoise_steps": 2,
                             "num_frames": 33, "video_num_frames": 9, "height": 224, "width": 448}}),
    )
    observations = [dict(prompt="open drawer", image=clip[1], history_image=clip[0],
                         state=np.zeros(10, np.float32))]
    engine = BatchedOpenWAMEngine(native)
    batch, _, _ = engine._prepare(observations)
    ordinary = deployed.video_backbone.preprocess_input_for_inference(
        prompt="open drawer", clean_condition_images=clip[:2], num_frames=9,
        height=224, width=448, seed=42, num_inference_steps=2,
    )
    assert batch["latents"].shape == ordinary["latents"].shape == (1, 8, 5, 14, 28)
    torch.testing.assert_close(batch["clean_condition_latents"], ordinary["clean_condition_latents"])
    plus = prepare_checkpoint(run, "checkpoint_step_1.safetensors", tmp_path / "plus",
                              cfg.model.video_backbone.text_embedding_dir)
    _, plus_model = load_from_checkpoint_dir(str(plus), "cpu", "checkpoint_step_1.safetensors")
    assert plus_model.video_backbone.num_condition_latents == 1
