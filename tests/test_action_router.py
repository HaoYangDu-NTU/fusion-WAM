"""E00+ math, gradient ownership, real joint-loop integration and resume."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.experiments import resolve_experiment
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.utils.action_router import ActionVisualRouter
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.training_utils import reduce_step_metrics
from tests.test_repa import tiny_model


def make_model(tmp_path, *, enabled=True, repa=False):
    torch.manual_seed(31)
    fixture_dir = tmp_path / f"fixture_{len(list(tmp_path.glob('fixture_*')))}"
    fixture_dir.mkdir()
    base = tiny_model(fixture_dir, enabled=repa, views=True)
    cfg = dict(base.cfg, video_attention_mask_mode="clean_condition_causal", action_router=dict(
        enabled=enabled, dim=16, num_heads=2, top_k=2,
        gate_init=0.01, sparse_weight=0.01, warmup_steps=1, ramp_steps=2))
    with patch.object(BaseWAMArchitecture, "_init_video_backbone",
                      lambda self, _: setattr(self, "video_backbone", base.video_backbone)):
        model = DualSystemSelfAttnArchitecture(cfg)
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    model.init_training_schedulers()
    model.freeze_modules(["video_backbone.video_encoder"])
    return model


def model_inputs(model):
    rng = np.random.default_rng(31)
    frames = [Image.fromarray(rng.integers(0, 256, (224, 448, 3), dtype=np.uint8)) for _ in range(5)]
    inputs = model.preprocess(frames=[frames], text=["open drawer"], ref_images=[[frames[0], frames[1]]])
    inputs["actions"] = torch.randn(1, 4, 7)
    # Last query entirely padded; the last two dimensions are always padding.
    inputs["action_is_pad"] = torch.zeros(1, 4, 7, dtype=torch.bool)
    inputs["action_is_pad"][:, -1] = True
    inputs["action_is_pad"][..., -2:] = True
    return {k: v.to(model.dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in inputs.items()}


def test_mass_math_padding_and_gradient_ownership():
    torch.manual_seed(5)
    router = ActionVisualRouter(6, 8, dim=8, num_heads=2, top_k=2)
    actions = torch.randn(2, 3, 6, requires_grad=True)
    clean = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.tensor([[True, False, False], [False, True, False]])
    result, aux = router(actions, clean, query_valid=valid, return_aux=True)
    q = router.q(router.action_norm(actions)).reshape(2, 3, 2, 4).transpose(1, 2)
    k = router.k(router.visual_norm(clean)).reshape(2, 5, 2, 4).transpose(1, 2)
    weights = (q @ k.transpose(-1, -2) / 2).softmax(-1)
    losses = 1 - weights.topk(2, -1).values.sum(-1)
    expected = (losses * valid[:, None]).sum() / 4
    torch.testing.assert_close(aux["router_sparse_raw"], expected)
    assert router.gate_logit.sigmoid().item() == pytest.approx(0.01)
    aux["router_sparse_raw"].backward()
    assert actions.grad.abs().sum() > 0 and clean.grad.abs().sum() > 0
    assert torch.count_nonzero(actions.grad[~valid]) == 0
    assert router.q.weight.grad.abs().sum() > 0
    assert router.k.weight.grad.abs().sum() > 0
    assert router.v.weight.grad is None and router.out.weight.grad is None
    assert router.gate_logit.grad is None
    router.zero_grad(set_to_none=True)
    result, _ = router(actions, clean)
    result.square().mean().backward()
    assert router.v.weight.grad.abs().sum() > 0
    assert router.out.weight.grad.abs().sum() > 0
    assert router.gate_logit.grad.abs() > 0
    _, empty = router(actions, clean, query_valid=torch.zeros_like(valid), return_aux=True)
    assert empty["router_sparse_raw"].item() == 0
    assert all(torch.isfinite(value) for value in empty.values())


def test_schedule_and_uniform_small_context():
    router = ActionVisualRouter(4, 4, dim=4, num_heads=1, top_k=2,
                                warmup_steps=10, ramp_steps=20)
    assert [router.coefficient(s) for s in (0, 10, 20, 30, 100)] == [0, 0, .005, .01, .01]
    with torch.no_grad():
        router.q.weight.zero_()
        router.q.bias.zero_()
    _, aux = router(torch.randn(1, 2, 4), torch.randn(1, 8, 4), return_aux=True)
    assert aux["router_sparse_raw"].item() == pytest.approx(.75)
    _, aux = router(torch.randn(1, 2, 4), torch.randn(1, 1, 4), return_aux=True)
    assert aux["router_sparse_raw"].item() == 0
    with pytest.raises(ValueError, match="nonempty"):
        router(torch.randn(1, 2, 4), torch.randn(1, 0, 4))


@pytest.mark.parametrize("repa", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_loss_inference_and_checkpoint(tmp_path, repa, dtype):
    model = make_model(tmp_path, repa=repa)
    inputs = model_inputs(model)
    model.to(dtype=dtype)
    model.set_dtype_device(dtype, torch.device("cpu"))
    inputs = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
              for k, v in inputs.items()}
    torch.manual_seed(44)
    off = model.compute_loss(**inputs, optimizer_step=0)
    torch.manual_seed(44)
    on = model.compute_loss(**inputs, optimizer_step=3, use_gradient_checkpointing=True)
    torch.testing.assert_close(on["loss"], off["loss"] + on["loss_router_sparse"])
    assert on["loss_router_sparse"] > 0
    on["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.action_router.parameters())
    assert model.video_backbone.dit.blocks[0].self_attn.q.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.video_backbone.video_encoder.parameters())
    params = build_trainable_parameters(SimpleNamespace(architecture=model))
    assert any(p is model.action_router.q.weight for p in params)
    model.eval()
    forward = {k: v for k, v in inputs.items() if k not in ("actions", "action_is_pad", "repa_target_latents")}
    forward["latents"] = forward.pop("input_latents")
    forward["timestep"] = torch.tensor([500.], dtype=dtype)
    with torch.no_grad():
        expected = model(inputs["actions"], forward["timestep"], **forward)
        disabled = model(inputs["actions"], forward["timestep"], use_action_router=False, **forward)
    assert len(expected) == 2 and not torch.equal(expected[1], disabled[1])
    model.save_checkpoint(str(tmp_path / "router.safetensors"))
    restored = make_model(tmp_path, repa=repa).to(dtype)
    restored.set_dtype_device(dtype, torch.device("cpu"))
    restored.load_checkpoint(str(tmp_path / "router.safetensors"), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(inputs["actions"], forward["timestep"], **forward)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_baseline_initialization_and_rng_unchanged(tmp_path):
    baseline = make_model(tmp_path, enabled=False)
    rng = torch.get_rng_state()
    routed = make_model(tmp_path)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, routed.state_dict()[key], rtol=0, atol=0)
    assert not any(k.startswith("action_router.") for k in baseline.state_dict())


def test_router_clean_keys_exclude_future_and_device_dispatch(tmp_path):
    model = make_model(tmp_path)
    inputs = model_inputs(model)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    assert model.action_router.q.weight.dtype == torch.bfloat16
    forward = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
               for k, v in inputs.items() if k not in ("actions", "action_is_pad")}
    latents = forward.pop("input_latents")
    forward["timestep"] = torch.tensor([500.], dtype=torch.bfloat16)
    captured = []
    hook = model.action_router.register_forward_pre_hook(lambda module, args: captured.append(args[1].detach().clone()))
    model.eval()
    with torch.no_grad():
        actions = inputs["actions"].to(torch.bfloat16)
        model(actions, forward["timestep"], latents=latents, **forward)
        changed = latents.clone()
        changed[:, :, 1:] = torch.randn_like(changed[:, :, 1:])
        model(actions, forward["timestep"], latents=changed, **forward)
    hook.remove()
    assert captured[0].shape[1] == 98
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)


def test_profiles_and_metrics():
    base, _, _ = resolve_experiment("E00")
    sparse, profile, _ = resolve_experiment("E00+")
    dense, _, _ = resolve_experiment("E00+-Dense")
    assert profile["geometry"]["tokens"] == 490
    assert not base.model.architecture.action_router.enabled
    assert sparse.model.architecture.action_router.enabled
    assert sparse.model.architecture.action_router.sparse_weight == .01
    assert dense.model.architecture.action_router.sparse_weight == 0
    assert sparse.training.batch_size == base.training.batch_size == 64
    metrics = reduce_step_metrics(None, dict(total=torch.tensor(3.), video=torch.tensor(1.),
        action=torch.tensor(2.), router_sparse=torch.tensor(.001), router_topk_mass=torch.tensor(.9)), torch.tensor(0.))
    assert metrics["loss_router_sparse"] == pytest.approx(.001)
    assert metrics["router_topk_mass"] == pytest.approx(.9)


def test_resume_next_update_matches(tmp_path):
    model = make_model(tmp_path)
    inputs = model_inputs(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def step(m, opt, number):
        torch.manual_seed(number)
        opt.zero_grad(set_to_none=True)
        losses = m.compute_loss(**inputs, optimizer_step=number)
        losses["loss"].backward()
        opt.step()
        return losses["loss"].detach()

    step(model, optimizer, 1)
    checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), opt_step=2))
    expected = step(model, optimizer, 2)
    restored = make_model(tmp_path)
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-4)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    actual = step(restored, restored_optimizer, checkpoint["opt_step"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("gain", [0.0, 1.0, 2.0])
def test_residual_gain_scales_output_and_router_gradient(gain):
    torch.manual_seed(17)
    options = dict(dim=8, num_heads=2, gate_init=0.1, fp32_gate=True, fp32_residual=True)
    native = ActionVisualRouter(6, 8, **options)
    scaled = ActionVisualRouter(6, 8, residual_gain=gain, **options)
    scaled.load_state_dict(native.state_dict(), strict=True)
    actions, clean = torch.randn(2, 3, 6), torch.randn(2, 5, 8)
    reference, _ = native(actions, clean)
    result, _ = scaled(actions, clean)
    torch.testing.assert_close(result, actions + gain * (reference - actions))
    result.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(scaled.gate_logit.grad, gain * native.gate_logit.grad)
