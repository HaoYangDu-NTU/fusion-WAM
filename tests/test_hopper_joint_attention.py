from types import SimpleNamespace
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from openwam.model.architectures.dual_system import mot_driver as mot


def flags():
    return tuple(
        f()
        for f in (
            torch.backends.cuda.cudnn_sdp_enabled,
            torch.backends.cuda.flash_sdp_enabled,
            torch.backends.cuda.mem_efficient_sdp_enabled,
            torch.backends.cuda.math_sdp_enabled,
        )
    )


def test_priority_restored_and_disabled_backends_respected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    q = SimpleNamespace(
        is_cuda=True, requires_grad=True, dtype=torch.bfloat16, device="cuda"
    )
    order = torch._C._get_sdp_priority_order()
    enabled = flags()
    with mot._training_attention_context(q):
        assert torch._C._get_sdp_priority_order()[0] == int(SDPBackend.CUDNN_ATTENTION)
        assert flags() == enabled
    assert torch._C._get_sdp_priority_order() == order
    assert flags() == enabled
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], set_priority=True):
        forced = torch._C._get_sdp_priority_order()
        forced_flags = flags()
        with mot._training_attention_context(q):
            assert torch._C._get_sdp_priority_order() == forced
            assert flags() == forced_flags
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with mot._training_attention_context(q):
        assert torch._C._get_sdp_priority_order() == order


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("first,length", [(16, 64), (64, 64), (98, 522)])
@pytest.mark.parametrize("dense", [False, True])
def test_hopper_joint_attention_matches_float64_math_output_and_gradients(first, length, dense):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper-specific dispatch")
    torch.manual_seed(48)
    driver = object.__new__(mot.DualSystemMoTDriver)
    driver.num_heads = 2
    xs = [
        torch.randn(2, length, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    weight = torch.randn_like(xs[0])
    if dense:
        mask = torch.ones(length, length, device="cuda", dtype=torch.bool)
        mask[:first, first:] = False
        mask = mot._CleanConditionAttention(first, mask)
    else:
        mask = mot._CleanConditionAttention(first)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        actual = driver._mixed_attention(*xs, mask)
        grads = torch.autograd.grad(actual, xs, weight)
    assert any("cudnn_attention" in e.key for e in prof.key_averages())
    ys = [x.detach().double().requires_grad_(True) for x in xs]
    with sdpa_kernel(SDPBackend.MATH):
        expected = driver._mixed_attention(*ys, mask)
        refs = torch.autograd.grad(expected, ys, weight.double())
    for a, b in zip((actual, *grads), (expected, *refs)):
        relative = (a.double() - b).norm() / b.norm().clamp_min(1e-10)
        assert relative < 0.006


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dense_hopper_mask_preserves_fully_masked_rows():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper-specific dispatch")
    torch.manual_seed(19)
    driver = object.__new__(mot.DualSystemMoTDriver)
    driver.num_heads = 2
    xs = [torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    mask = torch.ones(2, 1, 64, 64, device="cuda", dtype=torch.bool)
    mask[0, :, :16] = False
    mask[1] = False
    actual = driver._mixed_attention(*xs, mask)
    assert torch.isfinite(actual).all()
    assert not actual[0, :16].any()
    assert not actual[1].any()
    grads = torch.autograd.grad(actual, xs, torch.randn_like(actual))
    assert all(torch.isfinite(g).all() for g in grads)
    assert not grads[0][0, :16].any()
    assert all(not g[1].any() for g in grads)
