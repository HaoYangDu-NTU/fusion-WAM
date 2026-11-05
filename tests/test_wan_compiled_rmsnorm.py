import pytest
import torch
from torch.utils.checkpoint import checkpoint
from openwam.model.video_backbone.wan.models import dit


def test_cpu_norm_keeps_eager_path(monkeypatch):
    monkeypatch.setattr(
        dit, "_compiled_rms_norm", lambda *args: pytest.fail("CPU must stay eager")
    )
    m = dit.RMSNorm(128)
    x = torch.randn(2, 31, 128, requires_grad=True)
    torch.testing.assert_close(m(x), m._forward(x), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 1536])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_compiled_norm_matches_output_input_and_weight_gradients(
    dtype, width, use_checkpoint, monkeypatch
):
    torch.manual_seed(49)
    m = dit.RMSNorm(width, eps=1e-6).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        m.weight.normal_(1.0, 0.2)
    x = torch.randn(2, 513, width, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn_like(x)
    compiled = dit._compiled_rms_norm
    seen = []

    def record(*args):
        seen.append(True)
        return compiled(*args)

    monkeypatch.setattr(dit, "_compiled_rms_norm", record)
    actual = checkpoint(m, x, use_reentrant=False) if use_checkpoint else m(x)
    grads = torch.autograd.grad(actual, (x, m.weight), weight)
    assert seen
    expected = m._forward(x)
    refs = torch.autograd.grad(expected, (x, m.weight), weight)
    for a, b in zip((actual, *grads), (expected, *refs)):
        relative = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-10)
        assert relative < (2e-6 if dtype == torch.float32 else 0.005)
    m.eval()
    seen.clear()
    torch.testing.assert_close(m(x), m._forward(x), atol=0, rtol=0)
    assert not seen
