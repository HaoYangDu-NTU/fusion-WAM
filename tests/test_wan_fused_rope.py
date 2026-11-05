import pytest
import torch

from openwam.model.video_backbone.wan.models.dit import rope_apply
from openwam.model.video_backbone.wan.fused_rope import rope_apply_fused


def reference(x, phase, heads):
    z = x.to(torch.float64).reshape(x.shape[0], x.shape[1], heads, -1, 2)
    return torch.view_as_real(torch.view_as_complex(z) * phase).flatten(2).to(x.dtype)


def test_cpu_keeps_original_path():
    x = torch.randn(2, 7, 32, requires_grad=True)
    phase = torch.polar(torch.ones(7, 1, 8, dtype=torch.float64), torch.randn(7, 1, 8, dtype=torch.float64))
    assert rope_apply_fused(x, phase, 2) is None
    torch.testing.assert_close(rope_apply(x, phase, 2), reference(x, phase, 2), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('tokens,heads,dim', [(7, 2, 16), (480, 12, 128), (2400, 12, 128)])
def test_fused_fp64_rotation_output_and_gradients(dtype, tokens, heads, dim):
    torch.manual_seed(29)
    x = torch.randn(2, tokens, heads * dim, device='cuda', dtype=dtype, requires_grad=True)
    phase = torch.polar(torch.ones(tokens, 1, dim // 2, device='cuda', dtype=torch.float64),
                        torch.randn(tokens, 1, dim // 2, device='cuda', dtype=torch.float64))
    out = rope_apply(x, phase, heads)
    target = reference(x, phase, heads)
    torch.testing.assert_close(out, target, atol=0, rtol=0)
    weight = torch.randn_like(out)
    grad = torch.autograd.grad(out, x, weight, retain_graph=True)[0]
    expected = torch.autograd.grad(target, x, weight)[0]
    torch.testing.assert_close(grad, expected, atol=0, rtol=0)
    # Frequency gradients must keep the original differentiable implementation.
    phase.requires_grad_(True)
    assert rope_apply_fused(x, phase, heads) is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_fused_rope_higher_order_gradient():
    x = torch.randn(1, 7, 32, device='cuda', requires_grad=True)
    phase = torch.polar(torch.ones(7, 1, 8, device='cuda', dtype=torch.float64),
                        torch.randn(7, 1, 8, device='cuda', dtype=torch.float64))
    def second(fn):
        y = fn(x, phase, 2).square().sum()
        first = torch.autograd.grad(y, x, create_graph=True)[0]
        return torch.autograd.grad(first.sum(), x)[0]
    torch.testing.assert_close(second(rope_apply), second(reference), atol=1e-6, rtol=1e-6)
