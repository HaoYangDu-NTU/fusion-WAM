"""Fused frozen Q/K must retain all axes, prefixes, layout and dtype rounding."""
import pytest
import torch
from openwam.model.video_backbone.encoder.vjepa21_src import fused_rope, modules


def reference(q, k, positions, axis, registers, cls):
    result = []
    for x in (q, k):
        parts = [modules.rotate_queries_or_keys(x[..., i * axis:(i + 1) * axis], p, registers, cls).to(x.dtype)
                 for i, p in enumerate(positions)]
        if 3 * axis < x.shape[-1]:
            parts.append(x[..., 3 * axis:])
        result.append(torch.cat(parts, dim=-1))
    return result


def test_cpu_uses_original_path():
    x = torch.randn(2, 3, 7, 64)
    positions = [torch.arange(7).float()] * 3
    assert fused_rope.rotate_qk_frozen(x, x, positions, 20, 0, False) is None


gpu = pytest.mark.skipif(not torch.cuda.is_available() or fused_rope.triton is None, reason="CUDA/Triton required")


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,dim,registers,cls", [(196, 64, 0, False), (980, 64, 8, True), (2400, 128, 4, False)])
def test_qk_matches_original_exactly(dtype, length, dim, registers, cls):
    torch.manual_seed(75)
    qkv = torch.randn(2, length + registers + int(cls), 3, 3, dim, device="cuda", dtype=dtype).permute(2, 0, 3, 1, 4)
    ids = torch.arange(length, device="cuda")
    positions = [(ids // 196).float(), ((ids // 14) % 14).float() * 15 / 13, (ids % 14).float() * 15 / 13]
    args = (qkv[0], qkv[1], positions, 2 * ((dim // 3) // 2), registers, cls)
    actual, expected = fused_rope.rotate_qk_frozen(*args), reference(*args)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert a.stride() == b.stride()


@gpu
def test_gradients_and_sample_positions_keep_original_path():
    x = torch.randn(2, 3, 7, 64, device="cuda", requires_grad=True)
    p = torch.arange(7, device="cuda").float()
    assert fused_rope.rotate_qk_frozen(x, x, [p] * 3, 20, 0, False) is None
    assert fused_rope.rotate_qk_frozen(x.detach(), x.detach(), [p.expand(2, 3, -1)] * 3, 20, 0, False) is None


@gpu
@pytest.mark.parametrize("frames", [1, 5])
def test_full_frozen_attention_output_matches_original(frames, monkeypatch):
    from openwam.model.video_backbone.encoder.vjepa21_src.loader import prepare_vjepa_imports_and_patch
    prepare_vjepa_imports_and_patch()
    torch.manual_seed(44)
    attention = modules.RoPEAttention(128, num_heads=2, use_sdpa=True, interpolate_rope=True,
                                     n_registers=4, has_cls_first=True).cuda().bfloat16().eval().requires_grad_(False)
    x = torch.randn(2, frames * 196 + 5, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        actual, _ = attention(x, T=frames, H_patches=14, W_patches=14)
        monkeypatch.setattr(modules, "rotate_qk_frozen", lambda *args: None)
        expected, _ = attention(x, T=frames, H_patches=14, W_patches=14)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
