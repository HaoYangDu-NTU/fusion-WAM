"""Wan RoPE without materializing FP64 Q/K; arithmetic stays in FP64."""
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _rotate_kernel(X, PHASE, OUT, TOTAL: tl.constexpr, WIDTH: tl.constexpr,
                       TOKENS: tl.constexpr, HEAD_DIM: tl.constexpr,
                       CONJUGATE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < TOTAL
        d = i % HEAD_DIM
        token = i // WIDTH % TOKENS
        phase = (token * (HEAD_DIM // 2) + d // 2) * 2
        x = tl.load(X + i, valid, 0).to(tl.float64)
        partner = tl.load(X + (i ^ 1), valid, 0).to(tl.float64)
        cosine = tl.load(PHASE + phase, valid, 0).to(tl.float64)
        sine = tl.load(PHASE + phase + 1, valid, 0).to(tl.float64)
        if CONJUGATE:
            sine = -sine
        signed_partner = tl.where(d % 2 == 0, -partner, partner)
        result = x * cosine + signed_partner * sine
        # PyTorch converts double to half/bfloat16 through float32.
        tl.store(OUT + i, result.to(tl.float32), valid)


def _rotate(x, phase, heads, conjugate):
    out = torch.empty_like(x)
    _rotate_kernel[(triton.cdiv(x.numel(), 256),)](
        x, torch.view_as_real(phase), out, x.numel(), x.shape[-1], x.shape[1],
        x.shape[-1] // heads, conjugate, BLOCK=256, enable_fp_fusion=False,
    )
    return out


class _FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phase, heads):
        ctx.save_for_backward(phase)
        ctx.heads = heads
        return _rotate(x, phase, heads, False)

    @staticmethod
    def backward(ctx, grad):
        phase, = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve higher-order autograd through the original linear map.
            shape = (grad.shape[0], grad.shape[1], ctx.heads, -1, 2)
            z = torch.view_as_complex(grad.to(torch.float64).reshape(shape))
            dx = torch.view_as_real(z * phase.conj()).flatten(2).to(grad.dtype)
        else:
            dx = _rotate(grad.contiguous(), phase, ctx.heads, True)
        return dx, None, None


def rope_apply_fused(x, phase, heads):
    """Return None when the original PyTorch path is needed."""
    if (os.environ.get("OPENWAM_DISABLE_FUSED_ROPE", "").lower() in {"1", "true", "yes"}
            or triton is None or not x.is_cuda or x.ndim != 3 or not x.is_contiguous()
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or phase.requires_grad or phase.device != x.device or not phase.is_contiguous()
            or phase.dtype not in (torch.complex64, torch.complex128)
            or phase.is_conj() or heads <= 0 or x.shape[-1] % (2 * heads)):
        return None
    if phase.shape != (x.shape[1], 1, x.shape[-1] // heads // 2):
        return None
    return _FusedRoPE.apply(x, phase, heads)
