"""Fused frozen-encoder RoPE rotation; keep PyTorch's original phase values."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _rotate_kernel(X, SIN, COS, OUT, TOTAL: tl.constexpr,
                       HEADS: tl.constexpr, TOKENS: tl.constexpr, DIM: tl.constexpr,
                       S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
                       CLS: tl.constexpr, REGISTERS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < TOTAL
        d = i % DIM
        n = i // DIM % TOKENS
        h = i // (DIM * TOKENS) % HEADS
        b = i // (DIM * TOKENS * HEADS)
        base = b * S0 + h * S1 + n * S2
        x = tl.load(X + base + d * S3, valid, 0).to(tl.float32)
        partner = tl.load(X + base + (d ^ 1) * S3, valid, 0).to(tl.float32)
        rotate = (n >= CLS) & (n < TOKENS - REGISTERS)
        phase = (n - CLS) * (DIM // 2) + d // 2
        sine = tl.load(SIN + phase, valid & rotate, 0)
        cosine = tl.load(COS + phase, valid & rotate, 1)
        signed_partner = tl.where(d % 2 == 0, -partner, partner)
        # Separate FP32 products and addition match eager; launch disables FMA.
        result = x * cosine + signed_partner * sine
        tl.store(OUT + i, tl.where(rotate, result, x), valid)


def rotate_frozen(x, pos, n_registers, has_cls_first):
    """Return the fused result, or None for inputs needing the original path."""
    if (triton is None or not x.is_cuda or (torch.is_grad_enabled() and x.requires_grad)
            or pos.ndim != 1 or pos.dtype != torch.float32 or pos.requires_grad
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return None
    batch, heads, tokens, dim = x.shape
    n_cls = int(has_cls_first)
    if dim % 2 or pos.numel() != tokens - n_cls - n_registers:
        return None
    # Compute every phase table with the original PyTorch operations and dtypes.
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000 ** omega
    freq = torch.einsum('..., f -> ... f', pos, omega)
    sine, cosine = freq.sin(), freq.cos()
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rotate_kernel[(triton.cdiv(x.numel(), 512),)](
        x, sine, cosine, out, x.numel(), heads, tokens, dim, *x.stride(), n_cls, n_registers,
        BLOCK=512, enable_fp_fusion=False,
    )
    return out


if triton is not None:
    @triton.jit
    def _rotate_qk_kernel(Q, K, DS, DC, HS, HC, WS, WC, QO, KO,
                          TOTAL: tl.constexpr, HEADS: tl.constexpr, TOKENS: tl.constexpr,
                          DIM: tl.constexpr, AXIS: tl.constexpr,
                          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
                          CLS: tl.constexpr, REGISTERS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < TOTAL
        d = i % DIM
        n = i // DIM % TOKENS
        h = i // (DIM * TOKENS) % HEADS
        b = i // (DIM * TOKENS * HEADS)
        base = b * S0 + h * S1 + n * S2
        rotate = valid & (n >= CLS) & (n < TOKENS - REGISTERS) & (d < 3 * AXIS)
        phase = (n - CLS) * (AXIS // 2) + (d % AXIS) // 2
        ds = tl.load(DS + phase, rotate & (d < AXIS), 0)
        dc = tl.load(DC + phase, rotate & (d < AXIS), 0)
        hs = tl.load(HS + phase, rotate & (d >= AXIS) & (d < 2 * AXIS), 0)
        hc = tl.load(HC + phase, rotate & (d >= AXIS) & (d < 2 * AXIS), 0)
        ws = tl.load(WS + phase, rotate & (d >= 2 * AXIS), 0)
        wc = tl.load(WC + phase, rotate & (d >= 2 * AXIS), 0)
        sine = ds + hs + ws
        cosine = dc + hc + wc
        q = tl.load(Q + base + d * S3, valid, 0).to(tl.float32)
        k = tl.load(K + base + d * S3, valid, 0).to(tl.float32)
        qp = tl.load(Q + base + (d ^ 1) * S3, rotate, 0).to(tl.float32)
        kp = tl.load(K + base + (d ^ 1) * S3, rotate, 0).to(tl.float32)
        qr = q * cosine + tl.where(d % 2 == 0, -qp, qp) * sine
        kr = k * cosine + tl.where(d % 2 == 0, -kp, kp) * sine
        tl.store(QO + i, tl.where(rotate, qr, q), valid)
        tl.store(KO + i, tl.where(rotate, kr, k), valid)


def rotate_qk_frozen(q, k, positions, axis_dim, n_registers, has_cls_first):
    """Fuse all three axes for frozen Q/K; retain eager phase arithmetic.

    No persistent cache: positions may change between views, context/target,
    masks or resolutions. Q and K share only this call's identical phase tables.
    """
    if (triton is None or not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad))
            or q.shape != k.shape or q.stride() != k.stride() or q.dtype != k.dtype
            or axis_dim <= 0 or axis_dim % 2 or 3 * axis_dim > q.shape[-1]):
        return None
    batch, heads, tokens, dim = q.shape
    n_cls = int(has_cls_first)
    if any(pos.ndim != 1 or pos.dtype != torch.float32 or pos.requires_grad
           or pos.device != q.device or pos.numel() != tokens - n_cls - n_registers for pos in positions):
        return None
    omega = torch.arange(axis_dim // 2, dtype=q.dtype, device=q.device)
    omega /= axis_dim / 2.0
    omega = 1.0 / 10000 ** omega
    tables = []
    for pos in positions:
        freq = torch.einsum('..., f -> ... f', pos, omega)
        tables.extend((freq.sin(), freq.cos()))
    qo, ko = torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(k, memory_format=torch.contiguous_format)
    _rotate_qk_kernel[(triton.cdiv(q.numel(), 256),)](
        q, k, *tables, qo, ko, q.numel(), heads, tokens, dim, axis_dim,
        *q.stride(), n_cls, n_registers, BLOCK=256, enable_fp_fusion=False,
    )
    return qo, ko
