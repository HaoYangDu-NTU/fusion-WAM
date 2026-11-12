"""E00+ clean-visual residual readout; soft routing, not token pruning."""

import math

import torch
from torch import nn


class ActionVisualRouter(nn.Module):
    def __init__(self, action_dim, visual_dim, *, dim=512, num_heads=8,
                 top_k=32, gate_init=0.01, sparse_weight=0.01,
                 warmup_steps=1000, ramp_steps=4000, fp32_gate=False,
                 fp32_residual=False):
        super().__init__()
        for name, value in dict(dim=dim, num_heads=num_heads, top_k=top_k).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"action_router.{name} must be a positive integer")
        if dim % num_heads:
            raise ValueError("action_router.dim must be divisible by num_heads")
        for name, value in dict(warmup_steps=warmup_steps, ramp_steps=ramp_steps).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"action_router.{name} must be a nonnegative integer")
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("action_router.gate_init must be in (0, 1)")
        if not math.isfinite(sparse_weight) or sparse_weight < 0:
            raise ValueError("action_router.sparse_weight must be finite and nonnegative")
        self.num_heads, self.head_dim, self.top_k = num_heads, dim // num_heads, top_k
        self.sparse_weight = float(sparse_weight)
        self.warmup_steps, self.ramp_steps = warmup_steps, ramp_steps
        self.fp32_gate = bool(fp32_gate)
        self.fp32_residual = bool(fp32_residual)
        if self.fp32_residual and not self.fp32_gate:
            raise ValueError("action_router.fp32_residual requires fp32_gate")
        self.action_norm = nn.LayerNorm(action_dim, elementwise_affine=False)
        self.visual_norm = nn.LayerNorm(visual_dim, elementwise_affine=False)
        self.q = nn.Linear(action_dim, dim)
        self.k = nn.Linear(visual_dim, dim)
        self.v = nn.Linear(visual_dim, dim)
        self.out = nn.Linear(dim, action_dim)
        # Keep the logit FP32 in arithmetic, even when parameters are BF16.
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1 - gate_init))))

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        # The model is moved/cast before optimizer construction. Preserve this
        # one scalar's FP32 storage when the surrounding Router uses BF16.
        if self.fp32_gate and self.gate_logit.dtype != torch.float32:
            self.gate_logit.data = self.gate_logit.data.float()
        return module

    def coefficient(self, optimizer_step):
        if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int) or optimizer_step < 0:
            raise ValueError("router optimizer_step must be a nonnegative integer")
        if optimizer_step < self.warmup_steps:
            return 0.0
        progress = 1.0 if not self.ramp_steps else min(
            (optimizer_step - self.warmup_steps) / self.ramp_steps, 1.0)
        return self.sparse_weight * progress

    def forward(self, actions, clean, *, query_valid=None, view_ids=None, return_aux=False):
        if clean.ndim != 3 or clean.shape[1] == 0:
            raise ValueError("action_router requires a nonempty [B, clean_tokens, D] context")
        b, queries, _ = actions.shape
        a, c = self.action_norm(actions), self.visual_norm(clean)

        def linear(module, inputs):
            # ZeRO may keep the trainable Router matrix in FP32 when the Gate
            # is FP32, while the frozen M29 streams remain BF16. Preserve the
            # stream compute dtype and let gradients flow through this cast to
            # the optimizer's FP32 parameter/master copy.
            return torch.nn.functional.linear(
                inputs,
                module.weight.to(dtype=inputs.dtype),
                None if module.bias is None else module.bias.to(dtype=inputs.dtype),
            )

        def heads(x):
            return x.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(linear(self.q, a)), heads(linear(self.k, c)), heads(linear(self.v, c))
        # Explicit FP32 softmax: the small router alone needs attention weights.
        # Keep this outside AMP so BF16 matmul does not hide logit growth.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            weights = logits.softmax(dim=-1)
        routed = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b, queries, -1)
        gate = self.gate_logit.float().sigmoid()
        routed_output = linear(self.out, routed)
        if self.fp32_residual:
            delta = routed_output.float() * gate
            updated = (actions.float() + delta).to(actions.dtype)
        else:
            delta = routed_output * gate.to(actions.dtype)
            updated = actions + delta
        if not return_aux:
            return updated, None
        if query_valid is None:
            query_valid = torch.ones((b, queries), device=actions.device, dtype=torch.bool)
        if query_valid.shape != (b, queries):
            raise ValueError("router query_valid must match action batch and horizon")
        valid = query_valid.to(device=actions.device, dtype=torch.float32)

        def mean_rows(x):
            # x is [B, heads, queries]. All-padded batches have zero row averages.
            return (x * valid[:, None]).sum() / (valid.sum() * self.num_heads).clamp_min(1)

        selected = weights.topk(min(self.top_k, clean.shape[1]), dim=-1)
        mass = selected.values.sum(-1)
        aux = {"router_sparse_raw": mean_rows(1 - mass)}
        with torch.no_grad():
            detached = weights.detach()
            support = torch.zeros_like(detached, dtype=torch.bool).scatter_(-1, selected.indices, True)
            support &= query_valid.to(device=actions.device, dtype=torch.bool)[:, None, :, None]
            coverage = support.any(dim=1).any(dim=1).float().mean(-1)
            sample_valid = valid.any(-1).float()
            aux.update({
                "router_topk_mass": mean_rows(mass.detach()),
                "router_entropy": mean_rows(-(detached * detached.clamp_min(1e-12).log()).sum(-1)),
                "router_effective_tokens": mean_rows(detached.square().sum(-1).reciprocal()),
                "router_tokens90": mean_rows((detached.sort(descending=True).values.cumsum(-1) < 0.9).sum(-1).float() + 1),
                "router_q_norm": mean_rows(q.float().norm(dim=-1)),
                "router_k_norm": k.float().norm(dim=-1).mean(),
                "router_logits_std": mean_rows(logits.std(dim=-1, unbiased=False)),
                "router_gate": gate.detach(),
                "router_topk_union_coverage": (coverage * sample_valid).sum() / sample_valid.sum().clamp_min(1),
                "router_residual_ratio": ((delta.float().norm(dim=-1) /
                    actions.float().norm(dim=-1).clamp_min(1e-6)) * valid).sum() / valid.sum().clamp_min(1),
            })
            if view_ids is not None:
                for view in (0, 1):
                    aux[f"router_view{view}_mass"] = mean_rows(detached[..., view_ids == view].sum(-1))
        return updated, aux
