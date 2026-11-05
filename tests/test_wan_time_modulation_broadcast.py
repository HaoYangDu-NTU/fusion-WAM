"""Per-sample time MLPs must preserve per-token outputs and gradients."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.model.video_backbone.wan.dit_forward import build_time_modulation
from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d


def _dit(dtype):
    return SimpleNamespace(
        seperated_timestep=True, freq_dim=8, dim=16,
        time_embedding=nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 16)).to(dtype),
        time_projection=nn.Sequential(nn.SiLU(), nn.Linear(16, 96)).to(dtype),
    )


def _legacy(dit, timestep, latents, patch, clean):
    b, _, f, h, w = latents.shape
    n = h * w // (patch[1] * patch[2])
    ts = torch.ones(b, f, n, dtype=latents.dtype) * timestep.view(b, 1, 1)
    ts[:, :max(clean, 1)] = 0
    emb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, ts.reshape(-1)).to(latents.dtype))
    emb = emb.reshape(b, -1, dit.dim)
    return emb, dit.time_projection(emb).unflatten(2, (6, dit.dim))


def _run(dit, ts, latents, patch, clean):
    return build_time_modulation(
        dit, ts, latents, patch_size=patch, fuse_vae_embedding_in_latents=True,
        force_per_token_t_mod=True, num_clean_condition_latents=clean,
        zero_clean_prefix_t_mod=True, has_clean_condition_latents=True,
    )


@pytest.mark.parametrize('patch', [(1, 1, 1), (1, 2, 2)])
@pytest.mark.parametrize('frames,clean', [(1, 0), (5, 0), (5, 1), (5, 2), (5, 8)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
def test_time_modulation_matches_per_token_reference(patch, frames, clean, dtype):
    torch.manual_seed(42)
    dit = _dit(dtype)
    ref = copy.deepcopy(dit)
    latents = torch.empty(3, 4, frames, 4, 6, dtype=dtype)
    # Float32 timesteps with bf16 latents pin the original promotion semantics.
    ts = torch.tensor([0.0, 127.25, 719.5], dtype=torch.float64 if dtype == torch.float64 else torch.float32,
                      requires_grad=dtype != torch.bfloat16)
    ref_ts = ts.detach().clone().requires_grad_(ts.requires_grad)
    outputs = _run(dit, ts, latents, patch, clean)
    expected = _legacy(ref, ref_ts, latents, patch, clean)
    tol = 1e-12 if dtype == torch.float64 else 2e-6 if dtype == torch.float32 else 0.005
    for actual, target in zip(outputs, expected):
        torch.testing.assert_close(actual, target, atol=tol, rtol=tol)
    weights = [torch.randn_like(x) for x in outputs]
    params = [*dit.time_embedding.parameters(), *dit.time_projection.parameters()]
    ref_params = [*ref.time_embedding.parameters(), *ref.time_projection.parameters()]
    # Training samples timesteps without gradients; test timestep derivatives
    # separately in fp32/fp64, avoiding bf16 cancellation in the sinusoid.
    if ts.requires_grad:
        params.insert(0, ts)
        ref_params.insert(0, ref_ts)
    grads = torch.autograd.grad(outputs, params, grad_outputs=weights)
    ref_grads = torch.autograd.grad(expected, ref_params, grad_outputs=weights)
    for grad, target in zip(grads, ref_grads):
        # Broadcasting changes the reduction order, particularly in bf16.
        if dtype == torch.bfloat16:
            error = (grad.float() - target.float()).norm()
            assert error <= 0.015 * target.float().norm() + 1e-5
        else:
            torch.testing.assert_close(grad, target, atol=tol * 50, rtol=tol * 50)


def test_time_mlps_process_two_timesteps_per_sample():
    dit = _dit(torch.float32)
    counts = []
    handles = [module.register_forward_pre_hook(lambda _, args: counts.append(args[0].numel() // args[0].shape[-1]))
               for module in (dit.time_embedding, dit.time_projection)]
    try:
        _run(dit, torch.tensor([12., 24.]), torch.empty(2, 4, 5, 24, 20), (1, 1, 1), 1)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == [4, 4], f'time MLPs repeated across tokens: {counts}'
