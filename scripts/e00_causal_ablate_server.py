"""Intervene only on the added router's post-softmax value weights."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from benchmarks.vla_harness.model_server import OpenWAMModelServer
from e00_sequence_probe_server import digest

MODES = ("baseline", "top32", "random32", "equal_mass")
MASK_SEED = 17011
TOP_K = 32
original_load = OpenWAMModelServer._load_model
original_predict = OpenWAMModelServer._predict_batch


class ValueIntervention(TorchFunctionMode):
    """Scope the intervention to one native ActionVisualRouter.forward."""

    def __init__(self, condition, step, target_mass=None):
        super().__init__()
        self.condition = condition
        self.step = step
        self.target_mass = target_mass
        self.original_weights = None
        self.removed_mass = None
        self.value_ratio = None

    def modify(self, weights):
        if weights.ndim != 4 or weights.dtype != torch.float32 or weights.shape[-1] <= TOP_K:
            raise ValueError("Expected >32 clean tokens in native FP32 [B,H,Q,T] router attention")
        if self.original_weights is not None:
            raise RuntimeError("Expected exactly one router softmax")
        self.original_weights = weights
        if self.condition == "baseline":
            changed = weights
        elif self.condition == "equal_mass":
            if self.target_mass is None or self.target_mass.shape != weights.shape[:-1]:
                raise ValueError("Invalid top32 removed-mass reference")
            changed = weights * (1 - self.target_mass.unsqueeze(-1))
        else:
            scores = weights
            if self.condition == "random32":
                # Fresh local generator, deterministic for this denoising index on every task/checkpoint.
                generator = torch.Generator(device=weights.device).manual_seed(MASK_SEED + self.step)
                scores = torch.rand(weights.shape, device=weights.device, generator=generator)
            elif self.condition != "top32":
                raise ValueError(f"Unknown intervention: {self.condition}")
            selected = scores.topk(TOP_K, dim=-1).indices
            changed = weights.scatter(-1, selected, 0.)
        self.removed_mass = (weights - changed).sum(-1)
        if self.target_mass is not None and not torch.allclose(self.removed_mass, self.target_mass, atol=2e-6, rtol=0):
            raise ValueError("Uniform attenuation failed to match the top32 removed mass")
        return changed

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.Tensor.softmax:
            return self.modify(result)
        if func is torch.Tensor.matmul and self.original_weights is not None:
            # This is the only matmul after softmax inside the native router.
            if self.value_ratio is not None or args[0].shape != self.original_weights.shape:
                raise RuntimeError("Unexpected matmul after router softmax")
            if self.condition == "baseline":
                self.value_ratio = 0.0
            else:
                native = self.original_weights.to(args[1].dtype) @ args[1]
                self.value_ratio = float((result.float() - native.float()).norm() /
                                         native.float().norm().clamp_min(1e-12))
        return result


def append(name, value):
    with (Path(os.environ["E00_CAUSAL_CASE"]) / name).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def load(server, *args, **kwargs):
    original_load(server, *args, **kwargs)
    router = server.engine.architecture.action_router
    if not (router.fp32_gate and router.fp32_residual and router.gate_logit.dtype == torch.float32
            and router.top_k == TOP_K):
        raise ValueError("Expected a trained FP32 gate/residual router configured with K=32")
    append("loaded.jsonl", {"gate": float(router.gate_logit.float().sigmoid()),
                            "gate_logit": float(router.gate_logit), "sparse_weight": router.sparse_weight})
    native_forward = router.forward
    native_prepare = server.engine._prepare

    def forward(*a, **kw):
        if kw.get("return_aux"):
            raise RuntimeError("Value intervention requires the inference-only router path")
        condition = server.condition
        step = len(server.pass_stats)
        target = server.top_mass[step] if condition == "equal_mass" else None
        intervention = ValueIntervention(condition, step, target)
        with intervention:
            output = native_forward(*a, **kw)
        if intervention.value_ratio is None or intervention.removed_mass is None:
            raise RuntimeError("Native router value aggregation was not observed")
        mass = intervention.removed_mass
        if condition == "top32":
            server.top_mass.append(mass.detach())
        server.pass_stats.append({
            "rows": mass.numel(), "tokens": intervention.original_weights.shape[-1],
            "removed_mass_sum": float(mass.sum()), "value_mix_relative_l2": intervention.value_ratio,
        })
        return output

    def prepare(observations):
        inputs, actions, schedule = native_prepare(observations)
        prepared = {"action_noise": actions, "video_noise": inputs["noise"],
                    "conditioned_video_latents": inputs["latents"], "text": inputs["context"]}
        if "proprio" in inputs:
            prepared["proprio"] = inputs["proprio"]
        server.hashes.append({name: digest(tensor) for name, tensor in prepared.items()})
        return inputs, actions, schedule

    router.forward = forward
    server.engine._prepare = prepare


def predict(server, observations, contexts):
    if len(observations) != 1:
        raise ValueError("The fixed-state intervention requires a serial batch")
    server.hashes = []
    server.top_mass = []
    runs = {}
    for mode in MODES:
        server.condition = mode
        server.pass_stats = []
        output = original_predict(server, observations, contexts)
        if mode == "baseline":
            primary = output
        runs[mode] = {"router_calls": server.pass_stats,
                      "actions": np.asarray(output[0]["actions"]).tolist()}
    counts = {len(r["router_calls"]) for r in runs.values()}
    if len(counts) != 1 or 0 in counts or len(server.hashes) != len(MODES) or any(
        h != server.hashes[0] for h in server.hashes
    ):
        raise RuntimeError("Sampler inputs, noise or router call counts differ between interventions")
    ctx = contexts[0]
    append("observations.jsonl", {"sid": ctx.session_id, "eid": ctx.episode_id,
                                 "input_sha256": server.hashes[0], "conditions": runs})
    return primary


if __name__ == "__main__":
    OpenWAMModelServer._load_model = load
    OpenWAMModelServer._predict_batch = predict
    from vla_eval.model_servers.serve import run_server
    run_server(OpenWAMModelServer)
