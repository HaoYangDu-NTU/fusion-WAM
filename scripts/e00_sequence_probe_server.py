"""Instrument the report's paired sampler using the loaded router's actual forward."""
import hashlib
from pathlib import Path
import sys

import torch

DIAG = Path(__file__).resolve().parents[1] / "reports/E00_plus/diagnosis_20261111"
sys.path.insert(0, str(DIAG))
import diagnostic_server as diag


def residual_hook(server, module, args, kwargs, output):
    if not server._diag_enabled:
        raise RuntimeError("The router ran during disabled inference")
    if not server._probe_pair or not server._diag_collect:
        return
    before = args[0].detach().float()
    difference = output[0].detach().float() - before
    relative = difference.norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-6)
    metrics = dict(output[1])
    metrics["effective_residual_ratio"] = relative.mean()
    metrics["effective_residual_nonzero_fraction"] = difference.ne(0).float().mean()
    server._diag_rows.append({key: value.detach().float().reshape(1).cpu().numpy()
                              for key, value in metrics.items()})


def digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(data).hexdigest()


def load(server, *args, **kwargs):
    diag.load(server, *args, **kwargs)
    router = server.engine.architecture.action_router
    if not (router.fp32_gate and router.fp32_residual and router.gate_logit.dtype == torch.float32):
        raise ValueError("Probe requires the FP32 gate and FP32 residual configuration")
    server._probe_pair = False

    def request_native_aux(module, args, kwargs):
        if server._probe_pair and server._diag_collect:
            return args, {**kwargs, "return_aux": True}

    router.register_forward_pre_hook(request_native_aux, with_kwargs=True)
    original_prepare = server.engine._prepare

    def prepare(observations):
        inputs, actions, schedule = original_prepare(observations)
        if server._probe_pair:
            # Verify both noise tensors and the prepared image-conditioned input.
            server._probe_inputs.append({
                "action_noise": digest(actions), "video_noise": digest(inputs["noise"]),
                "conditioned_video_latents": digest(inputs["latents"]),
                "text": digest(inputs["context"]),
                **({"proprio": digest(inputs["proprio"])} if "proprio" in inputs else {}),
            })
        return inputs, actions, schedule

    server.engine._prepare = prepare


def predict(server, observations, contexts):
    ctx = contexts[0]
    key = (ctx.session_id, ctx.episode_id)
    server._probe_pair = diag.MODE == "on" and server._diag_paired.get(key, 0) < 2
    server._probe_inputs = []
    call = server._diag_call
    try:
        actions = diag.predict(server, observations, contexts)
        if server._probe_pair:
            values = server._probe_inputs
            if len(values) != 2 or values[0] != values[1] or not server._diag_rows:
                raise RuntimeError("Paired input/noise or router measurement verification failed")
            diag.append("paired_inputs.jsonl", {"call": call, "input_sha256": values[0]})
        return actions
    finally:
        server._probe_pair = False


if __name__ == "__main__":
    diag.hook = residual_hook
    diag.OpenWAMModelServer._load_model = load
    diag.OpenWAMModelServer._predict_batch = predict
    from vla_eval.model_servers.serve import run_server
    run_server(diag.OpenWAMModelServer)
