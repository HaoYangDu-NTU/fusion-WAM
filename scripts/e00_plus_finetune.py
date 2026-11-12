"""Prepare and run matched B0/B1/B2 warm-start pilots on idle GPUs.

Only initialization gains router tensors. Production checkpoint loaders remain
strict. No source checkpoint/config is modified. The queue is bounded to GPU
smokes, an initial B0 LIBERO evaluation, and 1000-step pilots + LIBERO each.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf, open_dict

PROFILES = {"B0": "E00", "B1": "E00+-Dense", "B2": "E00+"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(source, target):
    """Reject semantic changes which shape-compatible weights cannot detect."""
    paths = (
        "model.video_backbone.name", "model.video_backbone.text_encoder_type",
        "model.video_backbone.vlm_text_embedding_method", "model.video_backbone.encoder.name",
        "model.video_backbone.encoder.use_avg_t_pooling", "model.video_backbone.encoder.use_wan_style_patch_grid",
        "model.video_backbone.encoder.history_num_tubelets", "model.video_backbone.encoder.seperate_encode_view",
        "model.video_backbone.encoder.bottleneck_dim", "model.video_backbone.encoder.condition_fusion_method",
        "model.video_backbone.encoder.future_fusion_method", "model.video_backbone.shift_video",
        "model.action_backbone.shift_action", "model.architecture.framework", "model.architecture.variant",
        "model.architecture.video_prediction_type", "model.architecture.action_prediction_type",
        "model.architecture.attention_mask_mode", "model.architecture.video_attention_mask_mode",
        "model.architecture.action_dim", "model.architecture.state_dim", "model.architecture.use_proprioception",
        "model.video_backbone.dit_config.dim", "model.video_backbone.dit_config.num_layers",
        "model.video_backbone.dit_config.num_heads", "model.action_backbone.dim",
        "dataloader.num_frames", "dataloader.video_stride", "dataloader.window_stride",
        "dataloader.action_mode", "dataloader.normalize_mode", "dataloader.multiview",
        "dataloader.color_jitter.enabled", "training.noise_schedule_distribution",
    )
    for key in paths:
        before, after = OmegaConf.select(source, key), OmegaConf.select(target, key)
        if before != after:
            raise ValueError(f"Source is not the E00 baseline: {key}: {before!r} != {after!r}")
    if OmegaConf.select(source, "model.architecture.action_router.enabled", default=False):
        raise ValueError("Source baseline already has a router")


def plan_config(plan, variant):
    from openwam.experiments import resolve_experiment
    cfg, profile, overrides = resolve_experiment(PROFILES[variant])
    router = cfg.model.architecture.action_router
    if variant != "B0":
        router.warmup_steps, router.ramp_steps = plan.sparse_warmup, plan.sparse_ramp
    return cfg, profile, overrides


def prepare(plan):
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from openwam.experiments import preflight
    from openwam.model.architectures.utils.action_router import ActionVisualRouter
    torch.set_num_threads(4)
    root = ROOT / plan.output_root / "initialization"
    source = Path(plan.source_dir)
    weights = source / plan.source_checkpoint
    source_cfg = OmegaConf.load(source / "config.yaml")
    base, profile, _ = plan_config(plan, "B0")
    validate_source(source_cfg, base)
    cache = preflight(base, profile)
    if cache["model_fingerprint"] != source_cfg.model.video_backbone.text_embedding_fingerprint:
        raise ValueError("RynnBrain text cache fingerprint changed")
    source_sha = sha(weights)
    ready = root / "READY.json"
    if ready.exists():
        evidence = json.loads(ready.read_text())
        if evidence["source_sha256"] != source_sha or evidence["plan"] != OmegaConf.to_container(plan):
            raise ValueError("Initialization already exists with a different identity")
        return root
    if root.exists():
        raise ValueError(f"Incomplete initialization exists; inspect {root} before retrying")
    # Match the actual training reader's action/proprio normalization.
    from openwam.dataloader.registry import build_dataset
    dataset = build_dataset(base.dataloader, split="train")
    expected = np.load(source / "normalization_stats.npy", allow_pickle=True).item()
    actual = np.load(dataset.normalization_stats_path, allow_pickle=True).item()
    def equal(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        return np.array_equal(np.asarray(a), np.asarray(b))
    if not equal(expected, actual):
        raise ValueError("Dataset action/proprio normalization differs from the source checkpoint")
    root.mkdir(parents=True)
    state = load_file(str(weights), device="cpu")
    if any(k.startswith("action_router.") for k in state):
        raise ValueError("Source weights already contain a router")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(plan.seed))
        router = ActionVisualRouter(1024, 1536).to(torch.bfloat16)
    additional = {"action_router." + k: v for k, v in router.state_dict().items()}
    for variant in PROFILES:
        directory = root / variant
        directory.mkdir()
        cfg, _, _ = plan_config(plan, variant)
        with open_dict(cfg.model.video_backbone):
            cfg.model.video_backbone.components = source_cfg.model.video_backbone.components
        cfg.model.video_backbone.text_embedding_fingerprint = cache["model_fingerprint"]
        cfg.training.finetune_ckpt_path = None
        OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
        for name in ("manifest.json", "normalization_stats.npy"):
            shutil.copyfile(source / name, directory / name)
        destination = directory / "checkpoint_step_0.safetensors"
        if variant == "B0":
            destination.symlink_to(weights)
        elif variant == "B1":
            save_file({**state, **additional}, str(destination))
        else:
            os.link(root / "B1/checkpoint_step_0.safetensors", destination)
    # Audit tensor equality and the exact allowed key delta, not just shapes.
    from safetensors import safe_open
    with safe_open(str(root / "B1/checkpoint_step_0.safetensors"), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(state) | set(additional):
            raise RuntimeError("Unexpected augmented checkpoint keys")
        for key, tensor in state.items():
            if not torch.equal(tensor, handle.get_tensor(key)):
                raise RuntimeError(f"Shared pretrained tensor changed: {key}")
    evidence = {"plan": OmegaConf.to_container(plan), "source": str(weights), "source_sha256": source_sha,
                "shared_tensors": len(state), "added_router_tensors": sorted(additional),
                "added_router_parameters": sum(t.numel() for t in additional.values()),
                "shared_weights_exactly_equal": True, "B1_B2_initial_weights_identical": True,
                "normalization_matches_dataset": True, "text_fingerprint": cache["model_fingerprint"],
                "source_training_batch_per_gpu": source_cfg.training.batch_size,
                "note": "New optimizer/schedule at step0; source trained10690. B1/B2 share read-only initialization weights."}
    del state
    gc.collect()
    atomic_json(ready, evidence)
    print(f"Prepared {ready}", flush=True)
    return root


def training_command(plan, variant, smoke=False):
    _, _, overrides = plan_config(plan, variant)
    initial = ROOT / plan.output_root / "initialization" / variant
    stage = "smoke" if smoke else "train"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(ROOT / "scripts/train.py"), *overrides,
               f"training.finetune_ckpt_path={initial}", f"training.learning_rate={plan.learning_rate}",
               f"training.max_steps={plan.smoke_steps if smoke else plan.steps}",
               f"training.batch_size={plan.batch_size_per_gpu}",
               f"training.output_path={ROOT / plan.output_root / stage / variant}",
               "training.save_steps=500", "training.keep_last_k_ckpts=2",
               f"training.save_full_states_for_resume={'false' if smoke else 'true'}"]
    ready = ROOT / plan.output_root / "initialization/READY.json"
    if ready.exists():
        command.append("model.video_backbone.text_embedding_fingerprint=" + json.loads(ready.read_text())["text_fingerprint"])
    if variant != "B0":
        command += [f"model.architecture.action_router.warmup_steps={plan.sparse_warmup}",
                    f"model.architecture.action_router.ramp_steps={plan.sparse_ramp}"]
    return command


def validate_cpu(plan):
    import torch
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    if torch.cuda.is_available():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for CPU validation")
    torch.set_num_threads(4)
    root = ROOT / plan.output_root / "initialization"
    ready = json.loads((root / "READY.json").read_text())
    rows = []
    for variant in PROFILES:
        cfg, _, _ = plan_config(plan, variant)
        cfg.model.video_backbone.text_embedding_fingerprint = ready["text_fingerprint"]
        _, model, _ = build_architecture_from_ckpt_dir(str(root / variant), weights_required=True, override_cfg=cfg)
        model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
        for name, parameter in model.named_parameters():
            if parameter.is_meta or not torch.isfinite(parameter).all().item():
                raise RuntimeError(f"Unloaded/nonfinite parameter: {variant}/{name}")
        router = model.action_router
        if (router is not None) != (variant != "B0"):
            raise RuntimeError(f"Incorrect router state for {variant}")
        rows.append({"variant": variant, "strict_train_loader": True,
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "router_gate": float(router.gate_logit.float().sigmoid()) if router else None})
        del model
        gc.collect()
    atomic_json(root / "CPU_VALIDATED.json", {"validated_at": datetime.now().isoformat(),
                "initialization_manifest_sha256": sha(root / "READY.json"), "models": rows})
    print(f"CPU validation passed: {root}", flush=True)


def run_queue(plan, report):
    from benchmarks.vla_harness.run_resumable import resources_idle
    from openwam.experiments import runtime_env, source_fingerprint
    report.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (report / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    init = ROOT / plan.output_root / "initialization"
    ready = json.loads((init / "READY.json").read_text())
    validation = json.loads((init / "CPU_VALIDATED.json").read_text())
    if ready["plan"] != OmegaConf.to_container(plan) or validation["initialization_manifest_sha256"] != sha(init / "READY.json"):
        raise ValueError("Initialization / CPU validation identity changed")
    identity = {"plan": OmegaConf.to_container(plan), "source_fingerprint": source_fingerprint(),
                "initialization_manifest_sha256": sha(init / "READY.json")}
    if (report / "plan.json").exists() and json.loads((report / "plan.json").read_text()) != identity:
        raise ValueError("Queue identity changed; inspect existing work before continuing")
    atomic_json(report / "plan.json", identity)
    env = {**os.environ, **runtime_env(plan.gpus), "OPENBLAS_NUM_THREADS": "1"}
    sequence = [("smoke", v) for v in PROFILES] + [("initial_eval", "B0")]
    sequence += [(stage, v) for v in PROFILES for stage in ("train", "eval")]
    for number, (stage, variant) in enumerate(sequence):
        directory = report / f"{number:02d}_{stage}_{variant}"
        if (directory / "DONE.json").exists():
            continue
        if (directory / "command.json").exists():
            raise RuntimeError(f"Interrupted/failed stage requires inspection; not repeating it: {directory}")
        while not resources_idle(plan.gpus, ()):
            if (report / "STOP").exists():
                return
            atomic_json(report / "status.json", {"state": "waiting_for_gpus", "next": [stage, variant], "updated_at": datetime.now().isoformat()})
            time.sleep(plan.retry_seconds)
        if (report / "STOP").exists():
            return
        if source_fingerprint() != identity["source_fingerprint"]:
            raise RuntimeError("Code/config changed while queued; inspect before running mixed-source experiments")
        directory.mkdir(parents=True)
        if stage in ("smoke", "train"):
            command = training_command(plan, variant, smoke=stage == "smoke")
        else:
            run = (ROOT / plan.output_root / "initialization/B0" if stage == "initial_eval" else
                   max((ROOT / plan.output_root / "train" / variant).glob("*/config.yaml"), key=lambda p: p.stat().st_mtime).parent)
            checkpoint = "checkpoint_step_0.safetensors" if stage == "initial_eval" else f"checkpoint_step_{plan.steps}.safetensors"
            if not (run / checkpoint).is_file():
                raise FileNotFoundError(run / checkpoint)
            command = [sys.executable, "-m", "benchmarks.vla_harness.run_eval", "--benchmark", "libero",
                       "--ckpt-dir", str(run), "--ckpt-name", checkpoint, "--gpus", plan.gpus,
                       "--workers", str(plan.eval_workers), "--max-batch-size", str(plan.eval_batch),
                       "--render", "gpu", "--render-gpus", "3", "--image-format", "raw",
                       "--base-port", str(plan.eval_base_port + number * 4), "--output-dir", str(directory / "evaluation")]
        atomic_json(directory / "command.json", command)
        atomic_json(report / "status.json", {"state": "running", "stage": stage, "variant": variant, "started_at": datetime.now().isoformat()})
        with (directory / "console.log").open("w") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Stage exited {result.returncode}: {directory / 'console.log'}")
        if stage in ("smoke", "train"):
            run = max((ROOT / plan.output_root / stage / variant).glob("*/config.yaml"), key=lambda p: p.stat().st_mtime).parent
            completed = json.loads((run / "training_result.json").read_text())
            expected_steps = plan.smoke_steps if stage == "smoke" else plan.steps
            if completed["global_step"] != expected_steps or not (run / f"checkpoint_step_{expected_steps}.safetensors").is_file():
                raise RuntimeError(f"Training stopped before its matched budget: {run}")
            if variant != "B0":
                rows = [json.loads(line) for line in (run / "router_metrics.jsonl").read_text().splitlines()]
                if not all(math.isfinite(value) for row in rows for value in row.values() if isinstance(value, (int, float))):
                    raise RuntimeError(f"Nonfinite router diagnostics: {run}")
                if stage == "train" and variant == "B2" and max(row["router_sparse_weight"] for row in rows) < .0099:
                    raise RuntimeError("Sparse pilot never reached its intended regularization weight")
                atomic_json(directory / "router_summary.json", {"rows": len(rows), "last": rows[-1]})
            atomic_json(directory / "training_run.json", {"directory": str(run), **completed})
        if stage in ("eval", "initial_eval"):
            summary = json.loads((directory / "evaluation/summary.json").read_text())
            if not summary["complete"] or not summary["selected_run_complete"]:
                raise RuntimeError(f"Incomplete evaluation: {directory}")
        atomic_json(directory / "DONE.json", {"finished_at": datetime.now().isoformat()})
        write_results(report)
    atomic_json(report / "status.json", {"state": "complete", "finished_at": datetime.now().isoformat()})


def write_results(report):
    lines = ["# E00+ matched warm-start pilot", "", "Same E00/RynnBrain9B/V-JEPA checkpoint; each pilot receives1000 new updates.",
             "Scores appear only after complete standard LIBERO (2000 trials). No LIBERO-Plus in this queue.", "",
             "| Evaluation | Successes / trials | LIBERO |", "|---|---:|---:|"]
    records = {}
    for directory in sorted(report.glob("*")):
        path = directory / "evaluation/summary.json"
        if not (directory / "DONE.json").exists() or not path.exists():
            continue
        result = json.loads(path.read_text())
        if not result.get("complete") or not result.get("selected_run_complete"):
            continue
        successes = sum(row["successes"] for row in result["suites"])
        trials = sum(row["valid_trials"] for row in result["suites"])
        records[directory.name] = result
        lines.append(f"| {directory.name} | {successes}/{trials} | {result['total_average_success_rate']:.2%} |")
    (report / "RESULTS.md").write_text("\n".join(lines) + "\n")
    atomic_json(report / "results.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e00_plus_finetune.yaml")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/E00_plus/finetune_20261110")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--validate-cpu", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = OmegaConf.load(args.config)
    if args.prepare:
        prepare(plan)
    if args.validate_cpu:
        validate_cpu(plan)
    if args.run:
        if not (ROOT / plan.output_root / "initialization/CPU_VALIDATED.json").exists():
            raise ValueError("Run strict CPU loader validation before the GPU queue")
        try:
            run_queue(plan, args.report)
        except Exception as error:
            atomic_json(args.report / "status.json", {"state": "failed", "error": str(error), "updated_at": datetime.now().isoformat()})
            raise
    if not args.prepare and not args.run and not args.validate_cpu:
        print(json.dumps({"plan": OmegaConf.to_container(plan), "commands": {v: training_command(plan, v) for v in PROFILES}}, indent=2))


if __name__ == "__main__":
    main()
