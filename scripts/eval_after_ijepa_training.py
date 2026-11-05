"""Wait for an existing IJEPA run, then supervise full LIBERO and LIBERO-Plus."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from safetensors import safe_open

from benchmarks.vla_harness.checkpoint_contract import validate_ijepa_checkpoint
from benchmarks.vla_harness.libero_plus_protocol import task_catalog, text_cache_preflight
from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint
from benchmarks.vla_harness.run_resumable import (
    ROOT, atomic_json, log, process_identity, resources_idle,
)
from benchmarks.vla_harness.run_train_eval import alive


def status(queue, state, **details):
    atomic_json(queue / "status.json", {"state": state, "updated_at": time.time(), **details})
    log(f"{state}: {details}")


def deployment(run, checkpoint, target):
    """Project old native weights into a separate, self-contained eval directory."""
    from openwam.model.video_backbone.encoder.ijepa import IJEPAImageVideoEncoder

    target.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    encoder_cfg = cfg["model"]["video_backbone"]["encoder"]
    if encoder_cfg["name"] != "ijepa":
        raise ValueError("This queue expects native IJEPA weights")
    config_text = (run / "config.yaml").read_text()
    config_path = target / "config.yaml"
    if config_path.exists() and config_path.read_text() != config_text:
        raise ValueError("Training config changed after deployment preparation")
    config_path.write_text(config_text)
    for name in (checkpoint, "normalization_stats.npy"):
        link = target / name
        if link.is_symlink():
            if link.resolve() != (run / name).resolve():
                raise ValueError(f"Deployment link changed: {link}")
        elif link.exists():
            raise ValueError(f"Expected a deployment link: {link}")
        else:
            link.symlink_to(run / name)
    # Use the same CPU constructor as deployment, without reading pretrained
    # weights or allocating anything on GPUs occupied by training.
    encoder = IJEPAImageVideoEncoder.from_skeleton(
        {}, encoder_cfg=encoder_cfg, ckpt_dir=str(run),
    )
    encoder.save_deploy_assets(str(target), cfg)
    with safe_open(run / checkpoint, framework="np") as weights:
        saved = {
            key.removeprefix("video_backbone.video_encoder."): weights.get_slice(key).get_shape()
            for key in weights.keys() if key.startswith("video_backbone.video_encoder.")
        }
    expected = {key: list(value.shape) for key, value in encoder.state_dict().items()}
    if saved != expected:
        missing, extra = sorted(expected.keys() - saved.keys()), sorted(saved.keys() - expected.keys())
        raise ValueError(f"Native IJEPA reconstruction differs: missing={missing}, extra={extra}")
    validate_ijepa_checkpoint(target, checkpoint, cfg)
    deploy_cfg = yaml.safe_load((ROOT / "configs/deploy.yaml").read_text())
    deploy_cfg["optimization"]["dit_cache"]["enabled"] = False
    deploy_cfg["optimization"]["compile"]["enabled"] = False
    deploy_cfg["inference"].update(
        denoise_steps=10, denoise_mode="sync", inference_mode="sync", inference_horizon=10,
    )
    (target / "vla_harness_deploy.yaml").write_text(yaml.safe_dump(deploy_cfg, sort_keys=False))
    atomic_json(target / "source.json", {
        "source_run": str(run), "checkpoint": checkpoint,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "encoder_tensors_validated": len(saved), "weights_copied": False,
    })
    return target


def preflight(run, checkpoint, target, plus_cache):
    deployment(run, checkpoint, target)
    plus = prepare_checkpoint(
        target, checkpoint, target / "eval_libero_plus" / checkpoint.removesuffix(".safetensors") / "corrected_text",
        plus_cache, corrected=True,
    )
    args = SimpleNamespace(
        ckpt_dir=plus, libero_path=ROOT.parent / "benchmarks/LIBERO-plus",
        libero_plus_python=Path(sys.executable).parent.parent.parent / "libero-plus/bin/python",
        correct_instruction_suffix=True,
    )
    tasks = task_catalog(args.libero_path)
    cache = text_cache_preflight(args, tasks)
    manifests = []
    for benchmark, directory, workers, batch, port in (
        ("libero", target, 40, 16, 8990), ("libero-plus", plus, 128, 32, 8994),
    ):
        command = [
            sys.executable, "-m", "benchmarks.vla_harness.run_eval",
            "--benchmark", benchmark, "--ckpt-dir", str(directory), "--ckpt-name", checkpoint,
            "--gpus", "0,1,2,3", "--workers", str(workers), "--max-batch-size", str(batch),
            "--render-gpus", "0,1,2,3", "--base-port", str(port), "--dry-run",
        ]
        plan = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
        atomic_json(target / f"{benchmark}_plan.json", plan)
        manifests.append({
            "benchmark": benchmark, "tasks": len(plan["plan"]["tasks"]),
            "trials_per_task": plan["plan"]["num_trials"],
            "workers": len(plan["plan"]["workers"]), "max_batch_size": batch,
        })
    result = {"plans": manifests, "plus_text_cache": cache, "gpu_work_started": False}
    atomic_json(target / "preflight.json", result)
    log(f"CPU preflight passed: {result}")


def completed_checkpoint(run, training_log):
    """Require completed progress, final weights and the last-written resume marker."""
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    step = int(cfg["training"]["max_steps"])
    name = f"checkpoint_step_{step}.safetensors"
    text = training_log.read_text(errors="replace")
    final = re.findall(r"\[checkpoint\] Saved final: ([^\r\n]+)", text)
    if not any(Path(path.strip()).name == name for path in final):
        raise RuntimeError(f"Training exited without Saved final: {name}")
    progress = re.findall(r"Training:\s*100%[^\r\n]*?\b(\d+)/(\d+)", text)
    if (str(step), str(step)) not in progress:
        raise RuntimeError("Training exited without the expected completed N/N progress")
    if re.search(r"Traceback \(most recent call last\)|ChildFailedError|CUDA out of memory|\bloss=(?:nan|inf)\b", text):
        raise RuntimeError("Training log contains a failure; refusing to evaluate incomplete training")
    marker = run / f"accel_state_step_{step}" / "trainer_state.json"
    if json.loads(marker.read_text())["global_step"] != step:
        raise RuntimeError("Final optimizer/scheduler resume state is incomplete")
    with safe_open(run / name, framework="np") as weights:
        if not weights.keys():
            raise RuntimeError("Empty final checkpoint")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-pids", default="")
    parser.add_argument("--plus-cache", type=Path, required=True)
    parser.add_argument("--preflight-checkpoint")
    args = parser.parse_args()
    run, queue = args.run_dir.resolve(), args.output_dir.resolve()
    queue.mkdir(parents=True, exist_ok=True)
    if args.preflight_checkpoint:
        preflight(run, args.preflight_checkpoint, queue / "preflight", args.plus_cache)
        return 0
    with (queue / "waiter.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job_path = queue / "training.json"
        if not job_path.exists():
            processes = [process_identity(int(pid)) for pid in args.training_pids.split(",") if pid]
            if not processes or any(record is None for record in processes):
                raise ValueError("Capture live training PIDs on the first queue launch")
            atomic_json(job_path, {"run": str(run), "log": str(args.training_log), "processes": processes})
        job = json.loads(job_path.read_text())
        if job["run"] != str(run) or job["log"] != str(args.training_log):
            raise ValueError("Queue belongs to a different training run")
        stop_file = queue / "STOP"
        while not stop_file.exists():
            active = [record["pid"] for record in job["processes"] if alive(record)]
            if active:
                status(queue, "waiting_for_training", active_training_pids=active)
                time.sleep(30)
                continue
            checkpoint = completed_checkpoint(run, args.training_log)
            if not resources_idle("0,1,2,3", range(8990, 8998)):
                status(queue, "waiting_for_gpus_and_ports", checkpoint=checkpoint)
                time.sleep(30)
                continue
            # CPU validation runs again against the final checkpoint before any GPU work.
            preflight(run, checkpoint, queue / "deployment", args.plus_cache)
            os.environ["TEXT_EMBEDDING_DIR"] = str(args.plus_cache.resolve())
            os.environ["OPENWAM_PYTHON"] = sys.executable
            command = [
                "bash", str(ROOT / "scripts/eval_libero_then_plus.sh"),
                str(queue / "deployment"), checkpoint, str(queue / "evaluation"),
                "--gpus", "0,1,2,3", "--workers", "40", "--max-batch-size", "16",
                "--render-gpus", "0,1,2,3", "--base-port", "8990",
                "--plus-workers", "128", "--plus-max-batch-size", "32",
                "--plus-render-gpus", "0,1,2,3", "--plus-base-port", "8994",
                "--correct-instruction-suffix", "true", "--retry-delay", "30",
                "--wait-for-idle", "--stop-file", str(stop_file),
            ]
            atomic_json(queue / "evaluation_command.json", command)
            status(queue, "evaluating", checkpoint=checkpoint, command=command)
            with (queue / "evaluation_launcher.log").open("a") as handle:
                code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if (queue / "evaluation/DONE").is_file():
                atomic_json(queue / "DONE", {"checkpoint": checkpoint, "finished_at": time.time()})
                status(queue, "complete", checkpoint=checkpoint)
                return 0
            if not stop_file.exists():
                status(queue, "retrying_evaluation_supervisor", exit_code=code)
                time.sleep(30)
        status(queue, "stopped")
        return 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Leave a readable status on a terminal failure (e.g. failed training).
        if "--output-dir" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            if output.exists():
                status(output, "failed", error=repr(exc))
        raise
