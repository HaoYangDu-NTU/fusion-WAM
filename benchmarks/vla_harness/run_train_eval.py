"""One training run, then resumable full LIBERO and LIBERO-Plus on this host."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml
from safetensors import safe_open

from benchmarks.vla_harness.run_resumable import (
    OWNER_KEY, ROOT, atomic_json, log, owned_processes, resources_idle, stop_owned_processes,
    process_identity as identity,
)

TRAIN_SCRIPT = "scripts/train_libero_vjepa_1_3b_dev_share_dir.sh"
POLL_SECONDS = 10
RETRY_SECONDS = 30


def read_json(path):
    return json.loads(path.read_text())


def alive(record):
    current = identity(record["pid"])
    return (current is not None and current["starttime"] == record["starttime"]
            and current.get("boot_id") == record.get("boot_id") and current["state"] != "Z")


def stop_requested(queue, run=None):
    paths = [queue / "STOP", queue / "evaluation" / "STOP"]
    paths += [queue / "evaluation" / stage / "STOP" for stage in ("libero", "libero-plus")]
    if run:
        paths.append(run / "STOP")
    return any(path.exists() for path in paths)


def status(queue, state, **details):
    atomic_json(queue / "status.json", {"state": state, "updated_at": time.time(), **details})
    log(state)


def wait_resources(queue, run=None):
    while not stop_requested(queue, run):
        if resources_idle("0,1,2,3", range(8990, 8994)):
            return True
        status(queue, "waiting_for_gpus_and_ports")
        time.sleep(RETRY_SECONDS)
    return False


def training_worker(queue):
    """Own the exit record even if the parent supervisor is disconnected/killed."""
    job = read_json(queue / "job.json")
    train = queue / "training"
    with (train / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (train / "launch.json").exists():
            raise RuntimeError("Training already launched; refusing a second training run")
        atomic_json(train / "launch.json", {"worker": identity(os.getpid()), "command": job["command"]})
        code = 1
        try:
            if stop_requested(queue):
                code = 130
                return 130
            if hashlib.sha256(Path(job["command"][1]).read_bytes()).hexdigest() != job["script_sha256"]:
                raise RuntimeError("Training script changed before launch; use a new queue directory")
            with (train / "job.log").open("w") as handle:
                process = subprocess.Popen(job["command"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True,
                                           env={**os.environ, OWNER_KEY: job["train_id"], "PYTHONUNBUFFERED": "1"})
                atomic_json(train / "process.json", {"process": identity(process.pid), "command": job["command"]})
                try:
                    while process.poll() is None and not stop_requested(queue):
                        time.sleep(POLL_SECONDS)
                finally:
                    stop_owned_processes(job["train_id"])
                    code = process.wait()
        finally:
            atomic_json(train / "exit.json", {"exit_code": code, "finished_at": time.time()})
    return code


def training_checkpoint(queue):
    """Validate the training/output contract, without inspecting model structure."""
    train = queue / "training"
    if read_json(train / "exit.json")["exit_code"] != 0:
        raise RuntimeError("Training failed/stopped; it will not be retrained automatically")
    text = (train / "job.log").read_text(errors="replace")
    if re.search(r"Traceback \(most recent call last\)|ChildFailedError|CUDA out of memory|OutOfMemoryError|\bloss=(?:nan|inf)\b", text):
        raise RuntimeError("Training log contains an error/non-finite loss; inspect training/job.log")
    final = re.findall(r"\[checkpoint\] Saved final: ([^\r\n]+)", text)
    if len(set(final)) != 1:
        raise RuntimeError("Missing or ambiguous Saved final checkpoint in training log")
    checkpoint = Path(final[0].strip())
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    step = re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", checkpoint.name)
    progress = re.findall(r"Training:\s*100%[^\r\n]*?\b(\d+)/(\d+)", text)
    if not step or (step[1], step[1]) not in progress:
        raise RuntimeError("Final checkpoint has no matching completed N/N training progress")
    with safe_open(checkpoint, framework="np") as weights:
        if not list(weights.keys()):
            raise RuntimeError("Final checkpoint is empty")
        for key in weights.keys():
            weights.get_slice(key).get_shape()
    run = checkpoint.parent
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    encoder_name = cfg.get("model", {}).get("video_backbone", {}).get("encoder", {}).get("name")
    native_assets = {
        "dinov3": ("dinov3/config.json", "dinov3/encoder.json"),
        "ijepa": ("ijepa/encoder.json",),
        "wan21_features": ("wan21_features/encoder.json",),
        "wan_vae": ("wan_vae/encoder.json",),
    }.get(encoder_name, ("manifest.json",))
    for asset in ("config.yaml", *native_assets, "normalization_stats.npy"):
        if not (run / asset).is_file():
            raise RuntimeError(f"Missing saved deployment asset: {run / asset}")
    stat = checkpoint.stat()
    return {"run": str(run), "checkpoint": checkpoint.name,
            "config_sha256": hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest(),
            "weight_size_mtime": [stat.st_size, stat.st_mtime_ns]}


def prepare_deploy(run):
    path = run / "vla_harness_deploy.yaml"
    if not path.exists():
        cfg = yaml.safe_load((ROOT / "configs/deploy.yaml").read_text())
        cfg["optimization"]["dit_cache"]["enabled"] = False
        cfg["inference"].update(denoise_steps=10, denoise_mode="sync", inference_mode="sync", inference_horizon=10)
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
        temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_command(queue, result):
    return ["bash", str(ROOT / "scripts/eval_libero_then_plus.sh"), result["run"], result["checkpoint"],
            str(queue / "evaluation"), "--gpus", "0,1,2,3", "--workers", "32", "--max-batch-size", "8",
            "--render-gpus", "3", "--base-port", "8990", "--plus-workers", "128",
            "--plus-max-batch-size", "32", "--plus-render-gpus", "1,3", "--plus-base-port", "8990",
            "--correct-instruction-suffix", "true", "--wait-for-idle", "--stop-file", str(queue / "STOP")]


def finished_evaluation(queue):
    if not (queue / "evaluation" / "DONE").exists():
        return False
    for stage in ("libero", "libero-plus"):
        summary = read_json(queue / "evaluation" / stage / "summary.json")
        if not (summary["complete"] and summary["selected_run_complete"]):
            raise RuntimeError(f"Evaluation DONE does not match a complete {stage} summary")
    return True


def run_pipeline(queue, resource_lock, train_script=TRAIN_SCRIPT):
    train = queue / "training"
    train.mkdir(exist_ok=True)
    command = ["bash", str((ROOT / train_script).resolve())]
    if not (queue / "job.json").exists():
        atomic_json(queue / "job.json", {"repo": str(ROOT), "command": command,
                    "script_sha256": hashlib.sha256(Path(command[1]).read_bytes()).hexdigest(),
                    "train_id": uuid.uuid4().hex, "created_at": time.time()})
    job = read_json(queue / "job.json")
    if job["repo"] != str(ROOT):
        raise RuntimeError("Queue belongs to another checkout")
    if job["command"] != command:
        raise RuntimeError("Queue training entry differs; resume with the original entry point")
    if not (train / "started.json").exists():
        if not wait_resources(queue):
            return 130
        # Reserve before spawn: an interrupted launch can never silently retrain.
        atomic_json(train / "started.json", {"started_at": time.time(), "command": job["command"]})
        with (train / "worker.log").open("a") as handle:
            subprocess.Popen([sys.executable, "-u", "-m", "benchmarks.vla_harness.run_train_eval",
                              str(queue), "--training-worker"], cwd=ROOT,
                             stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                             pass_fds=(resource_lock.fileno(),))
        time.sleep(1)
    status(queue, "training")
    while not (train / "exit.json").exists():
        if stop_requested(queue):
            return 130
        launch = train / "launch.json"
        if not launch.exists() and time.time() - read_json(train / "started.json")["started_at"] < 30:
            time.sleep(1)
            continue
        if not launch.exists() or not alive(read_json(launch)["worker"]):
            raise RuntimeError("Training worker disappeared without exit.json; inspect logs, do not retrain")
        time.sleep(POLL_SECONDS)
    result = training_checkpoint(queue)
    run = Path(result["run"])
    if stop_requested(queue, run):
        return 130
    while owned_processes(job["train_id"]):
        if stop_requested(queue, run):
            return 130
        time.sleep(POLL_SECONDS)
    result["deploy_sha256"] = prepare_deploy(run)
    if (queue / "run.json").exists() and read_json(queue / "run.json") != result:
        raise RuntimeError("Saved run/config/checkpoint/deployment changed; refusing to mix results")
    atomic_json(queue / "run.json", result)
    command = evaluation_command(queue, result)
    # Evaluate using saved training config and the repository's current adapters.
    env = {key: value for key, value in os.environ.items()
           if key not in ("TEXT_EMBEDDING_DIR", "PLUS_DEPLOY_DIR", "CORRECT_INSTRUCTION_SUFFIX", OWNER_KEY)}
    env["OPENWAM_PYTHON"] = sys.executable
    while not stop_requested(queue, run):
        if finished_evaluation(queue):
            if not wait_resources(queue, run):
                return 130
            atomic_json(queue / "DONE", {"finished_at": time.time(), **result})
            status(queue, "complete", **result)
            return 0
        status(queue, "evaluating", **result)
        with (queue / "evaluation.log").open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       pass_fds=(resource_lock.fileno(),))
            atomic_json(queue / "evaluation_supervisor.json", {"command": command, "process": identity(process.pid)})
            code = process.wait()
        atomic_json(queue / "evaluation_exit.json", {"exit_code": code, "finished_at": time.time()})
        if not finished_evaluation(queue) and not stop_requested(queue, run):
            status(queue, "retrying_evaluation", exit_code=code)
            time.sleep(RETRY_SECONDS)
    return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path, help="New directory per experiment; reuse it to resume evaluations")
    parser.add_argument("--train-script", type=Path, default=Path(TRAIN_SCRIPT),
                        help="Training entry selected by the GPU2 or share_dir shell wrapper")
    parser.add_argument("--training-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    queue = args.queue_dir.resolve()
    queue.mkdir(parents=True, exist_ok=True)

    def request_stop(signum, frame):
        (queue / "STOP").touch()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.training_worker:
        return training_worker(queue)
    with (queue / "supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"Already supervised: {queue}")
            return 1
        atomic_json(queue / "supervisor.json", identity(os.getpid()))
        (ROOT / "outputs").mkdir(exist_ok=True)
        # Serialize this entry point across queue directories in this checkout.
        with (ROOT / "outputs/train_eval.gpu.lock").open("w") as resource_lock:
            while not stop_requested(queue):
                try:
                    fcntl.flock(resource_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status(queue, "waiting_for_other_pipeline")
                    time.sleep(RETRY_SECONDS)
            else:
                return 130
            try:
                code = run_pipeline(queue, resource_lock, args.train_script)
                if code != 0:
                    status(queue, "stopped", exit_code=code)
                return code
            except Exception as exc:
                status(queue, "error", message=str(exc))
                raise


if __name__ == "__main__":
    sys.exit(main())
