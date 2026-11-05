"""Run LIBERO then Plus, retrying unfinished tasks without changing evaluation code."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.libero_plus_instructions import parse_bool
from benchmarks.vla_harness.libero_plus_protocol import task_catalog
from benchmarks.vla_harness.summarize import SUITES, write_summary
from benchmarks.vla_harness.fm_policy import add_policy_arguments, fm_arguments

ROOT = Path(__file__).resolve().parents[2]
OWNER_KEY = "OPENWAM_RESUME_JOB"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "starttime": stat[19], "state": stat[0],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except FileNotFoundError:
        return None


def owned_processes(job_id):
    marker = f"{OWNER_KEY}={job_id}".encode()
    found = []
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            if marker in path.read_bytes().split(b"\0"):
                found.append(int(path.parent.name))
        except (OSError, ProcessLookupError):
            pass
    return found


def stop_owned_processes(job_id):
    # Each subprocess inherits this job's UUID, including the independent
    # process groups created by run_eval. Never kill by GPU or command name.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned_processes(job_id):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while owned_processes(job_id) and time.monotonic() < deadline:
            time.sleep(0.2)


def read_attempt(attempt, benchmark):
    manifest_path = attempt / "run.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    db = Path(manifest["recording_db"]["output_path"])
    if not db.exists():
        name = "OpenWAMLIBEROPlus" if benchmark == "libero-plus" else "OpenWAMLIBERO"
        aggregate = attempt / f"{name}_aggregate.json"
        data = json.loads(aggregate.read_text()) if aggregate.exists() else {}
        return [e for task in data.get("tasks", []) for e in task.get("episodes", [])]
    with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=5)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT sid, eid, episode_id, context, metrics, steps, elapsed_sec, status, failure_reason "
            "FROM episode_results"
        ).fetchall()
    return [
        {**json.loads(row["context"]), "sid": row["sid"], "eid": row["eid"],
         "episode_id": row["episode_id"], "metrics": json.loads(row["metrics"]),
         "steps": row["steps"], "elapsed_sec": row["elapsed_sec"],
         "status": row["status"], "failure_reason": row["failure_reason"]}
        for row in rows
    ]


def completed_tasks(records, tasks, trials):
    expected = {(task["suite"], task["task_id"]) for task in tasks}
    grouped = defaultdict(list)
    for row in records:
        key = (row.get("suite"), row.get("task_id"))
        if key in expected:
            grouped[key].append(row)
    return {
        key: sorted(rows, key=lambda row: row["episode_id"])
        for key, rows in grouped.items()
        if len(rows) == trials
        and {row.get("episode_id") for row in rows} == set(range(trials))
        and all(row.get("status") in ("success", "fail") and not row.get("failure_reason")
                and isinstance(row.get("metrics", {}).get("success"), bool) for row in rows)
    }


def collect(stage, attempt, benchmark, tasks, trials, completed):
    try:
        found = completed_tasks(read_attempt(attempt, benchmark), tasks, trials)
    except (OSError, ValueError, sqlite3.Error) as exc:
        log(f"Cannot read {attempt.name} yet: {exc}")
        return
    for key, episodes in found.items():
        if key in completed:
            continue
        entry = {"suite": key[0], "task_id": key[1], "source_attempt": str(attempt), "episodes": episodes}
        atomic_json(stage / "completed" / f"{key[0]}__{key[1]:05d}.json", entry)
        completed[key] = entry
        log(f"Saved {benchmark} {key[0]}/{key[1]} ({len(completed)}/{len(tasks)} tasks)")


def summarize_stage(stage, benchmark, completed):
    name = "OpenWAMLIBEROPlus" if benchmark == "libero-plus" else "OpenWAMLIBERO"
    atomic_json(stage / f"{name}_aggregate.json", {
        "tasks": [{"task": f"{suite}/{task}", "episodes": entry["episodes"],
                   "source_attempt": entry["source_attempt"]}
                  for (suite, task), entry in sorted(completed.items())]
    })
    return write_summary(stage)


def stopped(args, stage=None):
    paths = [args.output_dir / "STOP", args.ckpt_dir / "STOP"]
    if stage is not None:
        paths.append(stage / "STOP")
    if getattr(args, "stop_file", None):
        paths.append(args.stop_file)
    return any(path.exists() for path in paths)


def resources_idle(gpus, ports):
    """Wait for other work; never reclaim GPUs by killing unrelated processes."""
    common = ["nvidia-smi", "--id=" + gpus, "--format=csv,noheader,nounits"]
    if subprocess.check_output(common + ["--query-compute-apps=pid"], text=True, timeout=30).strip():
        return False
    memory = subprocess.check_output(common + ["--query-gpu=memory.used"], text=True, timeout=30)
    if len(memory.splitlines()) != len(gpus.split(",")) or any(int(line.strip()) >= 100 for line in memory.splitlines()):
        return False
    for port in ports:
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def stage_options(args, benchmark):
    plus = benchmark == "libero-plus"
    values = {}
    for name in ("workers", "max_batch_size", "render_gpus"):
        override = getattr(args, "plus_" + name, None) if plus else None
        values[name] = override if override is not None else getattr(args, name)
    default_port = args.base_port + (len(args.gpus.split(",")) if plus else 0)
    values["base_port"] = (getattr(args, "plus_base_port", None) if plus else None) or default_port
    return values


def run_stage(args, job_id, benchmark):
    stage = args.output_dir / benchmark
    (stage / "attempts").mkdir(parents=True, exist_ok=True)
    tasks = ([{"suite": suite, "task_id": task} for suite in SUITES for task in range(10)]
             if benchmark == "libero" else task_catalog(ROOT.parent / "benchmarks/LIBERO-plus"))
    trials = 50 if benchmark == "libero" else 1
    options = stage_options(args, benchmark)
    manifest = {
        "job_id": job_id, "checkpoint_dir": str(args.ckpt_dir), "checkpoint_name": args.ckpt_name,
        "plan": {"benchmark": benchmark, "tasks": tasks, "num_trials": trials, "smoke": False},
    }
    if getattr(args, "policy_backend", "openwam") == "fm-vjepa":
        manifest["plan"]["policy_backend"] = "fm-vjepa"
    if getattr(args, "policy_backend", "openwam") == "vpp2":
        manifest["plan"]["policy_backend"] = "vpp2"
    if benchmark == "libero-plus":
        if getattr(args, "dynamic_tasks", False):
            manifest["plan"]["dynamic_tasks"] = True
        manifest["plan"]["correct_instruction_suffix"] = getattr(args, "correct_instruction_suffix", True)
        atomic_json(stage / "task_catalog.json", tasks)
        manifest["task_catalog_sha256"] = hashlib.sha256((stage / "task_catalog.json").read_bytes()).hexdigest()
    atomic_json(stage / "run.json", manifest)
    completed = {}
    for path in (stage / "completed").glob("*.json"):
        entry = json.loads(path.read_text())
        key = (entry["suite"], entry["task_id"])
        if key not in completed_tasks(entry["episodes"], tasks, trials):
            raise ValueError(f"Invalid saved completion record: {path}")
        completed[key] = entry
    # Recover committed results even if the prior launcher died before merging.
    for attempt in sorted((stage / "attempts").glob("*/run.json")):
        collect(stage, attempt.parent, benchmark, tasks, trials, completed)
    summarize_stage(stage, benchmark, completed)
    while len(completed) < len(tasks) and not stopped(args, stage):
        if getattr(args, "wait_for_idle", False):
            ports = range(options["base_port"], options["base_port"] + len(args.gpus.split(",")))
            while not stopped(args, stage) and not resources_idle(args.gpus, ports):
                atomic_json(args.output_dir / "status.json", {"benchmark": benchmark, "state": "waiting_for_resources"})
                time.sleep(args.retry_delay)
            if stopped(args, stage):
                break
        excluded = stage / "excluded_tasks.json"
        atomic_json(excluded, sorted(completed))
        attempt = stage / "attempts" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        command = [
            "bash", str(ROOT / "scripts/eval_libero_vjepa.sh"), str(args.ckpt_dir), args.ckpt_name,
            "--benchmark", benchmark, "--exclude-tasks-file", str(excluded), "--output-dir", str(attempt),
            "--gpus", args.gpus, "--workers", str(options["workers"]), "--max-batch-size", str(options["max_batch_size"]),
            "--render", "gpu", "--render-gpus", options["render_gpus"], "--image-format", "raw",
            "--base-port", str(options["base_port"]),
        ]
        if getattr(args, "client_action_queue", False):
            command.append("--client-action-queue")
        if benchmark == "libero-plus":
            command.extend(["--correct-instruction-suffix", str(getattr(args, "correct_instruction_suffix", True)).lower()])
            if getattr(args, "dynamic_tasks", False):
                command.append("--dynamic-tasks")
        if getattr(args, "policy_backend", "openwam") == "fm-vjepa":
            command.extend(fm_arguments(args))
        if getattr(args, "policy_backend", "openwam") == "vpp2":
            command.extend(["--policy-backend", "vpp2"])
        log(f"Starting {benchmark}: {len(tasks) - len(completed)} unfinished tasks; {attempt}")
        with attempt.with_suffix(".log").open("w") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env={**os.environ, OWNER_KEY: job_id, "OPENWAM_PYTHON": sys.executable,
                                            "PYTHONUNBUFFERED": "1"})
            atomic_json(attempt.with_suffix(".launch.json"), {
                "command": command, "pid": process.pid, "job_id": job_id,
                "process": process_identity(process.pid),
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
            all_done_since = None
            try:
                while process.poll() is None:
                    collect(stage, attempt, benchmark, tasks, trials, completed)
                    atomic_json(args.output_dir / "status.json", {
                        "benchmark": benchmark, "completed_tasks": len(completed), "total_tasks": len(tasks),
                        "attempt": str(attempt), "launcher_pid": process.pid,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    if stopped(args, stage):
                        break
                    if len(completed) == len(tasks):
                        all_done_since = all_done_since or time.monotonic()
                        if time.monotonic() - all_done_since > 60:
                            break
                    time.sleep(10)
            finally:
                stop_owned_processes(job_id)
                process.wait()
                atomic_json(attempt.with_suffix(".exit.json"), {
                    "exit_code": process.returncode, "finished_at": datetime.now(timezone.utc).isoformat(),
                })
            collect(stage, attempt, benchmark, tasks, trials, completed)
            log(f"{benchmark} attempt exited {process.returncode}; completed {len(completed)}/{len(tasks)} tasks")
            summarize_stage(stage, benchmark, completed)
        if len(completed) < len(tasks) and not stopped(args, stage):
            time.sleep(args.retry_delay)
    if len(completed) != len(tasks) or stopped(args, stage):
        return False
    summary = summarize_stage(stage, benchmark, completed)
    return summary["complete"] and summary["selected_run_complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", choices=("libero", "libero-plus", "libero,libero-plus"), default="libero,libero-plus",
                        help="Select either benchmark or the original two-stage evaluation")
    add_policy_arguments(parser)
    parser.add_argument("--correct-instruction-suffix", type=parse_bool, default=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dynamic-tasks", action="store_true", help="Use shared single-trial task claims for Plus")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--client-action-queue", action="store_true",
                        help="Use worker-local action chunks for all selected model adapters")
    parser.add_argument("--render-gpus", default="3")
    parser.add_argument("--base-port", type=int, default=8940)
    parser.add_argument("--retry-delay", type=float, default=30)
    parser.add_argument("--plus-workers", type=int)
    parser.add_argument("--plus-max-batch-size", type=int)
    parser.add_argument("--plus-render-gpus")
    parser.add_argument("--plus-base-port", type=int)
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    args.ckpt_dir, args.output_dir = args.ckpt_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        if args.benchmarks == "libero,libero-plus":
            settings.pop("benchmarks")  # retain identities of existing two-stage queues
        # Older queues did not have these optional resource/stage settings.
        if not args.dynamic_tasks:
            settings.pop("dynamic_tasks", None)  # Preserve legacy resume identities.
        if not args.client_action_queue:
            settings.pop("client_action_queue", None)
        for key in ("plus_workers", "plus_max_batch_size", "plus_render_gpus", "plus_base_port", "stop_file", "wait_for_idle"):
            if not settings[key]:
                settings.pop(key)
        if not args.correct_instruction_suffix:
            # Preserve explicit-false and legacy job identities on resume.
            settings.pop("correct_instruction_suffix", None)
        if args.policy_backend == "openwam":
            # Keep existing job.json identity byte-for-byte compatible on resume.
            for key in ("policy_backend", "fm_python", "fm_encoder", "fm_text_cache"):
                settings.pop(key, None)
            config_path = args.ckpt_dir / "config.yaml"
        elif args.policy_backend == "vpp2":
            from benchmarks.vla_harness.vpp2_policy import validate_assets, SOURCE
            validate_assets(args.ckpt_dir)
            config_path = args.ckpt_dir / "config.json"
            settings["stats_sha256"] = hashlib.sha256((args.ckpt_dir / "libero/dataset_stats.json").read_bytes()).hexdigest()
            settings["source_revision"] = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
            video = (args.ckpt_dir / "libero/video_step010000.pt").stat()
            settings["video_size_mtime"] = [video.st_size, video.st_mtime_ns]
        else:
            fm_arguments(args)
            config_path = args.ckpt_dir / "resolved_args.json"
            settings["stats_sha256"] = hashlib.sha256((args.ckpt_dir / "dataset_stats.json").read_bytes()).hexdigest()
            settings["text_manifest_sha256"] = hashlib.sha256((args.fm_text_cache / "manifest.json").read_bytes()).hexdigest()
            encoder = args.fm_encoder.stat()
            settings["encoder_size_mtime"] = [encoder.st_size, encoder.st_mtime_ns]
        settings["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        weight = (args.ckpt_dir / args.ckpt_name).stat()
        settings["weight_size_mtime"] = [weight.st_size, weight.st_mtime_ns]
        job_path = args.output_dir / "job.json"
        if job_path.exists():
            job = json.loads(job_path.read_text())
            if job["settings"] != settings:
                raise ValueError("Resume settings/checkpoint changed; use the original invocation")
        else:
            job = {"job_id": uuid.uuid4().hex, "settings": settings}
            atomic_json(job_path, job)
        stop_owned_processes(job["job_id"])
        try:
            for benchmark in args.benchmarks.split(","):
                if not run_stage(args, job["job_id"], benchmark):
                    return 130 if stopped(args, args.output_dir / benchmark) else 1
            if args.wait_for_idle:
                ports = {port for benchmark in args.benchmarks.split(",")
                         for port in range(stage_options(args, benchmark)["base_port"],
                                           stage_options(args, benchmark)["base_port"] + len(args.gpus.split(",")))}
                while not stopped(args) and not resources_idle(args.gpus, ports):
                    time.sleep(args.retry_delay)
                if stopped(args):
                    return 130
        finally:
            stop_owned_processes(job["job_id"])
        atomic_json(args.output_dir / "DONE", {"finished_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(args.output_dir / "status.json", {"benchmark": "complete", "finished": True})
        log(f"Completed: {args.benchmarks}")
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
