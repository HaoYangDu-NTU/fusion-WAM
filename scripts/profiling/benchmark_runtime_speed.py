"""Bounded real-GPU training/cache and LIBERO transport comparisons.

Training compares the existing E01 online/cache profiles, not a RoboScope
training change. Evaluation changes only client_action_queue in ABBA order.
All artifacts are isolated from formal runs; no full training is started.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def execute(command, directory, env):
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "command.json", command)
    started = time.monotonic()
    with (directory / "console.log").open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    save(directory / "exit.json", {"exit_code": result.returncode, "wall_seconds": elapsed})
    if result.returncode:
        raise RuntimeError(f"Stage failed: {directory / 'console.log'}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--resume", action="store_true", help="Continue this benchmark's completed stages")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--base-port", type=int, default=9120)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/m29_rynn9b_dinov3/train/s42/2026-10-24_02-30-16")
    args = parser.parse_args()
    if args.train_steps <= args.warmup or min(args.batch_size, args.eval_steps, args.workers, args.max_batch_size) < 1:
        parser.error("Require positive budgets and train-steps > warmup")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from benchmarks.vla_harness.run_resumable import resources_idle
    from benchmarks.vla_harness.performance import summarize_performance
    from openwam.experiments import preflight, resolve_experiment, runtime_env
    from scripts.summarize_experiment_timing import summarize
    gpus = args.gpus.split(",")
    if not resources_idle(args.gpus, () if args.resume else range(args.base_port, args.base_port + len(gpus))):
        raise RuntimeError("Requested GPUs or ports are occupied")
    env = {**os.environ, **runtime_env(args.gpus), "PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "1"}
    for key in ("TEXT_EMBEDDING_DIR", "PLUS_DEPLOY_DIR", "OPENWAM_TIMING_DIR", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    report = {"started_at": datetime.now().isoformat(), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "training_scope": "E01 online vs existing frozen feature cache; not a RoboScope change; full 30-layer policy, smoke batch size",
              "evaluation_scope": "Same E01 final checkpoint, 40 tasks x trial0, ABBA; smoke step cap, not a formal success-rate evaluation",
              "training": [], "evaluation": []}
    if args.resume:
        previous = json.loads((out / "summary.json").read_text())
        for key, value in report["arguments"].items():
            if key != "resume" and previous["arguments"].get(key) != value:
                raise ValueError(f"Cannot resume with changed {key}")
        report = previous
    elif (out / "summary.json").exists():
        raise ValueError("Existing benchmark; use --resume or a fresh output directory")
    save(out / "summary.json", report)
    if args.stage in ("all", "train"):
        for name in ("E01", "E01-Cache"):
            if any(row["experiment"] == name for row in report["training"]):
                continue
            cfg, profile, overrides = resolve_experiment(name, batch_size=args.batch_size, smoke_steps=args.train_steps)
            cache = preflight(cfg, profile)
            directory = out / "training" / name
            train_env = {**env, "CUDA_VISIBLE_DEVICES": args.gpus,
                         "OPENWAM_TIMING_DIR": str(directory / "timing")}
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
                       str(ROOT / "scripts/train.py"), *overrides,
                       f"model.video_backbone.text_embedding_fingerprint={cache['model_fingerprint']}",
                       f"training.output_path={directory / 'runs'}", "training.save_full_states_for_resume=false"]
            print(f"START training {name}", flush=True)
            wall = execute(command, directory, train_env)
            result = summarize(directory / "timing", args.batch_size, args.warmup)
            result.update(experiment=name, process_wall_seconds=wall)
            rows = [json.loads(line) for line in (directory / "timing/rank0.jsonl").read_text().splitlines()]
            assert len(rows) == args.train_steps
            assert all(math.isfinite(value) for row in rows for value in row["metrics"].values())
            result["last_metrics"] = rows[-1]["metrics"]
            result["phases"] = {key: statistics.mean(row["phases"].get(key, {}).get("host_ms", 0) for row in rows[args.warmup:])
                                for key in rows[-1]["phases"]}
            report["training"].append(result)
            save(out / "summary.json", report)
            print(json.dumps(result), flush=True)
        a, b = report["training"]
        report["training_speedup"] = a["mean_step_seconds"] / b["mean_step_seconds"]
        save(out / "summary.json", report)
    if args.stage in ("all", "eval"):
        for number, queued in enumerate((False, True, True, False), 1):
            name = f"{number}_{'queue' if queued else 'legacy'}"
            if any(row["name"] == name for row in report["evaluation"]):
                continue
            directory = out / "evaluation" / name
            command = [sys.executable, "-m", "benchmarks.vla_harness.run_eval", "--benchmark", "libero",
                       "--ckpt-dir", str(args.checkpoint_dir.resolve()), "--ckpt-name", "checkpoint_step_10690.safetensors",
                       "--gpus", args.gpus, "--workers", str(args.workers), "--max-batch-size", str(args.max_batch_size),
                       "--max-wait-time", "0.01", "--render", "gpu", "--render-gpus", args.gpus,
                       "--image-format", "raw", "--base-port", str(args.base_port + (number - 1) * len(gpus)),
                       "--task-ids", "0,1,2,3,4,5,6,7,8,9", "--smoke", "--smoke-max-steps", str(args.eval_steps),
                       "--export-step-jsonl", "--output-dir", str(directory / "attempt")]
            if queued:
                command.append("--client-action-queue")
            print(f"START evaluation {name}", flush=True)
            wall = execute(command, directory, env)
            attempt = directory / "attempt"
            performance = summarize_performance(attempt / "performance")
            database = next(attempt.glob("recording-*.sqlite"))
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                statuses = dict(connection.execute("SELECT status, COUNT(*) FROM episode_results GROUP BY status"))
                episodes = connection.execute("SELECT task_name, steps, elapsed_sec, status FROM episode_results ORDER BY task_name").fetchall()
            assert len(episodes) == 40 and not statuses.get("error", 0), statuses
            steps = sum(row[1] for row in episodes)
            console = (directory / "console.log").read_text()
            startup = float(re.search(r"Model servers ready in ([\d.]+)s", console)[1])
            result = {"name": name, "client_action_queue": queued, "wall_seconds": wall, "model_startup_seconds": startup,
                      "control_steps": steps, "control_steps_per_second": steps / wall,
                      "summed_episode_seconds": sum(row[2] for row in episodes), "statuses": statuses,
                      "episodes": episodes, "performance": performance}
            report["evaluation"].append(result)
            save(out / "summary.json", report)
            print(json.dumps({k: v for k, v in result.items() if k not in ("performance", "episodes")}), flush=True)
        legacy = [r for r in report["evaluation"] if not r["client_action_queue"]]
        queued = [r for r in report["evaluation"] if r["client_action_queue"]]
        report["evaluation_speedup"] = statistics.mean(r["wall_seconds"] for r in legacy) / statistics.mean(r["wall_seconds"] for r in queued)
        report["evaluation_throughput_speedup"] = statistics.mean(r["control_steps_per_second"] for r in queued) / statistics.mean(r["control_steps_per_second"] for r in legacy)
    report["finished_at"] = datetime.now().isoformat()
    save(out / "summary.json", report)
    print(f"DONE {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
