"""Sequential, four-GPU E03 baseline/cache/patch probes. Never starts formal training."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.vla_harness.run_resumable import resources_idle


def run(command, log, env):
    with log.open("w") as stream:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")
    if code or re.search(r"Traceback \(most recent call last\)|CUDA out of memory|ChildFailedError", text):
        raise RuntimeError(f"Probe failed, inspect {log}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="reports/E03_speed_optimization/gpu")
    args = parser.parse_args()
    if args.steps <= args.warmup:
        raise ValueError("Need measured steps after warmup")
    if not resources_idle("0,1,2,3", range(8990, 8998)):
        raise RuntimeError("Four GPUs are occupied; existing training will not be stopped by this script")
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false", "OPENWAM_DISABLE_FUSED_ROPE": "1",
           "OPENWAM_DISABLE_FUSED_ATTENTION": "1", "OPENWAM_DISABLE_COMPILED_RMS_NORM": "1"}
    py = sys.executable
    run([py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
         "scripts/precompute_wan_latents.py", "--smoke-steps", str(args.steps)], output / "cache_build.log", env)
    results = {}
    metric_rows = {}
    for name in ("E03", "E03-Cache", "E03-Patch"):
        log = output / f"{name}.log"
        started = time.time()
        text = run([py, "scripts/experiment.py", "train", name, "--gpus", "0,1,2,3",
                    "--smoke-steps", str(args.steps)], log, env)
        launch = Path(re.search(r"Launch report: ([^\r\n]+)", text)[1])
        ranks = {f"rank{i}": [json.loads(line) for line in (launch / "timing" / f"rank{i}.jsonl").read_text().splitlines()]
                 for i in range(4)}
        for rank, rows in ranks.items():
            if [r["step"] for r in rows] != list(range(1, args.steps + 1)):
                raise RuntimeError(f"Incomplete smoke {name}/{rank}")
            if any(not math.isfinite(v) for row in rows for v in row["metrics"].values()):
                raise RuntimeError(f"Nonfinite smoke {name}/{rank}")
        rows = ranks["rank0"][args.warmup:]
        phases = sorted(set().union(*(r["phases"] for r in rows)))
        result = dict(launch=str(launch), wall_seconds=time.time()-started,
                      measured_steps=len(rows), warmup_steps=args.warmup,
                      step_seconds=sum(max(ranks[rank][i]["wall_ms"] for rank in ranks)
                                       for i in range(args.warmup, args.steps)) / len(rows) / 1000,
                      phases_ms={p: sum(r["phases"].get(p, {}).get("cuda_elapsed_ms", 0) for r in rows) / len(rows)
                                 for p in phases},
                      data_wait_host_ms=sum(r["phases"]["data_wait"]["host_ms"] for r in rows)/len(rows),
                      peak_allocated_gib={rank: max(r["peak_allocated_gb"] for r in all_rows) for rank, all_rows in ranks.items()},
                      shapes=rows[-1]["shapes"], last_metrics=rows[-1]["metrics"])
        results[name] = result
        metric_rows[name] = [r["metrics"] for r in ranks["rank0"]]
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({name: result}), flush=True)
    # Same initial state, epoch-0 batches, sampled noise and optimizer schedule.
    differences = [{k: abs(a[k] - b[k]) for k in a}
                   for a, b in zip(metric_rows["E03"], metric_rows["E03-Cache"])]
    results["online_cache_metric_max_abs_error"] = {k: max(d[k] for d in differences) for k in differences[0]}
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for a, b in zip(metric_rows["E03"], metric_rows["E03-Cache"]):
        for key in ("loss_total", "loss_video", "loss_action"):
            if key in a and not math.isclose(a[key], b[key], rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(f"Online/cache training losses differ: {key}")
    results["acceptance_passed"] = True
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
