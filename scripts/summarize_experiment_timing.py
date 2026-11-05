#!/usr/bin/env python3
"""Summarize a smoke's rank*.jsonl; exclude warmup and profiler steps."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory, batch_size, warmup):
    ranks = {}
    for path in sorted(Path(directory).glob("rank*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        steps = [row["step"] for row in rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"Repeated steps in {path}; use one fresh directory per run")
        ranks[path.stem] = {row["step"]: row for row in rows}
    if not ranks:
        raise ValueError("No rank*.jsonl found")
    common = set.intersection(*(set(rows) for rows in ranks.values()))
    selected = sorted(step for step in common if step > warmup and not any(
        rows[step]["profiled"] for rows in ranks.values()
    ))
    if not selected:
        raise ValueError("No measured steps after warmup")
    seconds = [max(rows[step]["wall_ms"] for rows in ranks.values()) / 1000 for step in selected]
    return {
        "directory": str(Path(directory).resolve()), "ranks": len(ranks),
        "batch_size_per_rank": batch_size, "warmup_steps_excluded": warmup,
        "measured_steps": len(selected), "step_range": [selected[0], selected[-1]],
        "mean_step_seconds": statistics.mean(seconds),
        "median_step_seconds": statistics.median(seconds),
        "samples_per_second": batch_size * len(ranks) * len(selected) / sum(seconds),
        "peak_allocated_gib_per_rank": {
            rank: max(row["peak_allocated_gb"] for row in rows.values())
            for rank, rows in ranks.items()
        },
        "scope": "StepTimer measured steps; checkpoint IO and end-to-end GPU hours are separate",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=64, help="Actual per-rank batch size")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0:
        parser.error("batch-size must be positive and warmup nonnegative")
    try:
        print(json.dumps(summarize(args.directory, args.batch_size, args.warmup), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
