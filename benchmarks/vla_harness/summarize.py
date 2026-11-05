"""Summarize harness recordings without treating missing/duplicate/error trials as scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def summarize(root: Path):
    manifest = json.loads((root / "run.json").read_text())
    if manifest["plan"].get("benchmark") == "libero-plus":
        from benchmarks.vla_harness.summarize_plus import summarize_plus

        return summarize_plus(root)
    plan = manifest["plan"]
    path = root / "OpenWAMLIBERO_aggregate.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    records = [e for t in data.get("tasks", []) for e in t.get("episodes", [])]
    valid = [
        e for e in records if not e.get("failure_reason") and isinstance(e.get("metrics", {}).get("success"), bool)
    ]
    keys = [(e.get("suite"), e.get("task_id"), e.get("episode_id")) for e in valid]
    expected = {(t["suite"], t["task_id"], ep) for t in plan["tasks"] for ep in range(plan["num_trials"])}
    selected_complete = len(valid) == len(records) == len(expected) and set(keys) == expected
    rows = []
    for suite in SUITES:
        eps = [e for e in valid if e.get("suite") == suite]
        received = [e for e in records if e.get("suite") == suite]
        complete = (
            not plan["smoke"]
            and len(eps) == len(received) == 500
            and {(e.get("task_id"), e.get("episode_id")) for e in eps}
            == {(task, ep) for task in range(10) for ep in range(50)}
        )
        successes = sum(e["metrics"]["success"] for e in eps)
        rows.append(
            {
                "suite": suite,
                "valid_trials": len(eps),
                "expected_trials": 500,
                "successes": successes,
                "errors": len(received) - len(eps),
                "complete": complete,
                "success_rate": successes / 500 if complete else None,
            }
        )
    complete = selected_complete and all(row["complete"] for row in rows)
    return {
        "smoke": plan["smoke"],
        "selected_run_complete": selected_complete,
        "suites": rows,
        "complete": complete,
        "total_average_success_rate": sum(r["successes"] for r in rows) / 2000 if complete else None,
    }


def write_summary(root: Path):
    manifest = json.loads((root / "run.json").read_text())
    if manifest["plan"].get("benchmark") == "libero-plus":
        from benchmarks.vla_harness.summarize_plus import write_summary_plus

        return write_summary_plus(root)
    result = summarize(root)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "suite_success_rates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["suites"][0]))
        writer.writeheader()
        writer.writerows(result["suites"])
    lines = [
        "# OpenWAM / harness LIBERO",
        "",
        f"Smoke run: {result['smoke']}",
        "",
        "| Suite | Valid trials | Errors | Success rate |",
        "|---|---:|---:|---:|",
    ]
    for row in result["suites"]:
        score = f"{row['success_rate']:.2%}" if row["complete"] else "pending"
        lines.append(f"| {row['suite']} | {row['valid_trials']}/500 | {row['errors']} | {score} |")
    average = result["total_average_success_rate"]
    lines += [
        "",
        f"Total average: {average:.2%}"
        if average is not None
        else "Total average: pending (requires 2,000 valid trials).",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_summary(args.output_dir.resolve()), indent=2))
