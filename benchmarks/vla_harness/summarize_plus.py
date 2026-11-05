"""Official Plus perturbation-category aggregation with harness integrity gates."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.vla_harness.libero_plus_protocol import scheduler


def summarize_plus(root: Path):
    manifest = json.loads((root / "run.json").read_text())
    plan = manifest["plan"]
    catalog_file = root / "task_catalog.json"
    if hashlib.sha256(catalog_file.read_bytes()).hexdigest() != manifest["task_catalog_sha256"]:
        raise ValueError("Saved Plus catalog changed after launch")
    catalog = json.loads(catalog_file.read_text())
    canonical = {(t["suite"], t["task_id"]): t for t in catalog}
    expected = {(t["suite"], t["task_id"], 0) for t in plan["tasks"]}
    if plan["num_trials"] != 1 or len(expected) != len(plan["tasks"]):
        raise ValueError("Plus requires unique tasks with exactly one trial each")
    path = root / "OpenWAMLIBEROPlus_aggregate.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    records = [e for task in data.get("tasks", []) for e in task.get("episodes", [])]
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.get("suite"), record.get("task_id"), record.get("episode_id"))].append(record)
    valid = {}
    for key, rows in grouped.items():
        if key not in expected or len(rows) != 1:
            continue
        row = rows[0]
        if not row.get("failure_reason") and isinstance(row.get("metrics", {}).get("success"), bool):
            valid[key] = row
    selected_complete = len(records) == len(expected) and set(valid) == expected
    full = selected_complete and not plan["smoke"] and {(suite, task) for suite, task, _ in expected} == set(canonical)
    task_rows, missing = [], []
    for task in plan["tasks"]:
        suite, task_id = task["suite"], task["task_id"]
        meta = canonical[(suite, task_id)]
        row = valid.get((suite, task_id, 0))
        successes = int(row["metrics"]["success"]) if row else 0
        task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task_name": meta["task_name"],
                "category": meta["category"],
                "difficulty_level": meta["difficulty_level"],
                "successes": successes,
                "trials": int(row is not None),
                "success_rate": float(successes) if row and not plan["smoke"] else None,
                "complete": row is not None,
            }
        )
        if row is None:
            missing.append({"suite": suite, "task_id": task_id, "trial_start": 0, "trial_stop": 1})

    def aggregate(rows):
        successes = sum(row["successes"] for row in rows)
        trials = sum(row["trials"] for row in rows)
        return {
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials if trials and selected_complete and not plan["smoke"] else None,
            "tasks_complete": sum(row["complete"] for row in rows),
            "tasks_expected": len(rows),
        }

    categories = [
        {"category": name, **aggregate([t for t in task_rows if t["category"] == name])}
        for name in scheduler.PLUS_CATEGORY_ORDER
        if any(t["category"] == name for t in task_rows)
    ]
    overall = aggregate(task_rows)
    return {
        "benchmark": "OpenWAMLIBEROPlus",
        "smoke": plan["smoke"],
        "selected_run_complete": selected_complete,
        "complete": full,
        "task_results": task_rows,
        "group_by": "category",
        "category_results": categories,
        "overall": overall,
        "missing_runs": missing,
        "invalid_records": len(records) - len(valid),
        "expected_full_trials": len(canonical),
        "total_average_success_rate": overall["success_rate"] if full else None,
    }


def write_summary_plus(root: Path):
    result = summarize_plus(root)
    compact = {key: value for key, value in result.items() if key not in ("task_results", "missing_runs")}
    (root / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    rows = result["category_results"] + [{"category": "Overall", **result["overall"]}]
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# OpenWAM / harness LIBERO-plus", "", f"Smoke run: {result['smoke']}", ""]
    for key, title, name_key in [
        ("category_results", "Perturbation category", "category"),
    ]:
        lines += [f"| {title} | Successes / valid trials | Complete tasks | Success rate |", "|---|---:|---:|---:|"]
        for row in result[key]:
            score = "pending" if row["success_rate"] is None else f"{row['success_rate']:.2%}"
            lines.append(
                f"| {row[name_key]} | {row['successes']}/{row['trials']} | {row['tasks_complete']}/{row['tasks_expected']} | {score} |"
            )
        lines.append("")
    selected = result["overall"]["success_rate"]
    if selected is not None and not result["complete"]:
        lines.append(f"Selected perturbation categories success rate: {selected:.2%}")
    average = result["total_average_success_rate"]
    lines.append(
        f"Full benchmark success rate: {average:.2%}"
        if average is not None
        else "Full benchmark success rate: pending."
    )
    lines.append(
        "Overall uses total successes / total trials, as in the official scheduler; categories have unequal sizes."
    )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result
