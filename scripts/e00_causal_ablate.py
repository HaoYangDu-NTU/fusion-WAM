"""Compare added-router value interventions at four fixed LIBERO initial states."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import uuid

import numpy as np
import e00_sequence_probe as probe

SERVER = Path(__file__).with_name("e00_causal_ablate_server.py")
CONDITIONS = ("top32", "random32", "equal_mass")


def plan(checkpoint, directory):
    args, config = probe.plan(checkpoint, directory, actions_only=True)
    args.smoke_max_steps = 1
    config["servers"][0]["command"][1] = str(SERVER)
    return args, config


def effect(baseline, altered, calls):
    return {
        "components": probe.component_stats(baseline, altered),
        "mean_removed_attention_mass": sum(c["removed_mass_sum"] for c in calls) / sum(c["rows"] for c in calls),
        "mean_value_mix_relative_l2": float(np.mean([c["value_mix_relative_l2"] for c in calls])),
        "router_calls": len(calls), "head_query_rows": sum(c["rows"] for c in calls),
    }


def summarize(directory):
    evaluation = directory / "evaluation"
    if not json.loads((evaluation / "summary.json").read_text())["selected_run_complete"]:
        raise ValueError("Incomplete four-state run")
    data = json.loads((evaluation / "OpenWAMLIBERO_aggregate.json").read_text())
    eps = [e for t in data["tasks"] for e in t["episodes"]]
    if len(eps) != 4 or {(e["suite"], e["task_id"], e["episode_id"], e["steps"]) for e in eps} != {
        (s, 2, 0, 1) for s in probe.SUITES
    }:
        raise ValueError("Unexpected initial-state identities or probe length")
    by_id = {(ep["sid"], ep["eid"]): ep["suite"] for ep in eps}
    baselines, per_task = [], {}
    chunks = {mode: [] for mode in CONDITIONS}
    calls = {mode: [] for mode in CONDITIONS}
    for row in probe.rows(directory / "observations.jsonl"):
        suite = by_id.pop((row["sid"], row["eid"]))
        runs = row["conditions"]
        baseline = np.asarray(runs["baseline"]["actions"], dtype=np.float64)
        if baseline.shape != (probe.CHUNK, 7):
            raise ValueError("Unexpected LIBERO action chunk")
        baselines.append(baseline)
        per_task[suite] = {"input_sha256": row["input_sha256"], "conditions": {}}
        for mode in CONDITIONS:
            altered = np.asarray(runs[mode]["actions"], dtype=np.float64)
            if altered.shape != baseline.shape:
                raise ValueError("Inconsistent intervention action horizon")
            local_calls = runs[mode]["router_calls"]
            chunks[mode].append(altered)
            calls[mode].extend(local_calls)
            per_task[suite]["conditions"][mode] = effect(baseline, altered, local_calls)
    if by_id:
        raise ValueError("Missing intervention outputs")
    baseline = np.concatenate(baselines)
    loaded, = probe.rows(directory / "loaded.jsonl")
    return {
        **loaded, "observations": len(per_task), "actions": len(baseline),
        "task_id": 2, "trial": 0, "simulator_seed": probe.POLICY["seed"], "sampler_seed": 42,
        "random_mask_seed": "17011 + zero-based denoising call index",
        "intervention": "Post-softmax added-router value weights; no renormalization. "
                        "For every head/query/denoising call, top32 zeros the highest-weight 32 tokens, "
                        "random32 zeros 32 tokens selected independently of weight, and equal_mass scales "
                        "all weights uniformly by one minus the top32 branch's removed mass for the same row.",
        "action_effect": "baseline minus intervention in native LIBERO actions. Relative L2 uses the baseline norm.",
        "interpretation": "The top32 and random32 interventions remove equal token counts but can remove different "
                          "weight mass. Equal-mass attenuation controls total weight per head/query. "
                          "Value vectors, projected residual amplitudes, and subsequent denoising states can still "
                          "differ. These four initial states measure action sensitivity; task success is unmeasured.",
        "conditions": {mode: effect(baseline, np.concatenate(chunks[mode]), calls[mode]) for mode in CONDITIONS},
        "per_task": per_task, "raw_directory": str(directory),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    checkpoint, output = cli.checkpoint.resolve(strict=True), cli.output.resolve()
    match = re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", checkpoint.name)
    if not match:
        parser.error("expected checkpoint_step_N.safetensors")
    stat = checkpoint.stat()
    identity = {"checkpoint": str(checkpoint), "step": int(match[1]), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
    if cli.dry_run:
        args, config = plan(checkpoint, output / "dry-run")
        print(json.dumps({**identity, "cap": args.smoke_max_steps, "conditions": CONDITIONS, "plan": config},
                         indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "identity.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError(f"Output already belongs to a different checkpoint: {output}")
        probe.write_json(manifest, identity)
        if (output / "result.json").exists():
            return
        directory = output / f"attempt-{uuid.uuid4().hex}"
        directory.mkdir()
        args, config = plan(checkpoint, directory)
        probe.write_json(directory / "plan.json", config)
        os.environ.update(probe.runtime_env("0"), E00_CAUSAL_CASE=str(directory))
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        if probe.launch(args, config):
            raise RuntimeError(f"Causal probe failed: {directory}")
        probe.write_json(output / "result.json", {**identity, **summarize(directory)})


if __name__ == "__main__":
    main()
