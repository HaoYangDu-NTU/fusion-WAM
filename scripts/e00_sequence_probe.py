"""Paired FP32-router diagnostics on task 2, initial state 0 in each LIBERO suite."""
import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/vla-evaluation-harness/src")]
from benchmarks.vla_harness.run_eval import launch, make_plan
from benchmarks.vla_harness.summarize import SUITES
from openwam.experiments import runtime_env

POLICY = yaml.safe_load((ROOT / "benchmarks/libero/policy_config.yml").read_text())
CHUNK = 10
SERVER = Path(__file__).with_name("e00_sequence_probe_server.py")


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def plan(checkpoint, directory, actions_only):
    # Match the November mechanism probe. The benchmark owns the per-suite
    # native horizons, so a common outer cap of their maximum permits each one.
    cap = CHUNK + 1 if actions_only else max(POLICY["max_steps_by_suite"].values())
    args = argparse.Namespace(
        benchmark="libero", policy_backend="openwam", correct_instruction_suffix=False,
        ckpt_dir=checkpoint.parent, ckpt_name=checkpoint.name, gpus="0", workers=1,
        max_batch_size=1, max_wait_time=.01, client_action_queue=False, dynamic_tasks=False,
        suites=",".join(SUITES), categories=None, task_ids="2", exclude_tasks_file=None,
        render="gpu", render_gpus="0", image_format="raw", libero_path=ROOT.parent / "benchmarks/LIBERO",
        libero_plus_python=None, base_port=10249, output_dir=directory / "evaluation",
        export_step_jsonl=False, smoke=True, smoke_max_steps=cap, eval_id=str(uuid.uuid4()),
    )
    config = make_plan(args)
    command = config["servers"][0]["command"]
    command[1:3] = [str(SERVER)]
    command.extend(["--args.inference_horizon", str(CHUNK)])
    return args, config


def run_case(checkpoint, output, mode, actions_only):
    base = output / mode
    base.mkdir(exist_ok=True)
    # A previous successful arm is reusable if its partner needs to be retried.
    for previous in sorted(base.glob("attempt-*"), reverse=True):
        if (previous / "DONE.json").is_file():
            return inspect_case(previous, mode)
    directory = base / f"attempt-{uuid.uuid4().hex}"
    directory.mkdir()
    args, config = plan(checkpoint, directory, actions_only)
    write_json(directory / "plan.json", config)
    os.environ.update(E00_DIAG_CASE=str(directory), E00_DIAG_ROUTER=mode)
    if launch(args, config):
        raise RuntimeError(f"{mode} failed; see {directory}")
    result = inspect_case(directory, mode)
    write_json(directory / "DONE.json", {"mode": mode})
    return result


def component_stats(on, off):
    delta = on - off
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite action differences")
    out = {}
    for name, section in (("translation", slice(0, 3)), ("rotation", slice(3, 6)),
                          ("gripper", slice(6, 7)), ("all", slice(None))):
        diff = delta[:, section]
        source = on[:, section]
        norm = np.linalg.norm(source)
        out[name] = {
            "mean_signed_by_axis": diff.mean(0).tolist(),
            "mean_absolute_by_axis": np.abs(diff).mean(0).tolist(),
            "mae": float(np.abs(diff).mean()), "rms": float(np.sqrt(np.mean(diff**2))),
            "max_abs": float(np.abs(diff).max()),
            "mean_step_l2": float(np.linalg.norm(diff, axis=-1).mean()),
            "relative_l2": float(np.linalg.norm(diff) / norm) if norm else None,
        }
    out["gripper"]["sign_disagreement_fraction"] = float(np.mean((on[:, 6] > 0) != (off[:, 6] > 0)))
    return out


def inspect_case(directory, mode):
    evaluation = directory / "evaluation"
    summary = json.loads((evaluation / "summary.json").read_text())
    loaded, = rows(directory / "loaded.jsonl")
    if not summary["selected_run_complete"] or loaded["mode"] != mode or loaded["dtype"] != "torch.float32":
        raise ValueError(f"Incomplete or incorrect {mode} run at {directory}")
    data = json.loads((evaluation / "OpenWAMLIBERO_aggregate.json").read_text())
    episodes = [ep for task in data["tasks"] for ep in task["episodes"]]
    if len(episodes) != 4 or {(e["suite"], e["task_id"], e["episode_id"]) for e in episodes} != {
        (suite, 2, 0) for suite in SUITES
    }:
        raise ValueError("Unexpected episode identities")
    result = {"gate": loaded["gate"], "gate_logit": loaded["gate_logit"], "directory": str(directory),
              "episodes": {ep["suite"]: {"steps": ep["steps"], "success": ep["metrics"]["success"]}
                           for ep in episodes}}
    if mode == "off":
        return result
    pairs = rows(directory / "paired_actions.jsonl")
    inputs = {r["call"]: r["input_sha256"] for r in rows(directory / "paired_inputs.jsonl")}
    by_id = {(e["sid"], e["eid"]): e for e in episodes}
    obs_count = {suite: 0 for suite in SUITES}
    on_chunks, off_chunks, measurements, details = [], [], [], []
    for pair in pairs:
        episode = by_id[ast.literal_eval(pair["episode"])]
        suite, call = episode["suite"], pair["call"]
        on, off = (np.array(pair[key], dtype=np.float64) for key in ("on", "off"))
        if on.shape != (CHUNK, 7) or off.shape != on.shape:
            raise ValueError("Unexpected paired chunk dimensions")
        on_chunks.append(on)
        off_chunks.append(off)
        with np.load(directory / "router_samples" / f"{call:05d}.npz") as samples:
            measurements.append({key: samples[key] for key in samples.files})
        details.append({"suite": suite, "chunk_index": obs_count[suite],
                        "input_sha256": inputs.pop(call), "components": component_stats(on, off)})
        obs_count[suite] += 1
    expected = {e["suite"]: min(2, (e["steps"] + CHUNK - 1) // CHUNK) for e in episodes}
    if inputs or obs_count != expected or any(n < 1 for n in obs_count.values()):
        raise ValueError(f"Missing paired inputs or chunks: {obs_count}, expected {expected}")
    result["paired_actions"] = {
        "coordinate_space": "native LIBERO delta xyz, rotation vector, continuous gripper",
        "difference": "router ON minus router OFF on identical prepared inputs",
        "trajectory": "first two policy observations per episode on the ON trajectory",
        "sampler_seed": 42, "chunks": len(pairs), "actions": len(pairs) * CHUNK,
        "components": component_stats(np.concatenate(on_chunks), np.concatenate(off_chunks)),
        "mean_over_observed_router_calls": {key: float(np.concatenate([m[key] for m in measurements]).mean())
                                            for key in measurements[0]},
        "observations": details,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print plan without creating directories or launching")
    cli = parser.parse_args()
    checkpoint, output = cli.checkpoint.resolve(strict=True), cli.output.resolve()
    match = re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", checkpoint.name)
    if not match:
        parser.error("expected checkpoint_step_N.safetensors")
    stat = checkpoint.stat()
    identity = {"checkpoint": str(checkpoint), "step": int(match[1]), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "actions_only": cli.actions_only}
    if cli.dry_run:
        args, config = plan(checkpoint, output / "on" / "dry-run", cli.actions_only)
        print(json.dumps({**identity, "cap": args.smoke_max_steps, "plan": config}, indent=2))
        return
    output.mkdir(exist_ok=True, parents=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "identity.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError(f"Output belongs to a different probe: {output}")
        write_json(manifest, identity)
        if (output / "result.json").is_file():
            return
        os.environ.update(runtime_env("0"))
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        on = run_case(checkpoint, output, "on", cli.actions_only)
        paired = on.pop("paired_actions")
        result = {**identity, "paired_actions": paired, "gate": on["gate"], "gate_logit": on["gate_logit"],
                  "simulator_seed": POLICY["seed"], "task_id": 2, "trial": 0,
                  "on_directory": on["directory"], "closed_loop": None}
        if not cli.actions_only:
            off = run_case(checkpoint, output, "off", False)
            result["off_directory"] = off["directory"]
            outcomes = {}
            for mode, arm in (("on", on), ("off", off)):
                episodes = arm["episodes"]
                outcomes[mode] = {"successes": sum(ep["success"] for ep in episodes.values()),
                                  "trials": len(episodes), "episodes": episodes}
            result["closed_loop"] = {"protocol": "four fixed full-horizon standard LIBERO episodes per mode",
                                     "native_caps": POLICY["max_steps_by_suite"], **outcomes}
        write_json(output / "result.json", result)


if __name__ == "__main__":
    main()
