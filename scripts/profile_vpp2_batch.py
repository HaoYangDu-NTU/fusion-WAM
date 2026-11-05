"""Measure real VPP2 batches and compare distinct observations to scalar inference."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/vpp2_plus/batch_profile.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/vpp2")
    args = parser.parse_args()
    import numpy as np
    import torch
    from benchmarks.vla_harness.libero_plus import OfficialPlusEnvironment
    from benchmarks.vla_harness.libero_plus_protocol import task_catalog, load_config
    from benchmarks.vla_harness.libero import make_observation
    from benchmarks.vla_harness.vpp2_policy import VPP2Policy
    os.environ.update(LIBERO_PLUS_PYTHON="/data/fcy/basics/miniconda3/envs/libero-plus/bin/python",
        LIBERO_PLUS_PATH=str(ROOT.parent / "benchmarks/LIBERO-plus"),
        PYTHONPATH=str(ROOT)+":"+str(ROOT.parent / "benchmarks/LIBERO-plus"),
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LP_NUM_THREADS="1")
    observations = []
    with tempfile.TemporaryDirectory(prefix="vpp2-batch-observations-") as temp:
        os.environ["LIBERO_CONFIG_PATH"] = temp
        os.environ["LIBERO_PLUS_CONFIG_ROOT"] = temp
        tasks = task_catalog(ROOT.parent / "benchmarks/LIBERO-plus", task_ids=[0])
        cfg = {**load_config(), "policy_backend": "vpp2"}
        for task in (tasks[0], tasks[-1]):
            env = OfficialPlusEnvironment(task)
            try:
                observations.append(make_observation(env.initial["obs"], env.initial["language"], cfg))
            finally:
                env.close()
    print("Real Plus observations captured; loading policy", flush=True)
    policy = VPP2Policy(args.weights)
    torch.cuda.synchronize()
    refs = [policy.predict(obs)["actions"] for obs in observations]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in (1, 2, 4, 8, 16, 32):
        samples = [observations[i % 2] for i in range(size)]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            start = time.monotonic()
            outputs = policy.predict_batch(samples)
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            errors = [float(np.max(np.abs(out["actions"][:, :6]-refs[i % 2][:, :6]))) for i, out in enumerate(outputs)]
            gripper = all(np.array_equal(out["actions"][:, -1], refs[i % 2][:, -1]) for i, out in enumerate(outputs))
            row = dict(batch=size, seconds=seconds, samples_per_second=size/seconds,
                peak_gib=torch.cuda.max_memory_allocated()/2**30, max_action_abs_diff=max(errors),
                same_gripper=gripper, finite=True)
        except torch.cuda.OutOfMemoryError as exc:
            row = dict(batch=size, oom=True, error=str(exc)[:300])
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2)+"\n")
        print(json.dumps(row), flush=True)
        if row.get("oom"):
            break


if __name__ == "__main__":
    main()
