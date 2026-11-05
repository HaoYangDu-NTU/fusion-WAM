"""Prepare DINO frame cache, validate it and run P1 smoke only. No formal training."""
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e01_ablation_queue import Queue, final_checkpoint, read
from benchmarks.vla_harness.run_train_eval import prepare_deploy
from benchmarks.vla_harness.run_resumable import atomic_json


def main():
    folder = ROOT / "reports/E01_P1_visual_cache_20261030"
    folder.mkdir(parents=True, exist_ok=True)
    state = Queue(folder)
    with (folder / "controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.cleanup_orphaned_stages()
        state.wait_idle()
        with (ROOT / "outputs/train_eval.gpu.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            state.resource_fd = gpu_lock.fileno()
            stages = [
                ("build", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                           "scripts/precompute_image_features.py", "--experiment", "E01", "--build",
                           "--batch-size", "8", "--loader-workers", "2", "--write-inflight", "2"]),
                ("validate", [sys.executable, "scripts/validate_image_features.py", "--experiment", "E01"]),
                ("smoke", [sys.executable, "scripts/experiment.py", "train", "E01-P1-Cache",
                           "--gpus", "0,1,2,3", "--smoke-steps", "4"]),
            ]
            logs = {}
            for name, command in stages:
                logs[name] = state.run_stage(folder / name, command)
            checkpoint = final_checkpoint(logs["smoke"])
            prepare_deploy(checkpoint.parent)
            state.run_stage(folder / "smoke_inference", [
                sys.executable, "-m", "benchmarks.vla_harness.run_eval", "--benchmark", "libero",
                "--ckpt-dir", str(checkpoint.parent), "--ckpt-name", checkpoint.name,
                "--gpus", "0", "--workers", "1", "--max-batch-size", "1", "--render-gpus", "3",
                "--suites", "libero_spatial", "--task-ids", "0", "--smoke", "--base-port", "8940",
                "--correct-instruction-suffix", "false",
            ])
            validation = read(ROOT / "cache/dinov3_libero_frames/validation.json")
            if not validation["passed"]:
                raise RuntimeError("Full cache validation failed")
            state.status("cache_and_smoke_complete", formal_training_started=False,
                         automatic_training_enabled=False, checkpoint=str(checkpoint))
            atomic_json(folder / "DONE.json", dict(validation=validation, checkpoint=str(checkpoint),
                                                   formal_training_started=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        folder = ROOT / "reports/E01_P1_visual_cache_20261030"
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "status.json", dict(state="failed", error=repr(exc), formal_training_started=False))
        raise
