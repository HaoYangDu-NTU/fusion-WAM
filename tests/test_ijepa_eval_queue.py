"""Guard the transition from the live training job to GPU evaluation."""

import json
import os
import sys

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from scripts import eval_after_ijepa_training as queue


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"max_steps": 4685}}))
    checkpoint = "checkpoint_step_4685.safetensors"
    save_file({"weight": np.ones(1, dtype=np.float32)}, run / checkpoint)
    state = run / "accel_state_step_4685"
    state.mkdir()
    (state / "trainer_state.json").write_text(json.dumps({"global_step": 4685}))
    log = tmp_path / "training.log"
    log.write_text(f"Training: 100%| 4685/4685\n[checkpoint] Saved final: {run / checkpoint}\n")
    return run, log, checkpoint


def test_completed_training_has_final_weights_and_resume_marker(completed_run):
    run, log, checkpoint = completed_run
    assert queue.completed_checkpoint(run, log) == checkpoint


def test_early_stop_checkpoint_is_not_the_requested_final_run(completed_run):
    run, log, _ = completed_run
    log.write_text("Training: 90%| 4200/4685\n[checkpoint] Saved final: checkpoint_step_4200.safetensors\n")
    with pytest.raises(RuntimeError, match="without Saved final"):
        queue.completed_checkpoint(run, log)


def test_waits_for_optimizer_state_to_finish_writing(completed_run):
    run, log, _ = completed_run
    (run / "accel_state_step_4685/trainer_state.json").unlink()
    with pytest.raises(FileNotFoundError):
        queue.completed_checkpoint(run, log)


def test_failed_training_does_not_start_evaluation(completed_run):
    run, log, _ = completed_run
    log.write_text(log.read_text() + "Traceback (most recent call last)\n")
    with pytest.raises(RuntimeError, match="contains a failure"):
        queue.completed_checkpoint(run, log)


def test_live_training_never_touches_gpu_resources(completed_run, tmp_path, monkeypatch):
    run, log, _ = completed_run
    output = tmp_path / "queue"
    monkeypatch.setattr(sys, "argv", [
        "waiter", "--run-dir", str(run), "--training-log", str(log),
        "--output-dir", str(output), "--plus-cache", str(tmp_path / "cache"),
        "--training-pids", str(os.getpid()),
    ])

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation or GPU resource probing ran while training was alive")

    monkeypatch.setattr(queue, "resources_idle", forbidden)
    monkeypatch.setattr(queue, "preflight", forbidden)
    monkeypatch.setattr(queue.time, "sleep", lambda _: (output / "STOP").touch())
    assert queue.main() == 130
    assert json.loads((output / "training.json").read_text())["processes"][0]["pid"] == os.getpid()
