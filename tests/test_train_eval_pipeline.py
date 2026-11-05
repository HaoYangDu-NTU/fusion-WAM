"""The remote queue must not depend on a particular model or train twice."""

import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from benchmarks.vla_harness import run_train_eval as pipeline


def saved_training(tmp_path, step=7):
    queue, run = tmp_path / "queue", tmp_path / "arbitrary_model"
    train = queue / "training"
    train.mkdir(parents=True)
    run.mkdir()
    checkpoint = run / f"checkpoint_step_{step}.safetensors"
    save_file({"some_future_architecture.weight": np.zeros((3, 11), dtype=np.float32)}, checkpoint)
    (run / "config.yaml").write_text("model: {unknown_future_architecture: true}\n")
    (run / "manifest.json").write_text("{}")
    np.save(run / "normalization_stats.npy", np.zeros(1))
    pipeline.atomic_json(train / "exit.json", {"exit_code": 0})
    (train / "job.log").write_text(f"Training: 100%|bar| {step}/{step} [loss=0.1]\n[checkpoint] Saved final: {checkpoint}\n")
    return queue, run


@pytest.mark.parametrize("step", [7, 12483])
def test_training_detection_is_independent_of_architecture_and_step(tmp_path, step):
    queue, run = saved_training(tmp_path, step)
    result = pipeline.training_checkpoint(queue)
    assert result["run"] == str(run)
    assert result["checkpoint"] == f"checkpoint_step_{step}.safetensors"


@pytest.mark.parametrize("failure", ["exit", "missing_final", "incomplete", "traceback", "nan", "truncated", "missing_config"])
def test_training_requires_success_and_final_checkpoint_evidence(tmp_path, failure):
    queue, run = saved_training(tmp_path)
    log = queue / "training/job.log"
    if failure == "exit":
        pipeline.atomic_json(queue / "training/exit.json", {"exit_code": 1})
    elif failure == "missing_final":
        log.write_text("Training: 100%|bar| 7/7\n")
    elif failure == "incomplete":
        log.write_text(log.read_text().replace("7/7", "6/7"))
    elif failure == "traceback":
        log.write_text(log.read_text() + "Traceback (most recent call last):\n")
    elif failure == "nan":
        log.write_text(log.read_text().replace("loss=0.1", "loss=nan"))
    elif failure == "truncated":
        (run / "checkpoint_step_7.safetensors").write_bytes(b"broken")
    else:
        (run / "config.yaml").unlink()
    with pytest.raises(Exception):
        pipeline.training_checkpoint(queue)


def test_process_identity_rejects_reused_pid_and_zombie(monkeypatch):
    old = {"pid": 123, "starttime": "10"}
    monkeypatch.setattr(pipeline, "identity", lambda _: {"pid": 123, "starttime": "11", "state": "S"})
    assert not pipeline.alive(old)
    monkeypatch.setattr(pipeline, "identity", lambda _: {**old, "state": "Z"})
    assert not pipeline.alive(old)
    monkeypatch.setattr(pipeline, "identity", lambda _: {**old, "state": "S"})
    assert pipeline.alive(old)


def test_disappeared_training_worker_is_never_retrained(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    (queue / "training").mkdir(parents=True)
    pipeline.atomic_json(queue / "job.json", {"repo": str(pipeline.ROOT),
                         "command": ["bash", str(pipeline.ROOT / pipeline.TRAIN_SCRIPT)]})
    pipeline.atomic_json(queue / "training/started.json", {"started_at": 0})
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not retrain"))
    with (tmp_path / "lock").open("w") as lock, pytest.raises(RuntimeError, match="without exit.json"):
        pipeline.run_pipeline(queue, lock)


def test_pipeline_trains_once_retries_evaluation_and_resumes_same_run(tmp_path, monkeypatch):
    queue, run = saved_training(tmp_path)
    training_text = (queue / "training/job.log").read_text()
    (queue / "training/exit.json").unlink()
    (tmp_path / "scripts").mkdir()
    train_script = tmp_path / pipeline.TRAIN_SCRIPT
    train_script.write_text("#!/bin/bash\nprintf '%s' " + "'" + training_text + "'\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/deploy.yaml").write_text("optimization: {dit_cache: {enabled: true}}\ninference: {}\n")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "wait_resources", lambda *a: True)
    monkeypatch.setattr(pipeline, "owned_processes", lambda _: [])
    monkeypatch.setattr(pipeline, "stop_owned_processes", lambda _: None)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    monkeypatch.setenv("TEXT_EMBEDDING_DIR", "wrong-cache")
    monkeypatch.setenv("PLUS_DEPLOY_DIR", "wrong-deployment")
    monkeypatch.setenv("CORRECT_INSTRUCTION_SUFFIX", "false")
    real_popen = subprocess.Popen
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        if "--training-worker" in command:
            assert pipeline.training_worker(queue) == 0
        elif command == ["bash", str(train_script)]:
            child = real_popen(command, **kwargs)
            child.wait(timeout=5)
            return child
        else:
            assert command[1] == str(tmp_path / "scripts/eval_libero_then_plus.sh")
            assert command[2:4] == [str(run), "checkpoint_step_7.safetensors"]
            assert not {"TEXT_EMBEDDING_DIR", "PLUS_DEPLOY_DIR", "CORRECT_INSTRUCTION_SUFFIX"} & kwargs["env"].keys()
            assert command[command.index("--plus-workers") + 1] == "128"
            assert command[command.index("--correct-instruction-suffix") + 1] == "true"
            # First supervisor dies; the second continues the same output.
            count = sum("eval_libero_then_plus.sh" in c[1] for c in calls)
            if count == 2:
                for stage in ("libero", "libero-plus"):
                    pipeline.atomic_json(queue / "evaluation" / stage / "summary.json",
                                         {"complete": True, "selected_run_complete": True})
                (queue / "evaluation/DONE").touch()
        return SimpleNamespace(pid=os.getpid(), wait=lambda: 137)

    monkeypatch.setattr(pipeline.subprocess, "Popen", launch)
    with (tmp_path / "lock").open("w") as lock:
        assert pipeline.run_pipeline(queue, lock) == 0
        assert (queue / "DONE").exists()
        assert len(calls) == 4  # worker, actual training, two eval supervisors
        assert pipeline.run_pipeline(queue, lock) == 0
        assert len(calls) == 4
    cfg = pipeline.yaml.safe_load((run / "vla_harness_deploy.yaml").read_text())
    assert not cfg["optimization"]["dit_cache"]["enabled"]
    assert cfg["inference"]["denoise_steps"] == 10
    assert pipeline.read_json(queue / "job.json")["script_sha256"] == hashlib.sha256(train_script.read_bytes()).hexdigest()


def test_done_requires_both_full_benchmarks(tmp_path):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/DONE").touch()
    pipeline.atomic_json(tmp_path / "evaluation/libero/summary.json", {"complete": True, "selected_run_complete": True})
    pipeline.atomic_json(tmp_path / "evaluation/libero-plus/summary.json", {"complete": False, "selected_run_complete": True})
    with pytest.raises(RuntimeError, match="libero-plus"):
        pipeline.finished_evaluation(tmp_path)


@pytest.mark.parametrize("original,other", [
    ("scripts/train_libero_vjepa_1_3b_dev.sh", pipeline.TRAIN_SCRIPT),
    (pipeline.TRAIN_SCRIPT, "scripts/train_libero_vjepa_1_3b_dev.sh"),
])
def test_resume_cannot_switch_training_entry(tmp_path, monkeypatch, original, other):
    queue = tmp_path / "queue"
    queue.mkdir()
    pipeline.atomic_json(queue / "job.json", {"repo": str(pipeline.ROOT),
                         "command": ["bash", str(pipeline.ROOT / original)]})
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with (tmp_path / "lock").open("w") as lock, pytest.raises(RuntimeError, match="training entry differs"):
        pipeline.run_pipeline(queue, lock, other)
    assert not (queue / "training/started.json").exists()


def test_stop_before_launch_does_not_create_training_marker(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "STOP").touch()
    pipeline.atomic_json(queue / "job.json", {"repo": str(pipeline.ROOT),
                         "command": ["bash", str(pipeline.ROOT / pipeline.TRAIN_SCRIPT)]})
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with (tmp_path / "lock").open("w") as lock:
        assert pipeline.run_pipeline(queue, lock) == 130
    assert not (queue / "training/started.json").exists()


def test_resource_wait_does_not_kill_other_experiments(tmp_path, monkeypatch):
    checks = iter([False, True])
    monkeypatch.setattr(pipeline, "resources_idle", lambda *a: next(checks))
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    monkeypatch.setattr(pipeline, "stop_owned_processes", lambda _: pytest.fail("no cleanup while waiting"))
    assert pipeline.wait_resources(tmp_path)


@pytest.mark.parametrize("entry_name,train_script", [
    ("scripts/train_eval_libero_share_dir.sh", pipeline.TRAIN_SCRIPT),
    ("scripts/train_eval_libero_share_dir.sh", "scripts/train_libero_vjepa_5b_dev_share_dir.sh"),
    ("scripts/train_eval_libero.sh", "scripts/train_libero_vjepa_1_3b_dev.sh"),
])
def test_real_supervisor_restart_keeps_training_once_and_retries_eval(tmp_path, entry_name, train_script):
    """Real Linux processes/locks, CPU fixtures only; no model or GPU is loaded."""
    import benchmarks
    import benchmarks.vla_harness as harness

    queue, run = saved_training(tmp_path)
    final_log = (queue / "training/job.log").read_text()
    (queue / "training/exit.json").unlink()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/deploy.yaml").write_text("optimization: {dit_cache: {enabled: true}}\ninference: {}\n")
    package = tmp_path / "benchmarks/vla_harness"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text(f"__path__.extend({list(benchmarks.__path__)!r})\n")
    (package / "__init__.py").write_text(f"__path__.extend({list(harness.__path__)!r})\n")
    ports = []
    for _ in range(4):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            ports.append(probe.getsockname()[1])
    for name in ("run_train_eval.py", "run_resumable.py"):
        code = (Path(pipeline.__file__).parent / name).read_text()
        (package / name).write_text(code.replace("POLL_SECONDS = 10", "POLL_SECONDS = 0.1")
                                       .replace("RETRY_SECONDS = 30", "RETRY_SECONDS = 0.1")
                                       .replace("range(8990, 8994)", repr(tuple(ports))))
    entry = tmp_path / entry_name
    entry.write_text((pipeline.ROOT / entry_name).read_text())
    python = shlex.quote(sys.executable)
    (tmp_path / train_script).write_text(
        f"#!/bin/bash\n{python} - <<'PY'\nfrom pathlib import Path\nimport time\n"
        f"with Path({str(tmp_path / 'train_count')!r}).open('a') as f: f.write('train\\n')\n"
        f"time.sleep(1)\nprint({final_log!r})\nPY\n"
    )
    (tmp_path / "scripts/eval_libero_then_plus.sh").write_text(
        f"#!/bin/bash\n{python} - <<'PY'\nfrom pathlib import Path\nimport json\n"
        f"q=Path({str(queue)!r})\np=q/'eval_count'\nn=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\n"
        "if n==1: raise SystemExit(137)\n"
        "for stage in ('libero','libero-plus'):\n"
        "    d=q/'evaluation'/stage; d.mkdir(parents=True,exist_ok=True)\n"
        "    (d/'summary.json').write_text(json.dumps(dict(complete=True,selected_run_complete=True)))\n"
        "(q/'evaluation/DONE').touch()\nPY\n"
    )
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text('#!/bin/bash\nif [[ "$*" == *query-gpu* ]]; then printf "10\\n10\\n10\\n10\\n"; fi\n')
    fake_smi.chmod(0o755)
    env = {**os.environ, "OPENWAM_PYTHON": sys.executable, "PATH": str(tmp_path) + ":" + os.environ["PATH"]}
    env.pop("OPENWAM_TRAIN_SCRIPT", None)
    if train_script == "scripts/train_libero_vjepa_5b_dev_share_dir.sh":
        env["OPENWAM_TRAIN_SCRIPT"] = train_script
    command = ["bash", str(entry), str(queue)]
    children = []
    with (tmp_path / "supervisors.log").open("w") as log:
        try:
            first = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(first)
            deadline = time.monotonic() + 15
            while not (tmp_path / "train_count").exists() and first.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (tmp_path / "train_count").exists(), (tmp_path / "supervisors.log").read_text()
            first.kill()
            first.wait(timeout=5)
            # Resume using only the queue directory; the wrapper recalls 5B.
            env.pop("OPENWAM_TRAIN_SCRIPT", None)
            second = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(second)
            assert second.wait(timeout=20) == 0, (tmp_path / "supervisors.log").read_text()
            assert (tmp_path / "train_count").read_text() == "train\n"
            assert (queue / "eval_count").read_text() == "2"
            assert (queue / "DONE").exists()
            assert pipeline.read_json(queue / "job.json")["command"] == ["bash", str(tmp_path / train_script)]
        finally:
            (queue / "STOP").touch()
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)
