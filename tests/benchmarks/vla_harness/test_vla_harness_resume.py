"""Task-level retry must preserve completed failures and never combine attempts."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from vla_eval.recording import RecordingStore

from benchmarks.vla_harness import run_resumable as resume
from benchmarks.vla_harness.run_eval import make_plan


def records(tasks, trials, *, success=False):
    return [{**task, "episode_id": i, "metrics": {"success": success}, "status": "success"}
            for task in tasks for i in range(trials)]


@pytest.mark.parametrize("trials", [1, 50])
@pytest.mark.parametrize("bad", ["incomplete", "error", "pending", "duplicate", "non_boolean"])
def test_only_complete_valid_tasks_are_skipped(trials, bad):
    tasks = [{"suite": "libero_spatial", "task_id": i} for i in range(2)]
    rows = records(tasks, trials)
    if bad == "incomplete":
        rows.pop()
    elif bad in ("error", "pending"):
        rows[-1]["status"] = bad
    elif bad == "duplicate":
        rows.append(dict(rows[-1]))
    else:
        rows[-1]["metrics"]["success"] = 0
    completed = resume.completed_tasks(rows, tasks, trials)
    assert set(completed) == {("libero_spatial", 0)}
    assert all(not row["metrics"]["success"] for row in completed[("libero_spatial", 0)])


def test_committed_sqlite_recovers_without_launcher_merge(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    db = attempt / "recordings.sqlite"
    (attempt / "run.json").write_text(json.dumps({"recording_db": {"output_path": str(db)}}))
    store = RecordingStore(db)
    store.upsert_episode_result(
        sid="s", eid="e", eval_id="test", task_name="libero_spatial/0", episode_id=0,
        status="success", metrics={"success": False}, steps=600, elapsed_sec=10,
        context={"suite": "libero_spatial", "task_id": 0}, jsonl_path="",
        failure_reason=None, failure_detail=None,
    )
    try:
        # Writer remains open: results may still live in the WAL.
        rows = resume.read_attempt(attempt, "libero-plus")
        assert rows[0]["metrics"]["success"] is False
        assert set(resume.completed_tasks(rows, [{"suite": "libero_spatial", "task_id": 0}], 1)) == {
            ("libero_spatial", 0)
        }
    finally:
        store.close()


def test_partial_attempts_are_never_stitched_together_and_first_completion_is_kept(tmp_path, monkeypatch):
    tasks = [{"suite": "libero_spatial", "task_id": i} for i in range(2)]
    rows = records(tasks, 50)
    monkeypatch.setattr(resume, "read_attempt", lambda *args: rows)
    complete = {}
    rows = records(tasks, 50)[:-1]
    resume.collect(tmp_path, tmp_path / "attempt1", "libero", tasks, 50, complete)
    assert set(complete) == {("libero_spatial", 0)}
    rows = records(tasks[1:], 50)[-1:]
    resume.collect(tmp_path, tmp_path / "attempt2", "libero", tasks, 50, complete)
    assert len(complete) == 1
    rows = records(tasks, 50, success=True)
    resume.collect(tmp_path, tmp_path / "attempt3", "libero", tasks, 50, complete)
    assert len(complete) == 2
    assert complete[("libero_spatial", 0)]["source_attempt"].endswith("attempt1")
    assert not complete[("libero_spatial", 0)]["episodes"][0]["metrics"]["success"]
    assert complete[("libero_spatial", 1)]["source_attempt"].endswith("attempt3")


def test_standard_libero_plan_excludes_only_finished_suite_task_pairs(tmp_path):
    excluded = tmp_path / "exclude.json"
    skipped = [["libero_spatial", 0], ["libero_object", 2]]
    excluded.write_text(json.dumps(skipped))
    args = SimpleNamespace(
        benchmark="libero", gpus="0,1,2,3", suites=",".join(resume.SUITES), task_ids=None,
        smoke=False, workers=32, max_batch_size=8, max_wait_time=0.01,
        render="gpu", render_gpus="3", base_port=8940, ckpt_dir=tmp_path,
        ckpt_name="model.safetensors", output_dir=tmp_path, eval_id="test", exclude_tasks_file=excluded,
    )
    plan = make_plan(args)
    selected = {(t["suite"], t["task_id"]) for t in plan["tasks"]}
    assert len(selected) == 38 and not selected.intersection(map(tuple, skipped))
    assigned = [(t["suite"], t["task_id"]) for w in plan["workers"] for t in w["tasks"]]
    assert len(assigned) == len(set(assigned)) == 38 and set(assigned) == selected
    assert ("libero_goal", 0) in selected and ("libero_spatial", 2) in selected


@pytest.mark.parametrize("correct_suffix", [False, True])
def test_retry_launch_uses_exclusions_then_finishes_both_stages(tmp_path, monkeypatch, correct_suffix):
    calls, attempts = [], {}
    plus_tasks = [{"suite": "libero_spatial", "task_id": i, "task_name": str(i),
                   "category": "Camera Viewpoints", "difficulty_level": 1} for i in range(2)]
    monkeypatch.setattr(resume, "task_catalog", lambda _: plus_tasks)
    monkeypatch.setattr(resume, "read_attempt", lambda attempt, _: attempts.get(str(attempt), []))
    monkeypatch.setattr(resume, "stop_owned_processes", lambda _: None)
    monkeypatch.setattr(resume.time, "sleep", lambda _: None)
    # The real Plus summary requires the official category identifiers.
    from benchmarks.vla_harness.libero_plus_protocol import scheduler
    for task in plus_tasks:
        task["category"] = scheduler.PLUS_CATEGORY_ORDER[0]

    def launch(command, **kwargs):
        value = lambda name: command[command.index(name) + 1]
        benchmark = value("--benchmark")
        assert int(value("--base-port")) == (8941 if benchmark == "libero-plus" else 8940)
        if benchmark == "libero-plus":
            assert value("--correct-instruction-suffix") == str(correct_suffix).lower()
        else:
            assert "--correct-instruction-suffix" not in command
        excluded = json.loads(Path(value("--exclude-tasks-file")).read_text())
        catalog = (plus_tasks if benchmark == "libero-plus" else
                   [{"suite": suite, "task_id": i} for suite in resume.SUITES for i in range(10)])
        previous = sum(call[0] == benchmark for call in calls)
        if previous == 0:
            assert excluded == []
            chosen = catalog[:1]
        else:
            assert excluded == [[catalog[0]["suite"], catalog[0]["task_id"]]]
            chosen = catalog[1:]
        calls.append((benchmark, excluded))
        attempts[value("--output-dir")] = records(chosen, 50 if benchmark == "libero" else 1)
        code = 1 if previous == 0 else 0
        return SimpleNamespace(pid=123, returncode=code, poll=lambda: code, wait=lambda: code)

    monkeypatch.setattr(resume.subprocess, "Popen", launch)
    args = SimpleNamespace(output_dir=tmp_path, ckpt_dir=tmp_path, ckpt_name="model.safetensors",
                           gpus="0", workers=2, max_batch_size=2, render_gpus="0", base_port=8940, retry_delay=0)
    args.correct_instruction_suffix = correct_suffix
    for benchmark in ("libero", "libero-plus"):
        assert resume.run_stage(args, "test", benchmark)
        summary = json.loads((tmp_path / benchmark / "summary.json").read_text())
        assert summary["complete"] and summary["total_average_success_rate"] == 0.0
    assert [call[0] for call in calls] == ["libero", "libero", "libero-plus", "libero-plus"]
    # Restarting the controller does not launch a completed benchmark again.
    assert resume.run_stage(args, "test", "libero")
    assert len(calls) == 4


def test_cleanup_targets_only_owned_processes():
    marker = uuid.uuid4().hex
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             env={**os.environ, resume.OWNER_KEY: marker})
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        resume.stop_owned_processes(marker)
        assert owned.wait(timeout=3) < 0
        assert other.poll() is None
    finally:
        for process in (owned, other):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)


def test_shell_restarts_crashed_controller_without_losing_output_dir(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/bin/bash\nif [[ ! -f "$TEST_OUTPUT/first_exit" ]]; then\n'
        '  touch "$TEST_OUTPUT/first_exit"; exit 42\nfi\ntouch "$TEST_OUTPUT/DONE"\n'
    )
    fake_python.chmod(0o755)
    fake_sleep = tmp_path / "sleep"
    fake_sleep.write_text("#!/bin/sh\nexit 0\n")
    fake_sleep.chmod(0o755)
    output = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(resume.ROOT / "scripts/eval_libero_then_plus.sh"), str(tmp_path), "model", str(output)],
        env={**os.environ, "OPENWAM_PYTHON": str(fake_python), "TEST_OUTPUT": str(output),
             "PATH": str(tmp_path) + ":" + os.environ["PATH"]},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0 and (output / "DONE").exists()
    assert "controller exited 42" in (output / "supervisor.log").read_text()


def test_per_stage_settings_do_not_change_legacy_defaults():
    args = SimpleNamespace(workers=32, max_batch_size=8, render_gpus="3", gpus="0,1,2,3", base_port=8990)
    assert resume.stage_options(args, "libero-plus") == dict(workers=32, max_batch_size=8, render_gpus="3", base_port=8994)
    args.plus_workers, args.plus_max_batch_size, args.plus_render_gpus, args.plus_base_port = 128, 32, "1,3", 8990
    assert resume.stage_options(args, "libero-plus") == dict(workers=128, max_batch_size=32, render_gpus="1,3", base_port=8990)
    assert resume.stage_options(args, "libero") == dict(workers=32, max_batch_size=8, render_gpus="3", base_port=8990)


def test_shell_stop_is_not_success(tmp_path):
    (tmp_path / "STOP").touch()
    result = subprocess.run(
        ["bash", str(resume.ROOT / "scripts/eval_libero_then_plus.sh"), str(tmp_path), "model", str(tmp_path / "eval"),
         "--stop-file", str(tmp_path / "STOP")], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 130
    assert not (tmp_path / "eval/DONE").exists()


def test_resource_gate_checks_compute_memory_and_ports(monkeypatch):
    replies = iter(["4321\n", "", "500\n", "", "10\n"])
    monkeypatch.setattr(resume.subprocess, "check_output", lambda *a, **kw: next(replies))
    assert not resume.resources_idle("0", [])
    assert not resume.resources_idle("0", [])
    with resume.socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        assert not resume.resources_idle("0", [busy.getsockname()[1]])
