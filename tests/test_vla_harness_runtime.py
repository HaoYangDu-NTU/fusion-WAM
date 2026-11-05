"""Preserve the real failing process and signal before terminating survivors."""

import json
import signal
import subprocess
import sys

import pytest
from vla_eval.recording import RecordingStore

from benchmarks.vla_harness.runtime import RuntimeMonitor


def test_records_killed_server_before_cleanup(tmp_path):
    survivor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    killed = subprocess.Popen([sys.executable, "-c", "import os,signal; os.kill(os.getpid(),signal.SIGKILL)"])
    try:
        killed.wait(timeout=10)
        monitor = RuntimeMonitor(tmp_path)
        with pytest.raises(RuntimeError, match="SIGKILL"):
            monitor.check([survivor, killed], [])
        report = json.loads((tmp_path / "process_failure.json").read_text())
        assert [(p["pid"], p["returncode"], p["signal"]) for p in report["failed_processes"]] == [
            (killed.pid, -signal.SIGKILL, "SIGKILL")
        ]
        assert survivor.poll() is None
        assert next(p for p in report["processes"] if p["pid"] == survivor.pid)["returncode"] is None
        assert "memory_events" in report
    finally:
        survivor.terminate()
        survivor.wait(timeout=10)


def test_completed_worker_is_not_a_failure(tmp_path):
    worker = subprocess.Popen([sys.executable, "-c", "pass"])
    worker.wait(timeout=10)
    RuntimeMonitor(tmp_path).check([], [worker])
    assert not (tmp_path / "process_failure.json").exists()


def test_live_workers_with_recorded_exception_fail_fast(tmp_path):
    database = tmp_path / "recording.sqlite"
    store = RecordingStore(database)
    common = dict(
        sid="session",
        eval_id="test",
        task_name="libero_spatial/0",
        steps=2,
        elapsed_sec=1.0,
        context={"suite": "libero_spatial", "task_id": 0},
        jsonl_path="",
    )
    monitor = RuntimeMonitor(tmp_path, recording_db=database)
    try:
        # A normal unsuccessful rollout is a valid score, not an infrastructure error.
        store.upsert_episode_result(
            **common,
            eid="0",
            episode_id=0,
            status="fail",
            metrics={"success": False},
            failure_reason=None,
            failure_detail=None,
        )
        monitor.check([], [], force=True)
        store.upsert_episode_result(
            **common,
            eid="1",
            episode_id=1,
            status="error",
            metrics={},
            failure_reason="exception",
            failure_detail="Model forward TypeError",
        )
        with pytest.raises(RuntimeError, match="Evaluation episode failed"):
            monitor.check([], [], force=True)
        report = json.loads((tmp_path / "episode_failure.json").read_text())
        assert len(report) == 1
        assert report[0]["episode_id"] == 1
        assert report[0]["failure_detail"] == "Model forward TypeError"
    finally:
        store.close()
