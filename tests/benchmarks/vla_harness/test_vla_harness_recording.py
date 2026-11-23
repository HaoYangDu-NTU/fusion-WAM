"""Exercise real concurrent harness recording and its final durable snapshot."""

import json
import multiprocessing
import shutil
import sqlite3
from pathlib import Path

from vla_eval.recording import RecordingStore
from vla_eval.results.merge import merge_db

from benchmarks.vla_harness.recording import persist_recording_db, prepare_recording_db
from benchmarks.vla_harness.summarize import SUITES, summarize


def _write_worker(path, worker, gate):
    gate.wait(timeout=60)
    store = RecordingStore(path)
    suite, task_id = SUITES[worker // 10], worker % 10
    for trial in range(50):
        sid, eid = str(worker), str(trial)
        store.upsert_step_rows(sid, eid, {0: {"reward": 0.0}, 1: {"reward": 1.0}})
        store.upsert_episode_result(
            sid=sid,
            eid=eid,
            eval_id="fixture-OpenWAMLIBERO",
            task_name=f"{suite}/{task_id}/storage_fixture",
            episode_id=trial,
            status="success",
            metrics={"success": True},
            steps=2,
            elapsed_sec=0.1,
            context={"suite": suite, "task_id": task_id},
            jsonl_path="",
            failure_reason=None,
            failure_detail=None,
        )
    store.close()


def test_forty_writers_local_wal_and_persistent_merge(tmp_path):
    local_db, output_db = prepare_recording_db(tmp_path, "fixture")
    assert local_db.is_relative_to(Path(__file__).resolve().parents[3] / "vlaharness_tmp")
    anchor = RecordingStore(output_db)
    try:
        anchor.upsert_eval_metadata(
            "fixture-OpenWAMLIBERO", "OpenWAMLIBERO", {"metric_keys": {"success": "mean"}, "config": {}}
        )
        assert Path(str(local_db) + "-wal").exists()
        assert Path(str(local_db) + "-shm").exists()
        assert not Path(str(output_db) + "-wal").exists()
        assert not Path(str(output_db) + "-shm").exists()
        ctx = multiprocessing.get_context("spawn")
        gate = ctx.Barrier(40)
        writers = [ctx.Process(target=_write_worker, args=(output_db, i, gate)) for i in range(40)]
        for process in writers:
            process.start()
        for process in writers:
            process.join(timeout=120)
        assert [process.exitcode for process in writers] == [0] * 40
    finally:
        for process in locals().get("writers", []):
            if process.is_alive():
                process.terminate()
                process.join()
        anchor.close()
    persist_recording_db(local_db, output_db)
    assert not output_db.is_symlink()
    with sqlite3.connect(output_db) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT count(*) FROM episode_results").fetchone()[0] == 2000
        assert connection.execute("SELECT count(*) FROM step_rows").fetchone()[0] == 4000
    merge_db(output_db, tmp_path)
    tasks = [{"suite": suite, "task_id": i} for suite in SUITES for i in range(10)]
    (tmp_path / "run.json").write_text(
        json.dumps({"synthetic_storage_fixture": True, "plan": {"tasks": tasks, "num_trials": 50, "smoke": False}})
    )
    assert summarize(tmp_path)["total_average_success_rate"] == 1.0
    shutil.rmtree(local_db.parent)


def test_summary_only_export_preserves_steps_and_matches_full_export(tmp_path):
    db = tmp_path / "recording.sqlite"
    store = RecordingStore(db)
    store.upsert_eval_metadata("eval", "benchmark", {"metric_keys": {"success": "mean"}, "config": {}})
    store.upsert_step_rows("session", "episode", {0: {"reward": 0.0}, 1: {"reward": 1.0}})
    store.upsert_episode_result(sid="session", eid="episode", eval_id="eval", task_name="task",
        episode_id=0, status="success", metrics={"success": True}, steps=2, elapsed_sec=0.1,
        context={"category": "Sensor Noise"}, jsonl_path="steps.jsonl", failure_reason=None, failure_detail=None)
    store.close()
    summary_only = merge_db(db, tmp_path / "summary_only", export_step_jsonl=False)[0]
    assert not (tmp_path / "summary_only/steps.jsonl").exists()
    with sqlite3.connect(db) as connection:
        assert connection.execute("select count(*) from step_rows").fetchone()[0] == 2
    full = merge_db(db, tmp_path / "full")[0]
    assert len((tmp_path / "full/steps.jsonl").read_text().splitlines()) == 2
    for value in (summary_only, full):
        value.pop("created_at")
    assert summary_only == full


def test_episode_only_recording_keeps_result_without_step_rows(tmp_path):
    from vla_eval.recording import EpisodeRecorder

    db = tmp_path / "metrics.sqlite"
    store = RecordingStore(db)
    store.upsert_eval_metadata("eval", "benchmark", {"metric_keys": {"success": "mean"}, "config": {}})
    recorder = EpisodeRecorder(store=store, sid="session", eid="episode", eval_id="eval",
        output_dir=tmp_path, filename_stem="trial", context={"category": "Sensor Noise"}, record_step=False)
    assert recorder.is_active
    assert recorder.db_path == ""  # Server must not write inference traces either.
    recorder.record_step(reward=1.0, action=[0.1])
    recorder.close(status="success", metrics={"success": True}, task_name="task", steps=1)
    store.close()
    with sqlite3.connect(db) as connection:
        assert connection.execute("select count(*) from step_rows").fetchone()[0] == 0
        assert connection.execute("select status from episode_results").fetchall() == [("success",)]
    result = merge_db(db, tmp_path / "summary", export_step_jsonl=False)[0]
    assert result["tasks"][0]["episodes"][0]["metrics"]["success"] is True
