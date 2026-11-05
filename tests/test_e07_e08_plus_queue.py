import copy
import json

import pytest

from scripts.e07_e08_plus_queue import COUNTS, plus_result


def full_summary():
    categories = [dict(category=name, successes=count-1, trials=count, success_rate=(count-1)/count,
                       tasks_complete=count, tasks_expected=count) for name, count in COUNTS.items()]
    total = sum(row["successes"] for row in categories)
    return dict(smoke=False, complete=True, selected_run_complete=True, invalid_records=0,
        expected_full_trials=10030, category_results=categories,
        overall=dict(successes=total, trials=10030, success_rate=total/10030, tasks_complete=10030, tasks_expected=10030),
        total_average_success_rate=total/10030)


@pytest.mark.parametrize("invalid", [None, "smoke", "incomplete", "errors", "category", "rate", "duplicate"])
def test_plus_acceptance_requires_full_seven_category_counts(tmp_path, invalid):
    data = full_summary()
    if invalid == "smoke": data["smoke"] = True
    if invalid == "incomplete": data["complete"] = False
    if invalid == "errors": data["invalid_records"] = 1
    if invalid == "category": data["category_results"][0]["trials"] -= 1
    if invalid == "rate": data["total_average_success_rate"] = .1
    if invalid == "duplicate": data["category_results"][-1] = copy.deepcopy(data["category_results"][0])
    path = tmp_path/"summary.json"
    path.write_text(json.dumps(data))
    if invalid:
        with pytest.raises(ValueError): plus_result(path)
    else:
        result = plus_result(path)
        assert result["trials"] == 10030 and result["successes"] == 10023


def test_resumable_plus_only_does_not_repeat_standard_libero(tmp_path, monkeypatch):
    import sys
    from benchmarks.vla_harness import run_resumable as controller
    checkpoint = tmp_path/"checkpoint"
    checkpoint.mkdir()
    (checkpoint/"config.yaml").write_text("model: {}\n")
    (checkpoint/"weights.safetensors").write_bytes(b"test")
    calls = []
    monkeypatch.setattr(controller, "run_stage", lambda args, job, stage: calls.append(stage) or True)
    monkeypatch.setattr(controller, "stop_owned_processes", lambda job: None)
    monkeypatch.setattr(sys, "argv", ["eval", "--benchmarks", "libero-plus", "--ckpt-dir", str(checkpoint),
        "--ckpt-name", "weights.safetensors", "--output-dir", str(tmp_path/"evaluation")])
    assert controller.main() == 0
    assert calls == ["libero-plus"]


def test_plus_report_preserves_standard_results_and_deduplicates(tmp_path, monkeypatch):
    import scripts.e07_e08_plus_queue as queue
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    destination = tmp_path/"queue"
    destination.mkdir()
    target = tmp_path/"result.md"
    target.write_text("| E07 | dino | 95.00% | 未安排 | 标准完成 |\n| E08 | siglip | 96.00% | 未安排 | 标准完成 |\nExisting suite details\n")
    summary = tmp_path/"summary.json"
    summary.write_text(json.dumps(full_summary()))
    results = {"E07": plus_result(summary)}
    queue.write_report(destination, results)
    queue.write_report(destination, results)
    text = target.read_text()
    assert text.count("<!-- E07_E08_PLUS_BEGIN -->") == 1
    assert text.count("| Camera Viewpoints |") == 1
    assert "95.00%" in text and "96.00%" in text and "10023/10030" in text
    assert "Existing suite details" in text
    assert "| E08 | siglip | 96.00% | 已安排，待完成 |" in text


def test_plus_queue_waits_then_runs_e07_and_e08_without_training(tmp_path, monkeypatch):
    import sys
    import scripts.e07_e08_plus_queue as queue
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path/"outputs").mkdir()
    standard = {name: tmp_path/(name+"_standard") for name in ("E07", "E08")}
    for name, folder in standard.items():
        (folder/name).mkdir(parents=True)
        (folder/name/"accepted.json").write_text(json.dumps(dict(summary="standard.json", run=name+"_run", checkpoint="checkpoint_step_10690.safetensors")))
    monkeypatch.setattr(queue, "STANDARD_QUEUES", standard)
    events = []
    monkeypatch.setattr(queue, "wait_for_completed_queue", lambda state, dependency: events.append(("wait", dependency)))
    monkeypatch.setattr(queue, "write_report", lambda *args: None)
    monkeypatch.setattr(queue, "summary_result", lambda path: {})
    monkeypatch.setattr(queue, "source_fingerprint", lambda: "fixed")
    def prepare(source, checkpoint, destination, *args, **kwargs):
        destination.mkdir(parents=True)
        assert kwargs["corrected"] is True
        events.append(("prepare", source))
        return destination
    monkeypatch.setattr(queue, "prepare_checkpoint", prepare)
    monkeypatch.setattr(queue, "prepare_deploy", lambda path: None)
    monkeypatch.setattr(queue, "plus_result", lambda path: dict(summary=str(path)))
    monkeypatch.setattr(queue, "validate_records", lambda *args: None)
    monkeypatch.setattr(queue.Queue, "wait_idle", lambda self: None)
    monkeypatch.setattr(queue.Queue, "cleanup_orphaned_stages", lambda self: None)
    def evaluate(self, stage, command):
        assert command[command.index("--benchmarks")+1] == "libero-plus"
        assert command[command.index("--correct-instruction-suffix")+1] == "true"
        events.append(("evaluate", stage.parent.name))
    monkeypatch.setattr(queue.Queue, "run_stage", evaluate)
    monkeypatch.setattr(queue.Queue, "train_evaluate", lambda *args: pytest.fail("Plus queue must never train"))
    destination = tmp_path/"queue"
    monkeypatch.setattr(sys, "argv", ["plus", "--queue", str(destination)])
    queue.main()
    assert events == [("wait", standard["E08"]), ("prepare", "E07_run"), ("evaluate", "E07"),
                      ("prepare", "E08_run"), ("evaluate", "E08")]
    assert (destination/"DONE").exists()
