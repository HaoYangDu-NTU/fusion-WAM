import json

import pytest
from omegaconf import OmegaConf

from openwam.experiments import resolve_experiment


@pytest.mark.parametrize("profile,weight", [("E01-LV05-Cache-W2", 0.5), ("E01-LV2-Cache-W2", 2.0)])
def test_video_loss_profiles_change_only_weight_and_runtime(profile, weight):
    base = OmegaConf.to_container(resolve_experiment("E01-Cache")[0], resolve=True)
    current = OmegaConf.to_container(resolve_experiment(profile)[0], resolve=True)
    assert current["training"]["lambda_video"] == weight
    assert current["training"]["lambda_action"] == 1
    assert current["training"]["resume_ckpt_path"] is None
    assert current["training"].pop("dataloader_worker_threads") == 2
    assert current["training"].pop("report_future_quality") is True
    current["training"]["lambda_video"] = 1
    for value in (base, current):
        value["training"].pop("output_path")
        value["project"].pop("output_dir")
        value["project"]["wandb"].pop("run_name")
    assert current == base


def test_loss_report_preserves_shift_results_and_updates_without_duplicates(tmp_path, monkeypatch):
    import scripts.e01_ablation_queue as queue

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    target = tmp_path/"result.md"
    target.write_text("Historical encoder results\n")
    destination = tmp_path/"queue"
    destination.mkdir()
    (destination/"video_loss_plan.json").write_text(json.dumps({"weights": [0.5, 2.0]}))
    row = dict(suites={suite: 0.9 for suite in queue.SUITES}, overall=0.9,
               summary=str(tmp_path/"summary.json"))
    results = {name: dict(row) for name in ("E01-Base", "E01-S12")}
    queue.write_report(destination, results)
    assert "| E01-LV05 | 0.5 | 1 | 待完成" in target.read_text()
    results["E01-LV05"] = dict(row)
    queue.write_report(destination, results)
    queue.write_report(destination, results)
    text = target.read_text()
    assert text.count("<!-- E01_VIDEO_LOSS_BEGIN -->") == 1
    assert text.count("| E01-LV05 |") == 1
    assert "| E01-LV05 | 0.5 | 1 | 90.00%" in text
    assert "| E01-LV2 | 2 | 1 | 待完成" in text
    assert "| E01-S12 |" in text and "Historical encoder results" in text
    assert "除以对应权重" in text
    assert (destination/"RESULTS.md").read_text().count("## E01 Video Loss") == 1


@pytest.mark.parametrize("profile,label", [("E01-LV05-Cache-W2", "E01-LV05"), ("E01-LV2-Cache-W2", "E01-LV2")])
def test_loss_queue_requires_predecessor_and_runs_only_requested_experiment(tmp_path, monkeypatch, profile, label):
    import sys
    import scripts.e01_ablation_queue as queue

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    destination, predecessor = tmp_path/"queue", tmp_path/"predecessor"
    events = []
    monkeypatch.setattr(queue, "wait_for_completed_queue", lambda state, path:
                        events.append(("dependency", path)) or {"prior": {"accepted": True}})
    monkeypatch.setattr(queue, "audit_base", lambda: {"base_result": {}})
    monkeypatch.setattr(queue, "source_fingerprint", lambda: "test-source")
    monkeypatch.setattr(queue, "write_report", lambda path, results, selection:
                        events.append(("report", dict(results))))
    monkeypatch.setattr(queue.Queue, "wait_idle", lambda self: None)
    monkeypatch.setattr(queue.Queue, "cleanup_orphaned_stages", lambda self: None)
    monkeypatch.setattr(queue.Queue, "train_evaluate", lambda self, name, result_label:
                        events.append(("run", name, result_label)) or {"accepted": True})
    monkeypatch.setattr(sys, "argv", ["queue", "--run", "--queue", str(destination),
        "--single-experiment", profile, "--result-label", label, "--after-queue", str(predecessor)])
    queue.main()
    assert events[0] == ("dependency", predecessor)
    assert [event for event in events if event[0] == "run"] == [("run", profile, label)]
    assert events[-1][1][label] == {"accepted": True}
    assert events[-1][1]["prior"] == {"accepted": True}
    assert json.loads((destination/"status.json").read_text())["benchmarks"] == ["libero"]
    assert (destination/"DONE").exists()
