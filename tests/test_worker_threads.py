from functools import partial
import torch
from omegaconf import OmegaConf

from openwam.experiments import resolve_experiment
from openwam.train.utils.seeding import init_worker_threads


class WorkerRuntimeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.get_num_threads(), torch.initial_seed()


def seed_worker(worker_id):
    torch.manual_seed(42 + worker_id)


def test_worker_threads_preserve_initializer_under_spawn():
    loader = torch.utils.data.DataLoader(
        WorkerRuntimeDataset(), num_workers=2, batch_size=1,
        multiprocessing_context="spawn",
        worker_init_fn=partial(init_worker_threads, num_threads=2, original=seed_worker),
    )
    assert [(int(n.item()), int(seed.item())) for n, seed in loader] == [
        (2, 42), (2, 43), (2, 42), (2, 43),
    ]


def test_p2_worker_profile_keeps_model_data_and_budget():
    base = OmegaConf.to_container(resolve_experiment("E01-P2-Cache")[0], resolve=True)
    current = OmegaConf.to_container(resolve_experiment("E01-P2-Cache-W2")[0], resolve=True)
    assert current["training"].pop("dataloader_worker_threads") == 2
    for value in (base, current):
        value["training"].pop("output_path")
        value["project"].pop("output_dir")  # Hydra's wall-clock-generated directory.
        value["project"]["wandb"].pop("run_name")
    assert current == base


def test_p3_changes_only_action_prediction_and_output_from_p2():
    base = OmegaConf.to_container(resolve_experiment("E01-P2-Cache-W2")[0], resolve=True)
    current = OmegaConf.to_container(resolve_experiment("E01-P3-Cache-W2")[0], resolve=True)
    assert current["model"]["architecture"]["video_prediction_type"] == "v"
    assert current["model"]["architecture"]["action_prediction_type"] == "v"
    current["model"]["architecture"]["action_prediction_type"] = "x"
    for value in (base, current):
        value["training"].pop("output_path")
        value["project"].pop("output_dir")
        value["project"]["wandb"].pop("run_name")
    assert current == base


def test_single_queue_stops_after_p2(tmp_path, monkeypatch):
    import sys
    import scripts.e01_ablation_queue as queue

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(queue, "audit_base", lambda: {"base_result": {}})
    monkeypatch.setattr(queue, "source_fingerprint", lambda: "test-source")
    monkeypatch.setattr(queue, "write_report", lambda *args: None)
    monkeypatch.setattr(queue.Queue, "wait_idle", lambda self: None)
    monkeypatch.setattr(queue.Queue, "cleanup_orphaned_stages", lambda self: None)
    calls = []
    monkeypatch.setattr(queue.Queue, "train_evaluate", lambda self, name, label: calls.append((name, label)) or {})
    monkeypatch.setattr(sys, "argv", ["queue", "--run", "--queue", str(tmp_path / "queue"),
        "--single-experiment", "E01-P2-Cache-W2", "--result-label", "E01-P2"])
    queue.main()
    assert calls == [("E01-P2-Cache-W2", "E01-P2")]
    assert (tmp_path / "queue" / "DONE").exists()


def test_s24_cache_w2_preserves_xx_baseline_except_shift_and_runtime():
    base = OmegaConf.to_container(resolve_experiment("E01-Cache")[0], resolve=True)
    current = OmegaConf.to_container(resolve_experiment("E01-S24-xx-Cache-W2")[0], resolve=True)
    assert current["model"]["architecture"]["video_prediction_type"] == "x"
    assert current["model"]["architecture"]["action_prediction_type"] == "x"
    assert current["model"]["video_backbone"]["shift_video"] == 24
    assert current["model"]["action_backbone"]["shift_action"] == 5
    current["model"]["video_backbone"]["shift_video"] = base["model"]["video_backbone"]["shift_video"]
    assert current["training"].pop("dataloader_worker_threads") == 2
    assert current["training"].pop("report_future_quality") is True
    for value in (base, current):
        value["training"].pop("output_path")
        value["project"].pop("output_dir")
        value["project"]["wandb"].pop("run_name")
    assert current == base


def test_single_shift_queue_preserves_prediction_selection(tmp_path, monkeypatch):
    import json
    import sys
    import scripts.e01_ablation_queue as queue

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    destination = tmp_path / "queue"
    destination.mkdir()
    selection = {"selected": "E01-Base", "rule": "User fixed x/x"}
    (destination / "selection.json").write_text(json.dumps(selection))
    monkeypatch.setattr(queue, "audit_base", lambda: {"base_result": {}})
    monkeypatch.setattr(queue, "source_fingerprint", lambda: "test-source")
    reports = []
    monkeypatch.setattr(queue, "write_report", lambda path, rows, chosen=None: reports.append((dict(rows), chosen)))
    monkeypatch.setattr(queue.Queue, "wait_idle", lambda self: None)
    monkeypatch.setattr(queue.Queue, "cleanup_orphaned_stages", lambda self: None)
    monkeypatch.setattr(queue.Queue, "train_evaluate", lambda self, name, label: {"complete": True})
    monkeypatch.setattr(sys, "argv", ["queue", "--run", "--queue", str(destination),
        "--single-experiment", "E01-S24-xx-Cache-W2", "--result-label", "E01-S24"])
    queue.main()
    assert reports[-1][0]["E01-S24"] == {"complete": True}
    assert reports[-1][1] == selection


def test_s60_matches_s24_except_video_shift_and_output():
    base = OmegaConf.to_container(resolve_experiment("E01-S24-xx-Cache-W2")[0], resolve=True)
    current = OmegaConf.to_container(resolve_experiment("E01-S60-xx-Cache-W2")[0], resolve=True)
    assert current["model"]["video_backbone"]["shift_video"] == 60
    current["model"]["video_backbone"]["shift_video"] = 24
    for value in (base, current):
        value["training"].pop("output_path")
        value["project"].pop("output_dir")
        value["project"]["wandb"].pop("run_name")
    assert current == base


def test_s12_matches_s60_except_video_shift_and_output():
    base = OmegaConf.to_container(resolve_experiment("E01-S60-xx-Cache-W2")[0], resolve=True)
    current = OmegaConf.to_container(resolve_experiment("E01-S12-xx-Cache-W2")[0], resolve=True)
    assert current["model"]["video_backbone"]["shift_video"] == 12
    current["model"]["video_backbone"]["shift_video"] = 60
    for value in (base, current):
        value["training"].pop("output_path")
        value["project"].pop("output_dir")
        value["project"]["wandb"].pop("run_name")
    assert current == base
