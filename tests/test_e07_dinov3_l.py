import copy
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from openwam.experiments import resolve_experiment


def test_e07_matches_e01_except_encoder_asset_runtime_and_output():
    base = OmegaConf.to_container(resolve_experiment("E01")[0], resolve=True)
    current, profile, _ = resolve_experiment("E07-Cache")
    current = OmegaConf.to_container(current, resolve=True)
    assert profile["geometry"] == dict(clean_latents=2, future_latents=8, channels=1024,
        tokens=980, temporal_design="native_framewise_history_and_current")
    assert current["dataloader"].pop("frame_feature_cache_dir").endswith("dinov3_l_libero_frames")
    assert current["training"].pop("dataloader_worker_threads") == 2
    assert current["training"].pop("report_future_quality") is True
    assert current["training"]["lambda_video"] == current["training"]["lambda_action"] == 1
    assert current["training"]["resume_ckpt_path"] is None
    assert current["model"]["video_backbone"]["encoder"]["model_path"].endswith("encoder/dinov3-l")
    current["model"]["video_backbone"]["encoder"]["model_path"] = base["model"]["video_backbone"]["encoder"]["model_path"]
    for cfg in (base, current):
        cfg["training"].pop("output_path")
        cfg["project"].pop("output_dir")
        cfg["project"]["wandb"].pop("run_name")
    assert current == base


@pytest.mark.parametrize("width", [1280, 1024])
def test_native_cache_width_preserves_hplus_identity(tmp_path, monkeypatch, width):
    from openwam.dataloader import frame_cache_specs as specs
    expected = dict(shape=[1280, 14, 28], code={"original.py": "hash"}, weights={"model": "weight"})
    monkeypatch.setattr(specs, "legacy_identity", lambda *args: copy.deepcopy(expected))
    (tmp_path/"config.json").write_text(json.dumps(dict(hidden_size=width, patch_size=16)))
    actual = specs.frame_cache_identity({}, dict(name="dinov3", model_path=str(tmp_path)), None)
    if width == 1280:
        assert actual == expected
    else:
        assert actual["shape"] == [1024, 14, 28]
        assert "dataloader/frame_cache_specs.py" in actual["code"]
        assert actual["weights"] == expected["weights"]


@pytest.mark.parametrize("cache_passes", [True, False])
@pytest.mark.parametrize("experiment", ["E07", "E08"])
def test_e07_waits_builds_validates_then_trains_libero_only(tmp_path, monkeypatch, cache_passes, experiment):
    import sys
    import scripts.e07_train_eval_queue as queue

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path/"outputs").mkdir()
    cfg, _, _ = resolve_experiment("E07-Cache")
    cfg.dataloader.frame_feature_cache_dir = str(tmp_path/"cache")
    monkeypatch.setattr(queue, "resolve_experiment", lambda name: (cfg, {}, []))
    monkeypatch.setattr(queue, "source_fingerprint", lambda: "unchanged")
    events = []
    monkeypatch.setattr(queue, "wait_for_completed_queue", lambda *args: events.append("predecessor"))
    monkeypatch.setattr(queue, "write_report", lambda *args: events.append("report"))
    monkeypatch.setattr(queue.Queue, "wait_idle", lambda self: None)
    monkeypatch.setattr(queue.Queue, "cleanup_orphaned_stages", lambda self: None)
    def stage(self, folder, command):
        events.append((folder.name, command))
        if folder.name == "cache_validation":
            Path(cfg.dataloader.frame_feature_cache_dir).mkdir()
            (Path(cfg.dataloader.frame_feature_cache_dir)/"validation.json").write_text(json.dumps({"passed": cache_passes}))
    monkeypatch.setattr(queue.Queue, "run_stage", stage)
    monkeypatch.setattr(queue.Queue, "train_evaluate", lambda self, name, label:
                        events.append(("train_evaluate", name, label)) or {})
    destination = tmp_path/"queue"
    monkeypatch.setattr(sys, "argv", ["queue", "--experiment", experiment, "--queue", str(destination), "--after-queue", str(tmp_path/"prior")])
    if cache_passes:
        queue.main()
        assert events[-2:] == [("train_evaluate", experiment+"-Cache", experiment), "report"]
        assert json.loads((destination/"status.json").read_text())["benchmarks"] == ["libero"]
        assert (destination/"DONE").exists()
    else:
        with pytest.raises(RuntimeError, match="equivalence failed"):
            queue.main()
        assert not any(isinstance(e, tuple) and e[0] == "train_evaluate" for e in events)
        assert not (destination/"DONE").exists()
    assert events[:2] == ["predecessor", "report"]
    assert events[2][0] == "cache_build" and events[3][0] == "cache_validation"
    assert experiment in events[2][1] and "--nproc_per_node=4" in events[2][1]
