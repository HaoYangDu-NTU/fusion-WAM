"""Evaluation overrides cover standard and nested Plus deployments without changing defaults."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from benchmarks.vla_harness.model_server import load_native_server

@pytest.mark.parametrize("subpath", ["training/run", "training/run/eval_libero_plus/checkpoint/corrected_text"])
@pytest.mark.parametrize("override", [False, True])
def test_experiment_evaluation_disables_velocity_cache(tmp_path, monkeypatch, subpath, override):
    from openwam.deploy import server
    stage = tmp_path / "stage"
    checkpoint = stage / subpath
    checkpoint.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/deploy.yaml")
    assert cfg.optimization.dit_cache.enabled is True
    if override:
        cfg.optimization.dit_cache.enabled = False
        OmegaConf.save(cfg, stage / "vla_harness_deploy.yaml")
    captured = []
    def build(config, *_args, **_kwargs):
        captured.append(config)
        return SimpleNamespace(_init_policy=lambda: None)
    monkeypatch.setattr(server, "build_server_from_config", build)
    load_native_server(str(checkpoint), "checkpoint.safetensors", "cpu", 10, 10, False)
    assert captured[0].optimization.dit_cache.enabled is False
    assert captured[0].inference.denoise_steps == 10
    assert captured[0].inference.denoise_mode == "sync"
    assert captured[0].optimization.compile.enabled is False
    assert captured[0].optimization.decode_video is False
