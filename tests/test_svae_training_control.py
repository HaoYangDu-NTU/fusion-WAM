"""S-VAE stopping policy and actual two-rank CPU training/save integration."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("svae_trainer", ROOT / "scripts/svae_train/train_svae.py")
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def test_consecutive_improvements_and_reset():
    monitor = TRAINER._EarlyStopping()
    assert monitor.update(100) == (False, None)
    assert not monitor.update(99)[0]  # exactly 1% is sufficient
    assert monitor.bad_epochs == 0
    assert not monitor.update(98.5)[0]
    assert not monitor.update(98.6)[0]  # an increase also counts
    assert monitor.bad_epochs == 2
    assert not monitor.update(95)[0]
    assert monitor.bad_epochs == 0
    assert not monitor.update(95)[0]
    assert not monitor.update(95)[0]
    assert monitor.update(95)[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_metric_rejected(value):
    with pytest.raises(ValueError, match="Non-finite"):
        TRAINER._EarlyStopping().update(value)


@pytest.mark.parametrize("patience,delta", [(0, .01), (3, -1), (3, 1)])
def test_invalid_policy_rejected(patience, delta):
    with pytest.raises(ValueError):
        TRAINER._EarlyStopping(patience, delta)


@pytest.mark.parametrize("mode", ["early_stop", "completed", "partial"])
def test_distributed_training_and_checkpoint(tmp_path, mode):
    features = tmp_path / "features"
    features.mkdir()
    # Four clips: one batch per rank per epoch; partial mode has two.
    count = 8 if mode == "partial" else 4
    torch.save({"raw_dim": 8, "features": torch.randn(count, 8, 1, 2, 2).half()},
               features / "features_rank0_part0.pt")
    torch.save({"raw_dim": 8, "mean": torch.zeros(8), "std": torch.ones(8)}, features / "stats.pt")
    output = tmp_path / "output"
    wrapper = tmp_path / "run.py"
    wrapper.write_text('''
import os, sys
sys.path.insert(0, sys.argv[1])
import torch
from omegaconf import OmegaConf
import openwam.model.video_backbone.encoder.svae as api
from scripts.svae_train.train_svae import main
original_loss = api.svae_loss
def controlled_metrics(*args, **kwargs):
    loss, stat = original_loss(*args, **kwargs)
    # Real forward/backward/optimizer; deterministic per-rank monitor values.
    stat["mse"] = loss.detach().new_tensor(float(int(os.environ["RANK"]) + 1))
    stat["cos"] = loss.detach().new_tensor(0.)
    return loss, stat
api.svae_loss = controlled_metrics
cfg = OmegaConf.load(sys.argv[1] + "/configs/model/video_backbone/encoder/svae/train.yaml")
cfg.svae.update(dict(latent_dim=2, num_heads=2, num_layers=1, intermediate_size=16))
cfg.train.update(dict(features_dir=sys.argv[2], output_dir=sys.argv[3], batch_size=2,
                      num_workers=0, mixed_precision="no", log_every=1))
cfg.train.wandb.project = None
mode = sys.argv[4]
cfg.train.early_stopping.enabled = True
if mode == "completed":
    cfg.train.num_epochs = 3
    cfg.train.num_workers = 2
    cfg.train.early_stopping.enabled = False
elif mode == "partial":
    cfg.train.max_steps = 3
    cfg.train.kl_warmup_ratio = 0
main.__wrapped__(cfg)
''')
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "true",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                             "--nproc_per_node=2", str(wrapper), str(ROOT), str(features), str(output), mode],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    if sys.platform == "linux" and shutil.which("fincore"):
        # A successful stage must leave the feature cache cold before WAM starts.
        resident = subprocess.check_output(
            ["fincore", "--bytes", "--noheadings", "--output", "RES",
             str(features / "features_rank0_part0.pt")], text=True,
        )
        assert int(resident.strip()) == 0
    final, = output.glob("*/svae.pt")
    payload = torch.load(final, weights_only=True)
    state = payload["training_state"]
    assert payload["world_size"] == 2
    assert all(row["reconstruction"] == 1.5 for row in payload["epoch_metrics"])
    assert state["optimizer"]["state"] and state["scheduler"]
    from openwam.model.video_backbone.encoder.svae.model import load_svae
    # Same loader used by the frozen encoder must accept the extended checkpoint.
    loaded = load_svae(str(final))
    assert loaded.latent_dim == 2
    if mode == "early_stop":
        assert state["stop_reason"] == "early_stopping"
        assert state["epoch"] == state["step"] == 14
        assert state["early_stopping"]["bad_epochs"] == 3
        assert not payload["epoch_metrics"][9]["monitored"]
        assert payload["epoch_metrics"][10]["monitored"]
        assert payload["epoch_metrics"][10]["relative_improvement"] is None
        periodic = torch.load(final.parent / "svae_epoch_0010.pt", weights_only=True)
        assert periodic["training_state"]["epoch"] == 10
        assert periodic["training_state"]["stop_reason"] == "periodic"
        assert len(list(final.parent.glob("svae_epoch_*.pt"))) == 1
    elif mode == "completed":
        assert state["stop_reason"] == "completed"
        assert state["epoch"] == state["step"] == 3
    else:
        assert state["stop_reason"] == "max_steps"
        assert state["step"] == 3 and state["epoch"] == 2
        assert not payload["epoch_metrics"][-1]["monitored"]
        assert not payload["epoch_metrics"][-1]["full_epoch"]
