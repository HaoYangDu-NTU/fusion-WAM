"""Resolve the real dev launchers without starting torchrun or loading weights."""

import os
import re
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def launcher(size, encoder=None, extra=(), env_overrides=None):
    path = ROOT / f"scripts/train_libero_vjepa_{size}_dev.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    script = path.read_text()
    env = os.environ.copy()
    for name in re.findall(r'^([A-Z_]+)="', script, re.MULTILINE):
        env.pop(name, None)
    if encoder is not None:
        env["VJEPA_ENCODER"] = encoder
    env.update(env_overrides or {})
    # Activation does not change overrides; intercept the only launcher call.
    script = "\n".join(line for line in script.splitlines() if not line.startswith(
        ("source /workspace/basics/miniconda3", "conda activate ")
    ))
    assert script.count("exec taskset -c ") == 1
    script = script.replace("exec taskset -c ", "capture_command -c ", 1)
    prefix = 'capture_command() { printf "%s\\0" "$@"; }\n'
    result = subprocess.run(
        ["bash", "-c", prefix + script, "launcher-test", *extra],
        env=env, capture_output=True, text=True,
    )
    return result


def config(size, encoder=None, extra=(), env_overrides=None):
    result = launcher(size, encoder, extra, env_overrides)
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("\0")
    overrides = [arg for arg in args[args.index("scripts/train.py") + 1:] if arg]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=overrides)


@pytest.mark.parametrize("size,default", [("0_6b", "L"), ("1_3b", "G"), ("5b", "G")])
def test_default_and_selectable_encoder(size, default):
    paths = {
        "G": "/workspace/model/vjepa",
        "L": "/workspace/model/vjepa/vitl_dist_vitG_384",
    }
    assert config(size).model.video_backbone.encoder.model_path == paths[default]
    for encoder, path in paths.items():
        cfg = config(size, encoder)
        assert cfg.model.video_backbone.encoder.model_path == path
        assert cfg.model.video_backbone.encoder.seperate_encode_view is True
        assert cfg.model.video_backbone.encoder.history_num_tubelets == 1
        assert cfg.model.video_backbone.encoder.use_avg_t_pooling is False
        assert cfg.training.batch_size == 64
        assert cfg.training.num_epochs == 10


@pytest.mark.parametrize("encoder", ["L", "G"])
def test_capacity_launchers_preserve_non_capacity_config(encoder):
    cfgs = [OmegaConf.to_container(config(size, encoder), resolve=False)
            for size in ("0_6b", "1_3b", "5b")]
    for cfg, expected in zip(cfgs, ((320, 1280), (1024, 4096), (1024, 4096))):
        action = cfg["model"]["action_backbone"]
        assert (action.pop("dim"), action.pop("ffn_dim")) == expected
        cfg["model"]["video_backbone"].pop("dit_config", None)
        cfg["model"]["video_backbone"].pop("name")
        cfg["training"].pop("output_path")
    assert cfgs[0] == cfgs[1] == cfgs[2]


@pytest.mark.parametrize("size", ["0_6b", "1_3b", "5b"])
def test_encoder_validation_and_trailing_overrides(size):
    invalid = launcher(size, "H")
    assert invalid.returncode == 2
    assert "VJEPA_ENCODER must be L or G" in invalid.stderr
    cfg = config(size, "L", (
        "model.video_backbone.encoder.model_path=/custom/encoder",
        "training.batch_size=8",
    ))
    assert cfg.model.video_backbone.encoder.model_path == "/custom/encoder"
    assert cfg.training.batch_size == 8


def test_1_3b_fusion_defaults_and_hydra_layer_list():
    encoder = config("1_3b").model.video_backbone.encoder
    assert encoder.condition_fusion_method == encoder.future_fusion_method == "last_layer"
    assert list(encoder.fusion_layers) == [27, 29, 31, 33, 35, 37, 39]


@pytest.mark.parametrize("layers,expected", [("all_layers", "all_layers"), ("[0,3,11]", [0, 3, 11])])
def test_1_3b_independent_fusion_env_and_trailing_override(layers, expected):
    env = {"CONDITION_FUSION_METHOD": "mean_plus_global", "FUTURE_FUSION_METHOD": "mean", "FUSION_LAYERS": layers}
    encoder = config("1_3b", env_overrides=env).model.video_backbone.encoder
    assert encoder.condition_fusion_method == "mean_plus_global"
    assert encoder.future_fusion_method == "mean"
    assert encoder.fusion_layers == expected
    encoder = config("1_3b", extra=("model.video_backbone.encoder.future_fusion_method=last_layer",),
                     env_overrides=env).model.video_backbone.encoder
    assert encoder.condition_fusion_method == "mean_plus_global"
    assert encoder.future_fusion_method == "last_layer"


@pytest.mark.parametrize("name", ["CONDITION_FUSION_METHOD", "FUTURE_FUSION_METHOD"])
def test_1_3b_invalid_fusion_method_rejected_before_launch(name):
    result = launcher("1_3b", env_overrides={name: "typo"})
    assert result.returncode == 2
    assert f"{name} must be last_layer, mean, mean_plus_global, concat_before_bottleneck, or adafuse" in result.stderr
    assert "scripts/train.py" not in result.stdout


@pytest.mark.parametrize("size", ["0_6b", "1_3b", "5b"])
@pytest.mark.parametrize("history", [0, 1, 2, 3])
def test_history_count_env_and_override(size, history):
    cfg = config(size, env_overrides={"HISTORY_NUM_TUBELETS": str(history)})
    assert cfg.model.video_backbone.encoder.history_num_tubelets == history
    assert cfg.dataloader.history_num_tubelets == history
    overridden = config(size, env_overrides={"HISTORY_NUM_TUBELETS": str(history)},
                        extra=("model.video_backbone.encoder.history_num_tubelets=2",))
    assert overridden.dataloader.history_num_tubelets == 2


@pytest.mark.parametrize("value", ["-1", "true", "1.5", "oops"])
def test_invalid_history_count_stops_before_training(value):
    result = launcher("1_3b", env_overrides={"HISTORY_NUM_TUBELETS": value})
    assert result.returncode == 2
    assert "HISTORY_NUM_TUBELETS" in result.stderr
    assert "scripts/train.py" not in result.stdout
