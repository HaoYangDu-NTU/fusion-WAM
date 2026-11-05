"""Resolve Haoyang training entries without torchrun or GPU allocation."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


def config(extra=(), environment=None, script_name="train_libero_vjepa_5b_dev_share_dir.sh"):
    path = ROOT / "scripts" / script_name
    subprocess.run(["bash", "-n", str(path)], check=True)
    script = path.read_text()
    env = os.environ.copy()
    for name in re.findall(r'^([A-Z_]+)="', script, re.MULTILINE):
        env.pop(name, None)
    env.pop("TEXT_EMBEDDING_DIR", None)
    env.update(environment or {})
    script = "\n".join(line for line in script.splitlines()
                       if not line.startswith(("source /data/fcy/basics/miniconda3", "conda activate ")))
    assert script.count("exec taskset -c ") == 1
    script = 'capture_command() { printf "%s\\0" "$@"; }\n' + script.replace(
        "exec taskset -c ", "capture_command -c ", 1)
    # Keep BASH_SOURCE's directory layout so repository-relative paths are real.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", dir=ROOT / "scripts") as captured:
        captured.write(script)
        captured.flush()
        result = subprocess.run(["bash", captured.name, *extra], env=env,
                                capture_output=True, text=True, check=True)
    args = result.stdout.rstrip("\0").split("\0")
    assert args[:3] == ["-c", "0-47,96-143", "/data/fcy/basics/miniconda3/envs/openwam/bin/python"]
    assert not any("/workspace/" in arg for arg in args)
    overrides = args[args.index("scripts/train.py") + 1:]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=overrides)


def test_5b_defaults_and_haoyang_paths():
    cfg = config()
    video = cfg.model.video_backbone
    enc = video.encoder
    arch = cfg.model.architecture
    assert video.name == "wan22_ti2v_5b"
    assert video.from_scratch is True
    assert video.model_path == "/data/fcy/model/wan2_2_openwam_native_from_diffusers"
    assert video.text_encoder_type == "rynnbrain1p1-2b"
    assert video.vlm_text_embedding_method == "avg_all_layers"
    assert enc.model_path == str(ROOT / "assets/vjepa")
    assert enc.history_num_tubelets == 1
    assert "add_one_history_context" not in enc
    assert enc.add_input_bottleneck is True and enc.bottleneck_dim == 256
    assert enc.use_avg_t_pooling is False
    assert enc.use_wan_style_patch_grid is True and enc.seperate_encode_view is True
    assert enc.condition_fusion_method == enc.future_fusion_method == "last_layer"
    assert arch.video_attention_mask_mode == "clean_condition_causal"
    assert arch.repa_enabled is False
    assert arch.video_prediction_type == arch.action_prediction_type == "x"
    assert arch.action_scale == 1.0
    assert video.shift_video == 42.0 and cfg.model.action_backbone.shift_action == 5.0
    assert cfg.training.num_epochs == 10 and cfg.training.batch_size == 64
    assert cfg.training.gradient_accumulation_steps == 1 and cfg.training.zero_stage == 1
    assert cfg.dataloader.dataset_dir == str(ROOT / "data/libero")
    assert str(cfg.training.output_path).startswith(str(ROOT / "outputs/libero_vjepa_5b_dev"))


def test_share_dir_preserves_env_and_hydra_overrides():
    cfg = config(extra=("training.batch_size=8", "model.video_backbone.encoder.bottleneck_dim=512"),
                 environment={"VJEPA_ENCODER": "L", "BOTTLENECK_DIM": "384"})
    assert cfg.training.batch_size == 8
    assert cfg.model.video_backbone.encoder.bottleneck_dim == 512
    assert cfg.model.video_backbone.encoder.model_path == str(ROOT / "assets/vjepa/vitl_dist_vitG_384")

def test_1_3b_defaults_follow_gpu2():
    cfg = config(script_name="train_libero_vjepa_1_3b_dev_share_dir.sh")
    video = cfg.model.video_backbone
    enc = video.encoder
    assert video.name == "wan22_style_1_3b"
    assert video.model_path == "/data/fcy/model/wan2_2_openwam_native_from_diffusers"
    assert video.text_encoder_type == "rynnbrain1p1-2b"
    assert video.vlm_text_embedding_method == "avg_all_layers"
    assert enc.history_num_tubelets == 1
    assert "add_one_history_context" not in enc
    assert enc.bottleneck_dim == 256
    assert enc.condition_fusion_method == enc.future_fusion_method == "last_layer"
    assert list(enc.fusion_layers) == [29, 39]
    assert cfg.model.architecture.video_attention_mask_mode == "clean_condition_causal"
    assert cfg.model.architecture.repa_enabled is False


@pytest.mark.parametrize("script_name", [
    "train_libero_vjepa_1_3b_dev_share_dir.sh",
    "train_libero_vjepa_5b_dev_share_dir.sh",
])
@pytest.mark.parametrize("history", [0, 2])
def test_share_dir_supports_new_history_interface(script_name, history):
    from openwam.history import condition_frame_offsets

    cfg = config(environment={"HISTORY_NUM_TUBELETS": str(history)}, script_name=script_name)
    assert cfg.model.video_backbone.encoder.history_num_tubelets == history
    assert condition_frame_offsets(history) == ([0] if history == 0 else [-12, -8, -4, 0])

@pytest.mark.parametrize("script_name", [
    "train_libero_vjepa_1_3b_dev_share_dir.sh",
    "train_libero_vjepa_5b_dev_share_dir.sh",
])
def test_share_dir_supports_new_text_and_fusion_options(script_name):
    cfg = config(environment={
        "TEXT_ENCODER_TYPE": "rynnbrain1p1-9b",
        "CONDITION_FUSION_METHOD": "adafuse",
        "FUTURE_FUSION_METHOD": "mean",
        "FUSION_LAYERS": "all_layers",
        "BOTTLENECK_DIM": "512",
    }, script_name=script_name)
    video = cfg.model.video_backbone
    assert video.text_encoder_type == "rynnbrain1p1-9b"
    assert video.encoder.condition_fusion_method == "adafuse"
    assert video.encoder.future_fusion_method == "mean"
    assert video.encoder.fusion_layers == "all_layers"
    assert video.encoder.bottleneck_dim == 512
