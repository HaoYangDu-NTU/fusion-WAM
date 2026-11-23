"""Standard LIBERO uses the saved concat model and the full benchmark protocol."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file, save_file

from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
from benchmarks.vla_harness.run_eval import SUITES, make_plan
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.model import build_architecture, resolve_architecture_config
from openwam.train.utils.checkpointing import save_config
from tests.test_vjepa_concat import METHOD, concat_source, encoder_source  # noqa: F401
from tests.test_wan22_style_capacity import launcher_config
from tests.test_wan_text_embedding_cache import make_cache


def checkpoint_fixture(tmp_path, source, layers=(1, 3)):
    cfg = launcher_config()
    vb = cfg.model.video_backbone
    vb.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in {
        "model_path": str(source), "add_input_bottleneck": True, "bottleneck_dim": 4,
        "condition_fusion_method": METHOD, "future_fusion_method": METHOD,
        "fusion_layers": layers if isinstance(layers, str) else list(layers),
        "use_avg_t_pooling": False, "history_num_tubelets": 1, "seperate_encode_view": True,
    }.items():
        vb.encoder[key] = value
    for key, value in {"dim": 96, "ffn_dim": 192, "num_heads": 4, "num_layers": 1, "freq_dim": 32}.items():
        vb.dit_config[key] = value
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    cfg.model.architecture.video_prediction_type = cfg.model.architecture.action_prediction_type = "x"
    cfg.model.architecture.repa_enabled = False
    # Match the BF16 cached-text deployment used by the real launcher.
    cfg.training.mixed_precision = "bf16"
    cfg.dataloader.normalize_mode = None
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.eval()
    checkpoint = tmp_path / "checkpoint"
    model.video_backbone.save_deploy_assets(str(checkpoint), cfg)
    # Deployment must reconstruct locally, without the original JEPA directory.
    cfg.model.video_backbone.encoder.model_path = "/unavailable/training/encoder"
    save_config(str(checkpoint), cfg)
    save_file(model.state_dict(), str(checkpoint / "model.safetensors"))
    return model, checkpoint, cfg


def libero_args(checkpoint):
    return SimpleNamespace(
        benchmark="libero", ckpt_dir=checkpoint, ckpt_name="model.safetensors",
        gpus="0,1,2,3", suites=",".join(SUITES), task_ids=None, smoke=False,
        workers=32, max_batch_size=8, max_wait_time=0.01, render="gpu", render_gpus="3",
        base_port=8990, output_dir=checkpoint / "evaluation", eval_id="concat-test",
    )


@pytest.mark.parametrize("layers", [[1, 3], "all_layers"])
def test_standard_libero_real_checkpoint_loading_and_batch_inference(tmp_path, concat_source, layers):
    original, checkpoint, cfg = checkpoint_fixture(tmp_path, concat_source, layers)
    channels = 16 if layers == [1, 3] else 32
    before = (checkpoint / "config.yaml").read_bytes()
    plan = make_plan(libero_args(checkpoint))
    assert len(plan["tasks"]) == 40 and plan["num_trials"] == 50
    assert {task["suite"] for task in plan["tasks"]} == set(SUITES)
    assert plan["categories"] is None and plan["correct_instruction_suffix"] is False
    assert len(plan["servers"]) == 4 and len(plan["workers"]) == 32
    assert plan["feature_fusion"]["latent_channels"] == channels
    assert plan["feature_fusion"]["source_norms"][-1] == "pretrained_final"
    assert all(str(checkpoint) in server["command"] for server in plan["servers"])

    saved, deployed = load_from_checkpoint_dir(str(checkpoint), "cpu", "model.safetensors")
    assert saved.model.video_backbone.text_embedding_dir == cfg.model.video_backbone.text_embedding_dir
    assert (checkpoint / "config.yaml").read_bytes() == before
    assert deployed.video_backbone.video_encoder.properties.z_dim == channels
    assert deployed.action_dim == original.action_dim == 80
    for name, value in original.state_dict().items():
        torch.testing.assert_close(deployed.state_dict()[name], value, rtol=0, atol=0)
    assert all(not parameter.is_meta for parameter in deployed.parameters())

    native = SimpleNamespace(
        architecture=deployed, _cfg_scale=1., _decode_video=False,
        _dit_cache=None, _prompt_embed_cache={},
        cfg=OmegaConf.create({"inference": {
            "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 5,
            "video_num_frames": 5, "height": 224, "width": 448,
        }}),
    )
    engine = BatchedOpenWAMEngine(native)
    image = Image.fromarray(np.random.default_rng(29).integers(0, 256, (224, 448, 3), dtype=np.uint8))
    observations = [dict(prompt=prompt, image=image, clean_condition_images=[image, image], state=np.zeros(10, np.float32))
                    for prompt in ("open drawer", "close drawer")]
    inputs, actions, _ = engine._prepare(observations)
    assert inputs["latents"].shape == (2, channels, 3, 14, 28)
    assert inputs["clean_condition_latents"].shape == (2, channels, 1, 14, 28)
    assert actions.shape == (2, 4, 80)
    expected = deployed.video_backbone.video_encoder.batch_encode(
        deployed.video_backbone.video_encoder.preprocess_video_batch([[image, image], [image, image]]))
    torch.testing.assert_close(inputs["latents"][:, :, :1], expected)
    chunks = engine.generate_batch(observations)
    assert len(chunks) == 2 and all(chunk.shape == (4, 10) and np.isfinite(chunk).all() for chunk in chunks)
    assert engine.last_stats["forward_batch_sizes"] == [2, 2]


@pytest.mark.parametrize("bad", ["layers", "norm", "branch", "repa", "projection", "head", "feature_norm"])
def test_standard_libero_rejects_concat_drift_before_launch(tmp_path, concat_source, bad):
    _, checkpoint, _ = checkpoint_fixture(tmp_path, concat_source)
    config_path, manifest_path = checkpoint / "config.yaml", checkpoint / "manifest.json"
    cfg, manifest = yaml.safe_load(config_path.read_text()), json.loads(manifest_path.read_text())
    if bad == "layers":
        cfg["model"]["video_backbone"]["encoder"]["fusion_layers"] = [0, 3]
    elif bad == "norm":
        manifest["feature_fusion"]["source_norms"][-1] = "ln0"
    elif bad == "branch":
        cfg["model"]["video_backbone"]["encoder"]["future_fusion_method"] = "last_layer"
    elif bad == "repa":
        cfg["model"]["architecture"]["repa_enabled"] = True
    else:
        weights = load_file(str(checkpoint / "model.safetensors"))
        key = {"projection": "video_backbone.dit.patch_embedding.clean_proj.weight",
               "head": "video_backbone.dit.head.head.weight",
               "feature_norm": "video_backbone.video_encoder.feature_norm.weight"}[bad]
        weights[key] = weights[key][:-1]
        save_file(weights, str(checkpoint / "model.safetensors"))
    config_path.write_text(yaml.safe_dump(cfg))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        make_plan(libero_args(checkpoint))
    assert not (checkpoint / "evaluation").exists()


def test_non_concat_libero_plan_keeps_existing_behavior(tmp_path):
    (tmp_path / "config.yaml").write_text("{}")
    plan = make_plan(libero_args(tmp_path))
    assert "feature_fusion" not in plan
    assert len(plan["tasks"]) * plan["num_trials"] == 2000
