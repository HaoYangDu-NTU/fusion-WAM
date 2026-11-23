"""Native-frame DINOv3: real HF encoding, training, checkpoint and both deploy paths."""

import json
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from PIL import Image
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
from benchmarks.vla_harness.run_eval import make_plan
from benchmarks.vla_harness.run_train_eval import training_checkpoint
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.model import build_architecture, resolve_architecture_config
from openwam.model.architectures.dual_system.mot_driver import _CleanConditionAttention
from openwam.model.architectures.utils.mask_modes import build_cross_modal_attention_mask
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d
from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
from openwam.train.utils.checkpointing import save_config
from tests.benchmarks.vla_harness.test_vla_harness_concat import libero_args
from tests.test_wan22_style_capacity import launcher_config
from tests.test_wan_text_embedding_cache import make_cache


OPTIONS = dict(use_avg_t_pooling=False, add_one_history_context=True,
               seperate_encode_view=True, add_input_bottleneck=True,
               bottleneck_dim=8, feature_norm_eps=1e-5)


@pytest.fixture
def dino_source(tmp_path):
    config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=4, num_register_tokens=4, patch_size=16)
    source = tmp_path / "dino-source"
    DINOv3ViTModel(config).save_pretrained(source)
    return source


@pytest.fixture
def small_model(tmp_path, dino_source):
    cfg = launcher_config()
    vb = cfg.model.video_backbone
    vb.encoder = OmegaConf.create(dict(name="dinov3", model_path=str(dino_source), **OPTIONS))
    vb.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        vb.dit_config[key] = value
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    cfg.model.architecture.video_prediction_type = cfg.model.architecture.action_prediction_type = "x"
    cfg.model.architecture.repa_enabled = False
    cfg.model.architecture.attention_mask_mode = "mutual"
    cfg.training.mixed_precision = "bf16"
    cfg.dataloader.normalize_mode = None
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.eval()
    return model, cfg


def frames(count=10):
    rng = np.random.default_rng(29)
    return [Image.fromarray(rng.integers(0, 256, (224, 448, 3), dtype=np.uint8)) for _ in range(count)]


def test_native_frames_and_views_match_individual_hf_encoding(dino_source):
    encoder = DinoV3VideoEncoder.from_pretrained(str(dino_source), **OPTIONS)
    video = encoder.preprocess_video_batch([frames()])
    with torch.no_grad():
        actual = encoder.batch_encode(video)
        expected = []
        for t in range(10):
            views = []
            for v in range(2):
                image = video[:, :, t, :, v * 224:(v + 1) * 224]
                tokens = encoder._m(image).last_hidden_state[:, 5:]
                # Native pretrained LN is already in last_hidden_state.
                tokens = torch.nn.functional.layer_norm(tokens, (32,), eps=1e-5)
                views.append(tokens.reshape(1, 14, 14, 32).permute(0, 3, 1, 2))
            expected.append(torch.cat(views, dim=-1))
        torch.testing.assert_close(actual, torch.stack(expected, dim=2))
        ref = encoder.batch_encode(video[:, :, :2])
        torch.testing.assert_close(actual[:, :, :2], ref)
        modified = video.clone()
        modified[:, :, 7, :, 224:] = 0
        changed = encoder.batch_encode(modified)
        torch.testing.assert_close(actual[:, :, :, :, :14], changed[:, :, :, :, :14])
        torch.testing.assert_close(actual[:, :, :7], changed[:, :, :7])
        assert not torch.equal(actual[:, :, 7, :, 14:], changed[:, :, 7, :, 14:])
    assert actual.shape == (1, 32, 10, 14, 28)
    assert encoder.properties.temporal_compression == 1
    assert encoder.properties.num_condition_latents == 2


def test_training_keeps_both_clean_frames_and_supervises_only_future(small_model, monkeypatch):
    model, _ = small_model
    model.init_training_schedulers()
    clip = frames()
    sample = dict(video=clip[1:], clean_condition_images=clip[:2], prompt="open drawer",
                  action=torch.zeros(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                  proprio=torch.zeros(80), video_mask=torch.tensor([True] * 7 + [False] * 2))
    inputs = model.prepare_inputs([sample])
    assert inputs["input_latents"].shape == (1, 32, 10, 14, 28)
    assert inputs["clean_condition_latents"].shape[2] == inputs["num_clean_condition_latents"] == 2
    assert inputs["video_is_pad"].tolist() == [[False] * 6 + [True] * 2]
    seen = {}
    original = model.forward
    monkeypatch.setattr("openwam.model.architectures.dual_system.mot_driver._CLEAN_CONDITION_SPLIT_MIN_TOKENS", 0)
    driver = model.mot_driver
    original_attention = driver._mixed_attention
    partitions = []
    def attention(q, k, v, mask):
        partitions.append(mask.tokens)
        return original_attention(q, k, v, mask)
    monkeypatch.setattr(driver, "_mixed_attention", attention)

    def forward(*args, **kwargs):
        seen["noisy"] = kwargs["latents"].detach().clone()
        return original(*args, **kwargs)

    projection = model.video_backbone.dit.patch_embedding
    def capture(name):
        def hook(module, args):
            seen[name] = args[0].shape[2]
        return hook
    hooks = [projection.clean_proj.register_forward_pre_hook(capture("clean")),
             projection.proj1.register_forward_pre_hook(capture("future"))]
    try:
        with patch.object(model, "forward", side_effect=forward):
            losses = model.compute_loss(**inputs)
        losses["loss"].backward()
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(seen["noisy"][:, :, :2], inputs["clean_condition_latents"])
    assert (seen["clean"], seen["future"]) == (2, 8)
    assert partitions and all(n == 196 for n in partitions)
    assert torch.isfinite(losses["loss"])
    for module in (projection.clean_proj, projection.proj1, projection.proj2):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
    pred = torch.zeros_like(inputs["input_latents"])
    target = torch.zeros_like(pred)
    pred[:, :, :2] = 1000
    pred[:, :, -2:] = 1000
    loss = model._compute_video_loss(pred, target, torch.tensor([500]), inputs, torch.device("cpu"))
    assert loss.item() == 0  # neither clean frames nor padded future contribute


def test_two_frame_attention_dense_and_optimized_agree(small_model):
    model, _ = small_model
    vb = model.video_backbone
    mask = build_cross_modal_attention_mask(vb, s_video=980, s_action=32,
                                           video_tokens_per_frame=98, mode="mutual", device="cpu",
                                           num_clean_condition_latents=2)
    assert mask[:196, :196].all() and not mask[:196, 196:].any()
    assert mask[196:].all()
    # Both clean frames get t=0 even when the eight future frames get t=500.
    state = vb.prepare(latents=torch.randn(1, 32, 10, 14, 28, dtype=torch.bfloat16),
                       timestep=torch.tensor([500.], dtype=torch.bfloat16),
                       context=torch.randn(1, 128, 4096, dtype=torch.bfloat16),
                       fuse_vae_embedding_in_latents=True, num_clean_condition_latents=2)
    times = torch.tensor([0., 500.], dtype=torch.bfloat16)
    expected = vb.dit.time_embedding(sinusoidal_embedding_1d(vb.dit.freq_dim, times).to(torch.bfloat16))
    torch.testing.assert_close(state.extras["time_embed"][:, :196], expected[:1].expand(1, 196, -1))
    torch.testing.assert_close(state.extras["time_embed"][:, 196:], expected[1:].expand(1, 784, -1))
    driver = model.build_mot_driver()
    q, k, v = [torch.randn(1, 1012, 96) for _ in range(3)]
    dense = driver._mixed_attention(q, k, v, mask)
    split = driver._mixed_attention(q, k, v, _CleanConditionAttention(196))
    torch.testing.assert_close(dense, split, atol=1e-6, rtol=1e-5)


def test_checkpoint_libero_plus_and_native_batch_inference(tmp_path, small_model):
    model, cfg = small_model
    run = tmp_path / "checkpoint"
    model.video_backbone.save_deploy_assets(str(run), cfg)
    cfg.model.video_backbone.encoder.model_path = "/unavailable/training/dino"
    save_config(str(run), cfg)
    name = "checkpoint_step_10690.safetensors"
    save_file(model.state_dict(), str(run / name))
    np.save(run / "normalization_stats.npy", {})
    args = libero_args(run)
    args.ckpt_name = name
    plan = make_plan(args)
    assert len(plan["tasks"]) == 40 and plan["num_trials"] == 50
    assert plan["visual_encoder"]["num_condition_latents"] == 2
    assert plan["visual_encoder"]["temporal_compression"] == 1
    assert plan["correct_instruction_suffix"] is False
    # The train->eval gate uses DINO sidecars, never a JEPA manifest.
    train = tmp_path / "queue/training"
    train.mkdir(parents=True)
    (train / "exit.json").write_text(json.dumps(dict(exit_code=0)))
    (train / "job.log").write_text(f"Training: 100% 10690/10690\n[checkpoint] Saved final: {run / name}\n")
    assert training_checkpoint(train.parent)["run"] == str(run)
    _, deployed = load_from_checkpoint_dir(str(run), "cpu", name)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(deployed.state_dict()[key], value, rtol=0, atol=0)
    clip = frames(2)
    native = SimpleNamespace(architecture=deployed, _cfg_scale=1., _decode_video=False,
                             _dit_cache=None, _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {
                                 "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 33,
                                 "video_num_frames": 9, "height": 224, "width": 448}}))
    engine = BatchedOpenWAMEngine(native)
    observations = [dict(prompt="open drawer", image=clip[1], history_image=clip[0], state=np.zeros(10, np.float32))]
    batch, actions, _ = engine._prepare(observations)
    ordinary = deployed.video_backbone.preprocess_input_for_inference(
        prompt="open drawer", clean_condition_images=clip, num_frames=9,
        height=224, width=448, seed=42, num_inference_steps=2)
    assert batch["latents"].shape == ordinary["latents"].shape == (1, 32, 10, 14, 28)
    assert batch["num_clean_condition_latents"] == ordinary["num_clean_condition_latents"] == 2
    assert actions.shape == (1, 32, 80)
    torch.testing.assert_close(batch["clean_condition_latents"], ordinary["clean_condition_latents"])
    clean_seen = []
    original = deployed.forward
    def record_clean(*args, **kwargs):
        clean_seen.append(kwargs["latents"][:, :, :2].clone())
        return original(*args, **kwargs)
    with patch.object(deployed, "forward", side_effect=record_clean):
        chunks = engine.generate_batch(observations)
    assert len(clean_seen) == 2
    for clean in clean_seen:
        torch.testing.assert_close(clean, batch["clean_condition_latents"])
    assert len(chunks) == 1 and chunks[0].shape == (32, 10) and np.isfinite(chunks[0]).all()
    # Plus relocates the same encoder, while swapping only the text cache.
    plus = prepare_checkpoint(run, name, tmp_path / "plus", cfg.model.video_backbone.text_embedding_dir)
    assert (plus / "dinov3/config.json").is_file() and (plus / "dinov3/encoder.json").is_file()
    _, plus_model = load_from_checkpoint_dir(str(plus), "cpu", name)
    assert plus_model.video_backbone.num_condition_latents == 2
    # Same-shaped processing drift must fail before any evaluation starts.
    processing = json.loads((run / "dinov3/encoder.json").read_text())
    processing["use_avg_t_pooling"] = True
    (run / "dinov3/encoder.json").write_text(json.dumps(processing))
    with pytest.raises(ValueError, match="processing config"):
        make_plan(args)


