"""Native Wan video encoding: temporal dependence, no future leakage and deployment."""
import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.history import select_condition_images, validate_condition_frame_offsets
from openwam.model.video_backbone.encoder.wan21_features import Wan21VideoFeatureEncoder, dit_config
from openwam.model.video_backbone.wan.models.dit import WanModel
from tests.dataloader.test_libero import native_libero_bucket


CONFIG = dict(model_type="i2v", in_dim=36, dim=32, ffn_dim=64, freq_dim=32,
              num_heads=1, num_layers=2, eps=1e-6)
OPTIONS = dict(feature_layer=2, microbatch_size=1, bottleneck_dim=8, feature_norm_eps=1e-5)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 16, 1)

    def batch_encode(self, video, device):
        video = self.proj(video)
        video = torch.nn.functional.avg_pool3d(video.float(), (1, 8, 8)).to(video.dtype)
        first = video[:, :, :1]
        if video.shape[2] > 1:
            rest = video[:, :, 1:].unflatten(2, (-1, 4)).mean(3)
            return torch.cat([first, rest], dim=2)
        return first


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 1280)

    def encode_image(self, images):
        return self.proj(images[0].mean((-1, -2))).unsqueeze(1).expand(-1, 257, -1)


def make_encoder():
    torch.manual_seed(7)
    encoder = Wan21VideoFeatureEncoder(WanModel(**dit_config(CONFIG)), TinyVAE(), TinyCLIP(),
                                      torch.zeros(1, 512, 4096), **OPTIONS)
    encoder._source_config = dict(CONFIG)
    return encoder


def frames(count=37):
    rng = np.random.default_rng(42)
    return [Image.fromarray(rng.integers(0, 256, (224, 448, 3), dtype=np.uint8)) for _ in range(count)]


def test_video_features_are_temporal_and_clean_never_reads_future():
    encoder = make_encoder()
    video = torch.randn(1, 3, 37, 224, 448)
    original = encoder.batch_encode(video)
    assert original.shape == (1, 32, 9, 14, 28)
    changed = video.clone()
    changed[:, :, 5:] = torch.randn_like(changed[:, :, 5:])
    modified = encoder.batch_encode(changed)
    torch.testing.assert_close(original[:, :, :1], modified[:, :, :1], rtol=0, atol=0)
    assert not torch.allclose(original[:, :, 1:], modified[:, :, 1:])
    torch.testing.assert_close(original[:, :, :1], encoder.batch_encode(video[:, :, :5]), rtol=0, atol=0)
    # Native DiT self-attention lets a late future change an earlier target.
    late = video.clone()
    late[:, :, -4:] += 2
    assert not torch.allclose(original[:, :, 1], encoder.batch_encode(late)[:, :, 1])
    history = video.clone()
    history[:, :, :4] += 2
    assert not torch.allclose(original[:, :, :1], encoder.batch_encode(history)[:, :, :1])


def test_independent_views_microbatch_and_frozen_encoder():
    encoder = make_encoder()
    video = torch.randn(2, 3, 5, 224, 448)
    original = encoder.batch_encode(video)
    modified = video.clone()
    modified[..., 224:] += 1
    changed = encoder.batch_encode(modified)
    torch.testing.assert_close(original[..., :14], changed[..., :14], rtol=0, atol=0)
    assert not torch.allclose(original[..., 14:], changed[..., 14:])
    encoder.microbatch_size = 4
    torch.testing.assert_close(original, encoder.batch_encode(video), rtol=1e-4, atol=1e-5)
    encoder.train()
    assert not encoder.training and all(not module.training for module in encoder.modules())
    assert not original.requires_grad and all(not p.requires_grad for p in encoder.parameters())
    projection = encoder.build_dit_input_proj(32)
    projection(original, num_clean_condition_latents=1).square().mean().backward()
    assert projection.clean_proj.weight.grad is not None
    assert all(p.grad is None for p in encoder.parameters())


def test_native_geometry_rejects_invalid_history_and_sampling():
    encoder = make_encoder()
    with pytest.raises(ValueError, match="five history"):
        encoder.batch_encode(torch.zeros(1, 3, 10, 224, 448))
    assert select_condition_images(deque(["first"]), frame_offsets=(-4, -3, -2, -1, 0)) == ["first"] * 5
    assert select_condition_images(deque(range(8)), frame_offsets=(-4, -3, -2, -1, 0)) == [3, 4, 5, 6, 7]
    for invalid in ([0, -1], [-1, 1], [-1, -1, 0], [False, 0]):
        with pytest.raises(ValueError):
            validate_condition_frame_offsets(invalid)


def test_checkpoint_local_skeleton_strict_reload(tmp_path, monkeypatch):
    encoder = make_encoder()
    encoder.save_deploy_assets(tmp_path, None)
    monkeypatch.setattr("openwam.model.video_backbone.wan.models.vae.WanVideoVAE", TinyVAE)
    monkeypatch.setattr("openwam.model.video_backbone.wan.models.image_encoder.WanImageEncoder", TinyCLIP)
    cfg = {**OPTIONS, "model_path": "/unavailable/source"}
    reloaded = Wan21VideoFeatureEncoder.from_skeleton({}, encoder_cfg=cfg, ckpt_dir=tmp_path).float()
    reloaded.load_state_dict(encoder.state_dict(), strict=True)
    pixels = torch.randn(1, 3, 5, 224, 448)
    torch.testing.assert_close(encoder.batch_encode(pixels), reloaded.float().batch_encode(pixels))
    with pytest.raises(ValueError, match="feature_layer"):
        Wan21VideoFeatureEncoder.from_skeleton({}, encoder_cfg={**cfg, "feature_layer": 1}, ckpt_dir=tmp_path)


@pytest.mark.parametrize("experiment,version,channels,tokens", [("E03", "2.1", 16, 3528), ("E04", "2.2", 48, 882)])
def test_native_vae_experiment_profiles(experiment, version, channels, tokens):
    from openwam.experiments import resolve_experiment
    cfg, profile, _ = resolve_experiment(experiment)
    assert cfg.dataloader.video_stride == 1
    assert cfg.dataloader.clean_condition_frame_offsets is None
    assert cfg.dataloader.history_num_tubelets == 0
    assert cfg.dataloader.num_frames == 33
    assert cfg.model.video_backbone.encoder.vae_version == version
    assert profile["geometry"]["channels"] == channels
    assert profile["geometry"]["tokens"] == tokens


@pytest.fixture
def small_model(tmp_path, monkeypatch):
    from tests.test_wan22_style_capacity import launcher_config
    from tests.test_wan_text_embedding_cache import make_cache
    from openwam.model import build_architecture, resolve_architecture_config
    cfg = launcher_config()
    cfg.model.video_backbone.encoder = OmegaConf.create(dict(name="wan21_features", model_path="unused",
                                                             empty_context_path="unused", seperate_encode_view=True,
                                                             add_input_bottleneck=True, **OPTIONS))
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        cfg.model.video_backbone.dit_config[key] = value
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    cfg.model.architecture.video_prediction_type = cfg.model.architecture.action_prediction_type = "x"
    cfg.model.architecture.repa_enabled = False
    cfg.model.architecture.attention_mask_mode = "mutual"
    cfg.dataloader.normalize_mode = None
    monkeypatch.setattr(Wan21VideoFeatureEncoder, "from_pretrained", lambda *a, **k: make_encoder())
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.eval()
    return model, cfg


def test_m29_training_loss_backward_and_future_padding(small_model):
    model, _ = small_model
    model.init_training_schedulers()
    clip = frames()
    sample = dict(video=clip[4:], clean_condition_images=clip[:5], prompt="open drawer",
                  action=torch.zeros(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                  proprio=torch.zeros(80), video_mask=torch.tensor([True] * 29 + [False] * 4))
    inputs = model.prepare_inputs([sample])
    assert inputs["input_latents"].shape == (1, 32, 9, 14, 28)
    assert inputs["num_clean_condition_latents"] == 1
    assert inputs["video_is_pad"].tolist() == [[False] * 7 + [True]]
    losses = model.compute_loss(**inputs)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.video_backbone.dit.patch_embedding.proj1.weight.grad is not None
    assert all(p.grad is None for p in model.video_backbone.video_encoder.parameters())
    pred = torch.zeros_like(inputs["input_latents"])
    target = torch.zeros_like(pred)
    pred[:, :, :1] = 1000
    pred[:, :, -1:] = 1000
    loss = model._compute_video_loss(pred, target, torch.tensor([500]), inputs, torch.device("cpu"))
    assert loss.item() == 0


def test_plus_relocation_and_eval_checkpoint_contract(small_model, tmp_path, monkeypatch):
    from openwam.train.utils.checkpointing import save_config
    from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
    from benchmarks.vla_harness.checkpoint_contract import validate_wan21_features_checkpoint
    model, cfg = small_model
    run = tmp_path / "run"
    run.mkdir()
    model.video_backbone.save_deploy_assets(str(run), cfg)
    save_config(str(run), cfg)
    np.save(run / "normalization_stats.npy", {})
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(run / "checkpoint.safetensors"))
    loaded_cfg = OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True)
    result = validate_wan21_features_checkpoint(run, "checkpoint.safetensors", loaded_cfg)
    assert result["latent_channels"] == 32
    cache = tmp_path / "plus_cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"version": 1}))
    plus = prepare_checkpoint(run, "checkpoint.safetensors", tmp_path / "plus", cache)
    assert (plus / "wan21_features/encoder.json").is_file()
    assert validate_wan21_features_checkpoint(plus, "checkpoint.safetensors", loaded_cfg)
    # Reconstruct every encoder tensor from saved sidecars/weights only.
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
    monkeypatch.setattr("openwam.model.video_backbone.wan.models.vae.WanVideoVAE", TinyVAE)
    monkeypatch.setattr("openwam.model.video_backbone.wan.models.image_encoder.WanImageEncoder", TinyCLIP)
    _, deployed = load_from_checkpoint_dir(str(run), "cpu", "checkpoint.safetensors")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(deployed.state_dict()[key], value, rtol=0, atol=0)
    clip = frames(5)
    native = SimpleNamespace(architecture=deployed, _cfg_scale=1., _decode_video=False,
                             _dit_cache=None, _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {
                                 "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 33,
                                 "video_num_frames": 33, "height": 224, "width": 448}}))
    engine = BatchedOpenWAMEngine(native)
    observations = [dict(prompt="open drawer", image=clip[-1], clean_condition_images=clip, state=np.zeros(10, np.float32))]
    batch, actions, _ = engine._prepare(observations)
    ordinary = deployed.video_backbone.preprocess_input_for_inference(
        prompt="open drawer", clean_condition_images=clip, num_frames=33,
        height=224, width=448, seed=42, num_inference_steps=2)
    assert batch["latents"].shape == ordinary["latents"].shape == (1, 32, 9, 14, 28)
    torch.testing.assert_close(batch["clean_condition_latents"], ordinary["clean_condition_latents"])
    chunks = engine.generate_batch(observations)
    assert chunks[0].shape == (32, 10) and np.isfinite(chunks[0]).all()


@pytest.mark.parametrize("step", [0, 1, 4, 12, 39])
def test_reader_returns_dense_past_and_future_without_changing_action_window(native_libero_bucket, monkeypatch, step):
    from openwam.dataloader.libero import LiberoDataset

    def decode(path, indices, height, width, **kwargs):
        return [Image.new("RGB", (width, height), (int(index),) * 3) for index in indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    config = dict(dataset_dir=str(native_libero_bucket), normalize_mode=None, multiview=True,
                  seperate_encode_view=True, num_frames=33, video_stride=1, history_num_tubelets=0)
    baseline = LiberoDataset(**config)
    dataset = LiberoDataset(**config, clean_condition_frame_offsets=[-4, -3, -2, -1, 0])
    sample = dataset[step]
    assert len(sample["video"]) == 33
    assert [image.getpixel((10, 10))[0] for image in sample["clean_condition_images"]] == [max(0, step + delta) for delta in [-4, -3, -2, -1, 0]]
    expected = baseline[step]
    for key in ("action", "action_mask", "proprio", "video_mask"):
        torch.testing.assert_close(sample[key], expected[key], rtol=0, atol=0)
    assert [image.getpixel((10, 10)) for image in sample["video"]] == [image.getpixel((10, 10)) for image in expected["video"]]


def test_policy_buffers_dense_control_step_history():
    from openwam.deploy.policy import WAMPolicy
    references = []

    def generate(conditions):
        references.append(conditions["clean_condition_images"])
        return {"actions": np.zeros((6, 7), dtype=np.float32)}

    encoder = SimpleNamespace(clean_condition_frame_offsets=(-4, -3, -2, -1, 0))
    engine = SimpleNamespace(generate=generate, architecture=SimpleNamespace(video_backbone=SimpleNamespace(video_encoder=encoder)))
    policy = WAMPolicy(engine, SimpleNamespace(), execution_config={"mode": "sync", "inference_horizon": 6})
    for step in range(13):
        policy.predict_action({"image": step, "prompt": "test"})
    assert references == [[max(0, step + offset) for offset in encoder.clean_condition_frame_offsets] for step in (0, 6, 12)]
    policy.reset()
    policy.predict_action({"image": 100})
    assert references[-1] == [100] * 5
