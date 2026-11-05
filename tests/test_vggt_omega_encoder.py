"""E05 uses native tokens internally and only spatial outputs in the E01 path."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.experiments import resolve_experiment
from openwam.model.video_backbone.encoder.vggt_omega import VGGTOmegaSpatialEncoder


class TinyAggregator(nn.Module):
    """Distinct spatial/view/channel information with joint-view interaction."""
    patch_size = 16
    patch_token_start = 17

    def __init__(self):
        super().__init__()
        self.frame_blocks = nn.ModuleList([nn.ModuleDict({"norm1": nn.LayerNorm(8)})])
        # Expose official-style norm1 attribute used by the real wrapper.
        self.frame_blocks[0].norm1 = self.frame_blocks[0]["norm1"]
        self.inter_frame_blocks = nn.ModuleList([nn.Linear(8, 8)])
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 8))
        self.register_token = nn.Parameter(torch.zeros(1, 2, 16, 8))
        self.patch_proj = nn.Linear(3, 8)
        self.calls = []

    def forward(self, images):
        self.calls.append(tuple(images.shape))
        b, v, c, h, w = images.shape
        pooled = torch.nn.functional.avg_pool2d(images.reshape(b*v, c, h, w).float(), 16)
        local = self.patch_proj(pooled.to(images.dtype).flatten(2).transpose(1, 2)).reshape(b, v, -1, 8)
        relation = local + local.flip(1).mean(2, keepdim=True)
        patches = torch.cat([local, relation], dim=-1)
        prefix = torch.full((b, v, 17, 16), -100, dtype=patches.dtype, device=patches.device)
        return [torch.cat([prefix, patches], dim=2)], 17


def frames():
    generator = np.random.default_rng(42)
    return [Image.fromarray(generator.integers(0, 256, (224, 448, 3), dtype=np.uint8)) for _ in range(10)]


def test_time_is_batch_stereo_is_native_sequence_and_spatial_order():
    enc = VGGTOmegaSpatialEncoder(TinyAggregator(), pair_microbatch_size=1)
    video = enc.preprocess_video(frames())
    assert video.shape == (1, 3, 10, 224, 448) and video.min() >= 0 and video.max() <= 1
    z = enc.batch_encode(video)
    assert z.shape == (1, 16, 10, 14, 28)
    assert enc._m.calls == [(1, 2, 3, 224, 224)] * 10
    pair = torch.stack(video[0, :, 0].split(224, dim=-1))
    layers, start = enc._m(pair.unsqueeze(0))
    expected = enc._out_norm(layers[-1][:, :, start:]).reshape(1, 2, 14, 14, 16)
    expected = expected.permute(0, 4, 2, 1, 3).reshape(1, 16, 14, 28)
    torch.testing.assert_close(z[:, :, 0], expected, rtol=0, atol=0)
    assert (z[:, :, :, :, 1:] - z[:, :, :, :, :-1]).abs().sum() > 0
    assert not z.requires_grad
    enc.train()
    assert all(not module.training for module in enc.modules())
    assert all(not parameter.requires_grad for parameter in enc.parameters())


def test_no_future_or_cross_time_information_in_features():
    enc = VGGTOmegaSpatialEncoder(TinyAggregator())
    video = enc.preprocess_video(frames())
    before = enc.batch_encode(video)
    video[:, :, 2:] = torch.rand_like(video[:, :, 2:])
    after = enc.batch_encode(video)
    torch.testing.assert_close(before[:, :, :2], after[:, :, :2], rtol=0, atol=0)
    assert not torch.equal(before[:, :, 2:], after[:, :, 2:])
    video[:, :, :1] = torch.rand_like(video[:, :, :1])
    last = enc.batch_encode(video)
    torch.testing.assert_close(after[:, :, 1:], last[:, :, 1:], rtol=0, atol=0)


def test_stereo_cache_reuses_content_and_keeps_view_order(tmp_path):
    from openwam.model.video_backbone.encoder.vggt_feature_cache import StereoFeatureCache
    enc = VGGTOmegaSpatialEncoder(TinyAggregator()).to(torch.bfloat16)
    pixels = enc.preprocess_video(frames()[:2])
    online = enc.batch_encode(pixels)
    identity = dict(weights="test-checkpoint", processing=enc.processing_config)
    enc.feature_cache = StereoFeatureCache(tmp_path, identity)
    cached = enc.batch_encode(pixels)
    assert enc.feature_cache.misses == 2
    torch.testing.assert_close(cached, online, rtol=0, atol=0)
    enc._m.calls.clear()
    # An overlapping window may start at a previous window's future position.
    repeated = enc.batch_encode(pixels[:, :, 1:])
    torch.testing.assert_close(repeated, online[:, :, 1:], rtol=0, atol=0)
    assert enc.feature_cache.hits == 1 and enc._m.calls == []
    swapped = torch.cat(pixels.split(224, dim=-1)[::-1], dim=-1)
    enc.batch_encode(swapped)
    assert enc.feature_cache.misses == 4
    changed = StereoFeatureCache(tmp_path, {**identity, "weights": "different"})
    assert changed.directory != enc.feature_cache.directory


def test_optional_resolution_preserves_spatial_grid():
    enc = VGGTOmegaSpatialEncoder(TinyAggregator(), image_size=512)
    packed = frames()[0].resize((1024, 512))
    features = enc.batch_encode(enc.preprocess_video([packed]))
    assert features.shape == (1, 16, 1, 32, 64)
    projection = enc.build_dit_input_proj(96)
    assert projection(features, num_clean_condition_latents=1).shape == (1, 96, 1, 16, 32)


def _cache_process_worker(directory):
    from pathlib import Path
    from openwam.model.video_backbone.encoder.vggt_feature_cache import StereoFeatureCache
    cache = StereoFeatureCache(directory, {"concurrency_test": True})
    def encode(pair):
        with (Path(directory) / "encode_calls.txt").open("a") as handle:
            handle.write("encoded\n")
        return pair + 1
    return float(cache.get_or_encode(torch.zeros(2, dtype=torch.bfloat16), encode).sum())


def test_cache_concurrent_processes_publish_one_complete_tensor(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_cache_process_worker, [str(tmp_path)] * 2))
    assert results == [2.0, 2.0]
    assert (tmp_path / "encode_calls.txt").read_text().splitlines() == ["encoded"]
    assert len(list(tmp_path.rglob("*.safetensors"))) == 1


def test_e05_matches_e01_protocol_and_exposes_e02_differences():
    first, p1, _ = resolve_experiment("E01")
    second, p2, _ = resolve_experiment("E02")
    cfg, profile, _ = resolve_experiment("E05")
    assert first.dataloader == second.dataloader
    for key in first.dataloader:
        if key not in ("history_num_tubelets", "clean_condition_frame_offsets"):
            assert cfg.dataloader[key] == first.dataloader[key], key
    assert cfg.dataloader.history_num_tubelets == 1
    assert cfg.dataloader == first.dataloader
    assert cfg.model.architecture == first.model.architecture == second.model.architecture
    assert cfg.model.action_backbone == first.model.action_backbone == second.model.action_backbone
    assert cfg.model.video_backbone.dit_config == first.model.video_backbone.dit_config
    for key in first.training:
        if key != "output_path":
            assert cfg.training[key] == first.training[key] == second.training[key], key
    assert profile["geometry"]["tokens"] == 980
    assert p1["geometry"]["tokens"] == 980
    assert p2["geometry"]["tokens"] == 490
    assert cfg.model.video_backbone.encoder.bottleneck_dim == 256
    assert cfg.model.video_backbone.encoder.feature_norm_eps == first.model.video_backbone.encoder.feature_norm_eps
    assert cfg.model.video_backbone.text_encoder_type == first.model.video_backbone.text_encoder_type


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    from tests.test_wan22_style_capacity import launcher_config
    from tests.test_wan_text_embedding_cache import make_cache
    from openwam.model import build_architecture, resolve_architecture_config
    from openwam.model.video_backbone.encoder import vggt_omega
    monkeypatch.setattr(vggt_omega, "_build_aggregator", TinyAggregator)
    monkeypatch.setattr(VGGTOmegaSpatialEncoder, "from_pretrained", classmethod(
        lambda cls, *a, **kw: cls(TinyAggregator(), bottleneck_dim=8)))
    cfg = launcher_config()
    cfg.model.video_backbone.encoder = OmegaConf.create(dict(
        name="vggt_omega", model_path="unused", pair_microbatch_size=1, bottleneck_dim=8,
        feature_norm_eps=1e-5, seperate_encode_view=True, add_one_history_context=True,
        add_input_bottleneck=True))
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for k, v in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        cfg.model.video_backbone.dit_config[k] = v
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    cfg.model.architecture.video_prediction_type = cfg.model.architecture.action_prediction_type = "x"
    cfg.model.architecture.repa_enabled = False
    cfg.model.architecture.attention_mask_mode = "mutual"
    cfg.dataloader.normalize_mode = None
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.init_training_schedulers()
    return model, cfg


def test_m29_targets_projection_gradients_and_native_deploy(tiny_model, tmp_path):
    from openwam.train.utils.checkpointing import save_config
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from benchmarks.vla_harness.checkpoint_contract import validate_vggt_omega_checkpoint
    from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint
    model, cfg = tiny_model
    images = frames()
    clip = images[1:]
    sample = dict(video=clip, clean_condition_images=images[:2], prompt="open drawer",
                  action=torch.zeros(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                  proprio=torch.zeros(80), video_mask=torch.ones(9, dtype=torch.bool))
    inputs = model.prepare_inputs([sample])
    assert inputs["input_latents"].shape == (1, 16, 10, 14, 28)
    assert inputs["num_clean_condition_latents"] == 2
    assert inputs["video_is_pad"].shape[-1] == 8
    target = inputs["input_latents"].clone()
    projection = model.video_backbone.dit.patch_embedding
    grid = projection(target, num_clean_condition_latents=2)
    assert grid.shape == (1, 96, 10, 7, 14) and grid.shape[2:].numel() == 980
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
                                 betas=(0.9, 0.95), weight_decay=0.01)
    before = projection.clean_proj.weight.detach().clone()
    loss = model.compute_loss(**inputs)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.video_backbone.video_encoder.parameters())
    for module in (projection.clean_proj, projection.proj1, projection.proj2, model.video_backbone.dit.head.head):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, projection.clean_proj.weight)
    after = model.prepare_inputs([sample])["input_latents"]
    torch.testing.assert_close(target, after, rtol=0, atol=0)
    run = tmp_path / "run"
    run.mkdir()
    model.video_backbone.save_deploy_assets(str(run), cfg)
    save_config(str(run), cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(run / "checkpoint.safetensors"))
    np.save(run / "normalization_stats.npy", {})
    contract = validate_vggt_omega_checkpoint(run, "checkpoint.safetensors", OmegaConf.to_container(cfg))
    assert contract["num_condition_latents"] == 2
    _, restored = load_from_checkpoint_dir(str(run), "cpu", "checkpoint.safetensors")
    assert all(not p.requires_grad for p in restored.video_backbone.video_encoder.parameters())
    from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
    native = SimpleNamespace(architecture=restored, _cfg_scale=1., _decode_video=False,
        _dit_cache=None, _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {
            "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 33,
            "video_num_frames": 9, "height": 224, "width": 448}}))
    engine = BatchedOpenWAMEngine(native)
    obs = [dict(prompt="open drawer", image=clip[0], history_image=images[0], state=np.zeros(10, np.float32))]
    batch, _, _ = engine._prepare(obs)
    ordinary = restored.video_backbone.preprocess_input_for_inference(prompt="open drawer",
        clean_condition_images=images[:2], num_frames=9, height=224, width=448,
        seed=42, num_inference_steps=2)
    torch.testing.assert_close(batch["clean_condition_latents"], ordinary["clean_condition_latents"])
    assert batch["latents"].shape == (1, 16, 10, 14, 28)
    chunks = engine.generate_batch(obs)
    assert chunks[0].shape == (32, 10) and np.isfinite(chunks[0]).all()
    plus = prepare_checkpoint(run, "checkpoint.safetensors", tmp_path / "plus", cfg.model.video_backbone.text_embedding_dir)
    assert (plus / "vggt_omega/encoder.json").is_file()
