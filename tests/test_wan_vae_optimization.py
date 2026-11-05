"""Cache semantics and native patch4 train/deploy contracts."""
import json
from concurrent.futures import ProcessPoolExecutor

import pytest
import torch
from safetensors.torch import save_file

from tests.test_wan_vae_encoder import small_model, frames, TinyVAE
from openwam.dataloader.wan_latent_cache import WanLatentCache
from openwam.model.video_backbone.encoder.wan_vae import WanVAEFeatureEncoder
from openwam.experiments import resolve_experiment


def _read_cache(args):
    path, i = args
    return float(WanLatentCache(path).read(i).sum())


def test_shard_roundtrip_worker_safety_and_missing_failure(tmp_path):
    z = torch.randn(16, 9, 28, 56).bfloat16()
    manifest = dict(shape=list(z.shape), shard_size=256, length=2)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    save_file({"0": z, "1": z.clone()}, str(tmp_path / "shard_000000.safetensors"))
    torch.testing.assert_close(WanLatentCache(tmp_path).read(0), z, rtol=0, atol=0)
    WanLatentCache(tmp_path).require_complete()
    with ProcessPoolExecutor(2) as pool:
        values = list(pool.map(_read_cache, [(str(tmp_path), 0), (str(tmp_path), 1)]))
    assert values[0] == values[1]
    with pytest.raises(Exception):
        WanLatentCache(tmp_path).read(2)
    manifest["length"] = 3
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete cache"):
        WanLatentCache(tmp_path).require_complete()


def sample():
    clip = frames(33)
    return dict(video=clip, clean_condition_images=clip[:1], prompt="open drawer",
                action=torch.zeros(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                proprio=torch.zeros(80), video_mask=torch.ones(33, dtype=torch.bool))


def test_online_cached_inputs_losses_and_gradients_match(small_model):
    model, _ = small_model
    model.init_training_schedulers()
    online = model.prepare_inputs([sample()])
    cached_sample = {**sample(), "video": [], "clean_condition_images": None,
                     "wan_cached_latent": online["input_latents"][0].clone()}
    cached = model.prepare_inputs([cached_sample])
    for key in ("input_latents", "clean_condition_latents", "video_is_pad", "actions", "context"):
        torch.testing.assert_close(online[key], cached[key], rtol=0, atol=0)
    assert online.keys() == cached.keys()
    torch.manual_seed(99)
    first = model.compute_loss(**online)["loss"]
    first.backward()
    grad = model.video_backbone.dit.patch_embedding.clean_proj.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(99)
    second = model.compute_loss(**cached)["loss"]
    second.backward()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(grad, model.video_backbone.dit.patch_embedding.clean_proj.weight.grad, rtol=0, atol=0)


def test_patch4_backward_preserves_raw_latent_target(small_model):
    from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder
    model, _ = small_model
    model.init_training_schedulers()
    enc = WanVAEFeatureEncoder(TinyVAE(), vae_version="2.1", bottleneck_dim=8, spatial_patch_size=4)
    model.video_backbone.video_encoder = enc.to(dtype=torch.bfloat16)
    adapt_dit_to_external_encoder(model.video_backbone, enc, enc.properties.dit_patch_size)
    model.video_backbone._dit_patch_size = enc.properties.dit_patch_size
    model.to(dtype=torch.bfloat16)
    inputs = model.prepare_inputs([sample()])
    assert inputs["input_latents"].shape == (1, 16, 9, 28, 56)
    proj = model.video_backbone.dit.patch_embedding
    grid = proj(inputs["input_latents"], num_clean_condition_latents=1)
    assert grid.shape[2:] == (9, 7, 14) and grid.shape[2:].numel() == 882
    before = proj.clean_proj.weight.detach().clone()
    optimizer = torch.optim.SGD(proj.parameters(), lr=0.1)
    loss = model.compute_loss(**inputs)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for p in proj.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    optimizer.step()
    assert not torch.equal(before, proj.clean_proj.weight)


def test_variants_preserve_protocol():
    base, _, _ = resolve_experiment("E03")
    for name, tokens, patch in (("E03-Cache", 3528, 2), ("E03-Patch", 882, 4)):
        cfg, profile, _ = resolve_experiment(name)
        assert profile["geometry"]["tokens"] == tokens
        assert cfg.model.video_backbone.encoder.spatial_patch_size == patch
        for key in ("batch_size", "num_epochs", "learning_rate", "lambda_video", "lambda_action"):
            assert cfg.training[key] == base.training[key]
        assert cfg.model.action_backbone == base.model.action_backbone
        assert cfg.model.architecture == base.model.architecture


def test_cache_identity_rejects_augmented_or_subsampled_inputs():
    from openwam.dataloader.wan_latent_cache import cache_identity
    cfg, _, _ = resolve_experiment("E03")
    cfg.dataloader.color_jitter.enabled = True
    with pytest.raises(ValueError, match="deterministic LIBERO"):
        cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, 1)
    cfg.dataloader.color_jitter.enabled = False
    cfg.dataloader.video_stride = 4
    with pytest.raises(ValueError, match="deterministic LIBERO"):
        cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, 1)


@pytest.mark.parametrize("small_model", [4], indirect=True)
def test_patch4_libero_and_plus_checkpoint_compatibility(small_model, tmp_path, monkeypatch):
    from tests.test_wan_vae_encoder import test_training_and_checkpoint_batch_deploy
    test_training_and_checkpoint_batch_deploy(small_model, tmp_path, monkeypatch)
