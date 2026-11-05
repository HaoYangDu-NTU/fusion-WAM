import hashlib
import torch
import pytest

from tests.test_wan_vae_encoder import small_model, encoder
from tests.test_wan_vae_optimization import sample

from openwam.experiments import resolve_experiment
from openwam.dataloader.wan_latent_cache import cache_identity


def test_wan_versions_use_separate_weight_identity_and_geometry(tmp_path):
    for version, experiment, shape in (("2.1", "E03", [16, 9, 28, 56]),
                                       ("2.2", "E04", [48, 9, 14, 28])):
        cfg, _, _ = resolve_experiment(experiment)
        cfg.dataloader.dataset_dir = str(tmp_path / "dataset")
        cfg.model.video_backbone.encoder.model_path = str(tmp_path)
        payload = f"encoder-only fixture {version}".encode()
        (tmp_path / f"Wan{version}_VAE.pth").write_bytes(payload)
        identity = cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, 7)
        assert identity["vae_version"] == version
        assert identity["shape"] == shape
        assert identity["weights_sha256"] == hashlib.sha256(payload).hexdigest()
        assert "latent_cache_dir" not in identity["data"]


def test_e04_preserves_e03_patch_training_and_token_budget():
    baseline, first, _ = resolve_experiment("E03-Patch")
    cfg, profile, _ = resolve_experiment("E04")
    assert first["geometry"]["tokens"] == profile["geometry"]["tokens"] == 882
    assert cfg.model.video_backbone.encoder.vae_version == "2.2"
    assert cfg.model.video_backbone.encoder.spatial_patch_size == 2
    assert cfg.model.action_backbone == baseline.model.action_backbone
    assert cfg.model.architecture == baseline.model.architecture
    assert cfg.model.video_backbone.dit_config == baseline.model.video_backbone.dit_config
    assert cfg.model.video_backbone.text_encoder_type == baseline.model.video_backbone.text_encoder_type
    for key in ("batch_size", "num_epochs", "learning_rate", "lambda_video", "lambda_action"):
        assert cfg.training[key] == baseline.training[key]
    assert cfg.training.resume_ckpt_path is None and cfg.training.finetune_ckpt_path is None
    assert cfg.dataloader.latent_cache_dir != baseline.dataloader.latent_cache_dir


def test_wan22_cache_preparation_and_backward(small_model):
    from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder
    model, _ = small_model
    enc = encoder("2.2").to(dtype=torch.bfloat16)
    model.video_backbone.video_encoder = enc
    adapt_dit_to_external_encoder(model.video_backbone, enc, enc.properties.dit_patch_size)
    model.video_backbone._dit_patch_size = enc.properties.dit_patch_size
    model.to(dtype=torch.bfloat16)
    model.init_training_schedulers()
    latent = torch.randn(48, 9, 14, 28, dtype=torch.bfloat16)
    cached = {**sample(), "video": [], "clean_condition_images": None, "wan_cached_latent": latent}
    inputs = model.prepare_inputs([cached])
    torch.testing.assert_close(inputs["input_latents"][0], latent, rtol=0, atol=0)
    torch.testing.assert_close(inputs["clean_condition_latents"][0], latent[:, :1], rtol=0, atol=0)
    loss = model.compute_loss(**inputs)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.video_backbone.dit.patch_embedding.clean_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with pytest.raises(ValueError, match="33-frame"):
        model.video_backbone.preprocess_cached_input_for_train([torch.zeros(16, 9, 28, 56)], ["open drawer"])
