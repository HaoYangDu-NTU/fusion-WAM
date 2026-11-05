"""Only native VAE encoders are instantiated, with causal video and local reload."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.model.video_backbone.encoder.wan_vae import WanVAEFeatureEncoder


class TinyVAE(nn.Module):
    def __init__(self, version="2.1"):
        super().__init__()
        self.z_dim = 16 if version == "2.1" else 48
        self.upsampling_factor = 8 if version == "2.1" else 16
        self.model = nn.Module()
        self.model.encoder = nn.Conv3d(3, self.z_dim, 1)
        self.model.conv1 = nn.Conv3d(self.z_dim, self.z_dim * 2, 1)

    def batch_encode(self, video, device):
        video = torch.nn.functional.avg_pool3d(video.float(), (1, self.upsampling_factor, self.upsampling_factor)).to(video.dtype)
        z = self.model.conv1(self.model.encoder(video)).chunk(2, 1)[0]
        if z.shape[2] == 1:
            return z
        return torch.cat([z[:, :, :1], z[:, :, 1:].unflatten(2, (-1, 4)).mean(3)], dim=2)


def encoder(version="2.1", spatial_patch_size=2):
    return WanVAEFeatureEncoder(TinyVAE(version), vae_version=version, bottleneck_dim=8,
                                spatial_patch_size=spatial_patch_size)


@pytest.mark.parametrize("version,channels,size", [("2.1", 16, 28), ("2.2", 48, 14)])
def test_native_vae_causality_geometry_and_freeze(version, channels, size):
    enc = encoder(version)
    video = torch.randn(1, 3, 33, 224, 448)
    original = enc.batch_encode(video)
    assert original.shape == (1, channels, 9, size, size * 2)
    torch.testing.assert_close(original[:, :, :1], enc.batch_encode(video[:, :, :1]), rtol=0, atol=0)
    video[:, :, 5:] += 1
    modified = enc.batch_encode(video)
    torch.testing.assert_close(original[:, :, :2], modified[:, :, :2], rtol=0, atol=0)
    enc.train()
    assert all(not module.training for module in enc.modules())
    assert all(not p.requires_grad for p in enc.parameters())
    assert all("decoder" not in key and "dit" not in key and "text" not in key for key in enc.state_dict())


@pytest.fixture
def small_model(tmp_path, monkeypatch, request):
    from tests.test_wan22_style_capacity import launcher_config
    from tests.test_wan_text_embedding_cache import make_cache
    from openwam.model import build_architecture, resolve_architecture_config
    cfg = launcher_config()
    cfg.model.video_backbone.encoder = OmegaConf.create(dict(name="wan_vae", model_path="unused", vae_version="2.1",
                                                             microbatch_size=1, bottleneck_dim=8,
                                                             seperate_encode_view=True, add_input_bottleneck=True,
                                                             spatial_patch_size=getattr(request, "param", 2)))
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        cfg.model.video_backbone.dit_config[key] = value
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    cfg.model.architecture.video_prediction_type = cfg.model.architecture.action_prediction_type = "x"
    cfg.model.architecture.repa_enabled = False
    cfg.model.architecture.attention_mask_mode = "mutual"
    cfg.dataloader.normalize_mode = None
    monkeypatch.setattr(WanVAEFeatureEncoder, "from_pretrained", lambda *a, **k: encoder(spatial_patch_size=k.get("spatial_patch_size", 2)))
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.eval()
    return model, cfg


def frames(count):
    return [Image.new("RGB", (448, 224), (i * 3,) * 3) for i in range(count)]


def test_training_and_checkpoint_batch_deploy(small_model, tmp_path, monkeypatch):
    from openwam.train.utils.checkpointing import save_config
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
    from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
    from benchmarks.vla_harness.checkpoint_contract import validate_wan_vae_checkpoint
    model, cfg = small_model
    model.init_training_schedulers()
    clip = frames(33)
    sample = dict(video=clip, clean_condition_images=clip[:1], prompt="open drawer",
                  action=torch.zeros(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                  proprio=torch.zeros(80), video_mask=torch.ones(33, dtype=torch.bool))
    inputs = model.prepare_inputs([sample])
    assert inputs["input_latents"].shape == (1, 16, 9, 28, 56)
    loss = model.compute_loss(**inputs)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.video_backbone.dit.patch_embedding.proj1.weight.grad is not None
    assert all(p.grad is None for p in model.video_backbone.video_encoder.parameters())
    run = tmp_path / "run"
    run.mkdir()
    model.video_backbone.save_deploy_assets(str(run), cfg)
    save_config(str(run), cfg)
    name = "checkpoint.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(run / name))
    np.save(run / "normalization_stats.npy", {})
    config = OmegaConf.to_container(cfg, resolve=True)
    assert validate_wan_vae_checkpoint(run, name, config)["latent_channels"] == 16
    monkeypatch.setattr("openwam.model.video_backbone.encoder.wan_vae.build_vae", TinyVAE)
    _, deployed = load_from_checkpoint_dir(str(run), "cpu", name)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(deployed.state_dict()[key], value, rtol=0, atol=0)
    native = SimpleNamespace(architecture=deployed, _cfg_scale=1., _decode_video=False,
                             _dit_cache=None, _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {
                                 "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 33,
                                 "video_num_frames": 33, "height": 224, "width": 448}}))
    engine = BatchedOpenWAMEngine(native)
    obs = [dict(prompt="open drawer", image=clip[0], state=np.zeros(10, np.float32))]
    batch, _, _ = engine._prepare(obs)
    ordinary = deployed.video_backbone.preprocess_input_for_inference(prompt="open drawer", clean_condition_images=clip[:1],
                        num_frames=33, height=224, width=448, seed=42, num_inference_steps=2)
    torch.testing.assert_close(batch["clean_condition_latents"], ordinary["clean_condition_latents"])
    chunks = engine.generate_batch(obs)
    assert chunks[0].shape == (32, 10) and np.isfinite(chunks[0]).all()
    plus = prepare_checkpoint(run, name, tmp_path / "plus", cfg.model.video_backbone.text_embedding_dir)
    assert validate_wan_vae_checkpoint(plus, name, config)
