"""V-JEPA patch-grid configuration through training and checkpoint reconstruction."""

import json
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.encoder.bottleneck import BottleneckPatchEmbed
from openwam.model.video_backbone.encoder.registry import build_video_encoder
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.encoder.vjepa21_src import loader
from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
from openwam.train.utils.checkpointing import save_config
from tests.test_vjepa_svae import _MockVJEPAViT
from tests.test_wan22_style_capacity import ROOT, launcher_config
from tests.test_wan_text_embedding_cache import make_cache


@pytest.fixture
def encoder_source(tmp_path, monkeypatch):
    source = tmp_path / "encoder"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "patch": 16, "tubelet": 2, "embed_dim": 8, "variant": "tiny", "arch_name": "tiny",
    }))
    monkeypatch.setattr(loader, "prepare_vjepa_imports_and_patch", lambda: None)
    monkeypatch.setattr(loader, "build_vit_from_manifest", lambda _, m: _MockVJEPAViT(m["embed_dim"]))
    monkeypatch.setattr(loader, "load_vit_weights", lambda *args: None)
    return source


def _joint_model(backbone):
    model = DualSystemSelfAttnArchitecture()
    model.video_backbone = backbone
    model.action_backbone = ActionDiT(
        action_dim=80, dim=48, ffn_dim=96, num_heads=4, num_layers=1,
        video_dim=96, bridge_layers=(0,), variant="joint_self_attn",
        attn_head_dim=24, text_dim=4096,
    )
    model._mot_driver_kwargs = {
        "attention_mask_mode": "mutual", "video_attention_mask_mode": "clean_condition_causal",
    }
    model.build_mot_driver()
    model.to(dtype=torch.float32)
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    model.init_training_schedulers(1000)
    return model


@pytest.mark.parametrize("override,expected", [(None, True), ("true", True), ("false", False)])
def test_hydra_boolean_is_the_public_grid_setting(override, expected):
    overrides = ["model/video_backbone/encoder=vjepa21"]
    if override is not None:
        overrides.append(f"model.video_backbone.encoder.use_wan_style_patch_grid={override}")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    assert cfg.model.video_backbone.encoder.use_wan_style_patch_grid is expected
    assert "dit_patch_size" not in cfg.model.video_backbone.encoder
    assert "patch_size" not in cfg.model.video_backbone.encoder


@pytest.mark.parametrize("cfg_type", [dict, OmegaConf.create, SimpleNamespace])
@pytest.mark.parametrize("flag", [True, False, None])
def test_skeleton_reads_boolean_and_legacy_default(encoder_source, cfg_type, flag):
    cfg = {"vjepa21_forward": "video"}
    if flag is not None:
        cfg["use_wan_style_patch_grid"] = flag
    encoder_cfg = cfg_type(**cfg) if cfg_type is SimpleNamespace else cfg_type(cfg)
    encoder = VJEPA21VideoEncoder.from_skeleton(
        {}, encoder_cfg=encoder_cfg, ckpt_dir=str(encoder_source),
    )
    expected = flag is not False
    assert encoder.use_wan_style_patch_grid is expected
    assert encoder.properties.dit_patch_size == ((1, 2, 2) if expected else (1, 1, 1))


def test_legacy_training_config_keeps_original_geometry(encoder_source):
    encoder = build_video_encoder({"name": "vjepa21", "model_path": str(encoder_source)})
    assert encoder.use_wan_style_patch_grid is True
    assert encoder.properties.dit_patch_size == (1, 2, 2)
    with pytest.raises(TypeError, match="dit_patch_size"):
        build_video_encoder({"name": "vjepa21", "model_path": str(encoder_source), "dit_patch_size": (1, 1, 1)})


@pytest.mark.parametrize("input_bottleneck", [False, True])
@pytest.mark.parametrize("separate_views", [False, True])
@pytest.mark.parametrize("history", [1, 0, 2])
@pytest.mark.parametrize("wan_grid", [True, False])
@pytest.mark.parametrize("temporal_pooling", [True, False])
def test_joint_forward_backward_and_saved_checkpoint(tmp_path, encoder_source, wan_grid, temporal_pooling, history, separate_views, input_bottleneck):
    torch.manual_seed(29)
    cfg = launcher_config()
    cfg.model.video_backbone.encoder.add_input_bottleneck = input_bottleneck
    cfg.model.video_backbone.encoder.bottleneck_dim = 4
    cfg.model.video_backbone.encoder.model_path = str(encoder_source)
    cfg.model.video_backbone.encoder.use_wan_style_patch_grid = wan_grid
    cfg.model.video_backbone.encoder.use_avg_t_pooling = temporal_pooling
    cfg.model.video_backbone.encoder.history_num_tubelets = int(history)
    cfg.model.video_backbone.encoder.seperate_encode_view = separate_views
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in {"dim": 96, "ffn_dim": 192, "num_heads": 4, "num_layers": 1, "freq_dim": 32}.items():
        cfg.model.video_backbone.dit_config[key] = value
    encoder = build_video_encoder(cfg.model.video_backbone.encoder)
    backbone = Wan22Style.from_pretrained(cfg, external_encoder=encoder, device="cpu")
    backbone.reinit_for_from_scratch(external_encoder=encoder)
    model = _joint_model(backbone)
    height, width = (224, 448) if separate_views else (64, 64)
    latents = encoder.batch_encode(torch.randn(1, 3, 8 + max(1, 2 * history), height, width))
    num_clean = max(1, history)
    latent_t = num_clean + (2 if temporal_pooling else 4)
    channels = sum(width for _, width in encoder.latent_channel_groups) or 8
    assert latents.shape == (1, channels, latent_t, height // 16, width // 16)
    # The shape-only ViT stub emits zeros. Use nonzero context to test the
    # clean projection weight gradients as well as the noisy future branch.
    latents = latents + torch.randn_like(latents)
    ps = (1, 2, 2) if wan_grid else (1, 1, 1)
    assert backbone.dit.patch_size == backbone.dit.head.patch_size == backbone.dit_patch_size == ps
    projection = backbone.dit.patch_embedding
    if input_bottleneck:
        assert isinstance(projection, BottleneckPatchEmbed)
        assert tuple(projection.clean_proj.weight.shape) == (96, channels, *ps)
        assert tuple(projection.proj1.weight.shape) == (4, channels, *ps)
        assert tuple(projection.proj2.weight.shape) == (96, 4, 1, 1, 1)
    else:
        assert tuple(projection.weight.shape) == (96, channels, *ps)
    assert backbone.dit.head.head.out_features == channels * ps[1] * ps[2]
    inputs = dict(
        latents=latents, context=torch.randn(1, 4, 4096), seq_lens=torch.tensor([3]),
        timestep=torch.tensor([500.0]), clean_condition_latents=latents[:, :, :num_clean], num_clean_condition_latents=num_clean,
        fuse_vae_embedding_in_latents=True,
    )
    state = backbone.prepare(**inputs)
    tokens = latent_t * (height // (16 * ps[1])) * (width // (16 * ps[2]))
    assert state.hidden_states.shape == (1, tokens, 96)
    assert state.time_mod.shape == (1, tokens, 6, 96)
    assert state.rope_freqs.shape[0] == tokens
    actions = torch.randn(1, 32, 80)
    action_t = torch.tensor([350.0])
    model.eval()
    with torch.no_grad():
        video, action = model(actions, action_t, **inputs)
    assert video.shape == latents.shape and action.shape == actions.shape
    assert torch.isfinite(video).all() and torch.isfinite(action).all()
    if separate_views:
        model.freeze_modules(["video_backbone.video_encoder"])
        assert backbone.dit.view_embedding.weight.requires_grad
    else:
        assert not hasattr(backbone.dit, "view_embedding")
    video_pad = torch.zeros(1, latent_t - num_clean, dtype=torch.bool)
    video_pad[:, -1] = True
    action_pad = torch.zeros_like(actions, dtype=torch.bool)
    action_pad[:, -2:] = True
    model.train()
    if input_bottleneck:
        model.video_prediction_type = model.action_prediction_type = "x"
    loss = model.compute_loss(
        actions=actions, input_latents=latents, context=inputs["context"], seq_lens=inputs["seq_lens"],
        clean_condition_latents=latents[:, :, :num_clean], num_clean_condition_latents=num_clean, fuse_vae_embedding_in_latents=True,
        video_is_pad=video_pad, action_is_pad=action_pad, use_gradient_checkpointing=True,
    )
    loss["loss"].backward()
    if encoder.latent_channel_groups:
        per_source = [loss[f"loss_video_{name}"] for name, _ in encoder.latent_channel_groups]
        torch.testing.assert_close(loss["loss_video"], torch.stack(per_source).mean())
    for parameter in (*projection.parameters(), backbone.dit.head.head.weight,
                      model.action_backbone.action_encoder.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    if separate_views:
        grad = backbone.dit.view_embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and (grad.abs().sum(dim=1) > 0).all()
    output = tmp_path / "checkpoint"
    backbone.save_deploy_assets(str(output), cfg)
    save_config(str(output), cfg)
    weights = output / "model.safetensors"
    save_file(model.state_dict(), str(weights))
    saved = OmegaConf.load(output / "config.yaml")
    assert saved.model.video_backbone.encoder.add_input_bottleneck is input_bottleneck
    assert saved.model.video_backbone.encoder.bottleneck_dim == 4
    assert saved.model.video_backbone.encoder.use_wan_style_patch_grid is wan_grid
    assert saved.model.video_backbone.encoder.history_num_tubelets == int(history)
    # Deployment must use checkpoint-local assets, not the training ViT directory.
    saved.model.video_backbone.encoder.model_path = "/unavailable/training/encoder"
    restored_encoder = VJEPA21VideoEncoder.from_skeleton(
        {}, encoder_cfg=saved.model.video_backbone.encoder, ckpt_dir=str(output),
    )
    assert restored_encoder.history_num_tubelets == int(history)
    assert restored_encoder.seperate_encode_view is separate_views
    restored_backbone = Wan22Style.from_pretrained(
        OmegaConf.to_container(saved.model.video_backbone, resolve=True),
        external_encoder=restored_encoder, device="cpu", ckpt_dir=str(output),
    )
    # Architecture initialization applies the encoder hooks before checkpoint loading.
    restored_backbone.reinit_for_from_scratch(
        external_encoder=restored_encoder, source=saved.model.video_backbone,
    )
    assert isinstance(restored_backbone.dit.patch_embedding, BottleneckPatchEmbed) is input_bottleneck
    restored = _joint_model(restored_backbone)
    restored.load_checkpoint(str(weights), strict=True)
    restored.eval()
    with torch.no_grad():
        restored_video, restored_action = restored(actions, action_t, **inputs)
    torch.testing.assert_close(restored_video, video)
    torch.testing.assert_close(restored_action, action)
