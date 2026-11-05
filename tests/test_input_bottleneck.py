"""Optional JEPA input bottleneck: legacy behavior, geometry, and launchers."""

import os
import re
import subprocess
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from openwam.model.video_backbone.encoder.base import VideoEncoder
from openwam.model.video_backbone.encoder.bottleneck import BottleneckPatchEmbed
from openwam.model.video_backbone.encoder.registry import build_video_encoder
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from tests.test_vjepa_patch_grid import encoder_source
from tests.test_wan22_style_capacity import ROOT


def encoder(**kwargs):
    kwargs.setdefault("history_num_tubelets", 0)
    return VJEPA21VideoEncoder(nn.Identity(), embed_dim=8, variant="test", **kwargs)


def test_disabled_is_exactly_the_legacy_projection():
    for cfg in ({}, {"add_input_bottleneck": False, "bottleneck_dim": 0}):
        enc = encoder(**cfg)
        torch.manual_seed(42)
        original = VideoEncoder.build_dit_input_proj(enc, 24)
        torch.manual_seed(42)
        current = enc.build_dit_input_proj(24)
        assert type(current) is nn.Conv3d
        assert set(current.state_dict()) == {"weight", "bias"}
        for name, value in original.state_dict().items():
            torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
        current.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(1, 8, 2, 4, 4)
        torch.testing.assert_close(current(x), original(x), rtol=0, atol=0)


@pytest.mark.parametrize("width", [1024, 1536, 3072])
@pytest.mark.parametrize("channels", [1024, 1408])
@pytest.mark.parametrize("wan_grid", [False, True])
def test_all_model_widths_and_jepa_encoders(width, channels, wan_grid):
    enc = VJEPA21VideoEncoder(
        nn.Identity(), history_num_tubelets=0, embed_dim=channels, variant="test",
        use_wan_style_patch_grid=wan_grid, add_input_bottleneck=True,
    )
    projection = enc.build_dit_input_proj(width)
    assert projection.proj1.out_channels == 512
    ps = enc.properties.dit_patch_size
    x = torch.randn(1, channels, 2, 2, 2, requires_grad=True)
    y = projection(x, num_clean_condition_latents=1)
    assert y.shape == (1, width, 2, 2 // ps[1], 2 // ps[2])
    y.square().mean().backward()
    for p in (*projection.parameters(), x):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0
    assert enc.build_dit_output_proj(width).out_features == channels * ps[1] * ps[2]


def test_projection_is_two_linear_maps_without_activation():
    layer = BottleneckPatchEmbed(8, 24, (1, 2, 2), 4).double()
    x = torch.randn(2, 8, 3, 4, 6, dtype=torch.float64)
    combined = (layer.proj2.weight.flatten(1) @ layer.proj1.weight.flatten(1)).reshape(24, 8, 1, 2, 2)
    expected = F.conv3d(x, combined, layer.proj2.bias, stride=(1, 2, 2))
    torch.testing.assert_close(layer(x), expected)
    assert layer.proj1.bias is None


@pytest.mark.parametrize("dim", [0, -1, True, 1.5, "512"])
def test_invalid_enabled_dimension_fails(dim):
    with pytest.raises(ValueError, match="positive integer"):
        encoder(add_input_bottleneck=True, bottleneck_dim=dim)
    with pytest.raises(ValueError, match="positive integer"):
        BottleneckPatchEmbed(8, 24, (1, 2, 2), dim)


@pytest.mark.parametrize("cfg_type", [dict, OmegaConf.create, SimpleNamespace])
@pytest.mark.parametrize("enabled", [False, True])
def test_training_and_skeleton_configs_match(encoder_source, cfg_type, enabled):
    values = {"vjepa21_forward": "video"}
    if enabled:
        values.update(add_input_bottleneck=True, bottleneck_dim=4)
    train = build_video_encoder(dict(name="vjepa21", model_path=str(encoder_source), **values))
    cfg = cfg_type(**values) if cfg_type is SimpleNamespace else cfg_type(values)
    deploy = VJEPA21VideoEncoder.from_skeleton({}, encoder_cfg=cfg, ckpt_dir=str(encoder_source))
    train_proj, deploy_proj = train.build_dit_input_proj(24), deploy.build_dit_input_proj(24)
    assert isinstance(deploy_proj, BottleneckPatchEmbed) is enabled
    deploy_proj.load_state_dict(train_proj.state_dict(), strict=True)


def launcher_config(script_name, env_values, extra=()):
    path = ROOT / "scripts" / script_name
    subprocess.run(["bash", "-n", str(path)], check=True)
    script = path.read_text()
    env = os.environ.copy()
    for name in re.findall(r'^([A-Z_]+)="', script, re.MULTILINE):
        env.pop(name, None)
    env.update(env_values)
    # Capture the real arguments without activating Conda or starting training.
    script = "\n".join(line for line in script.splitlines()
                       if not line.startswith(("cd ", "source ", "conda activate ")))
    assert script.count("exec taskset -c ") == 1
    script = script.replace("exec taskset -c ", "capture_command -c ", 1)
    result = subprocess.run(
        ["bash", "-c", 'capture_command() { printf "%s\\0" "$@"; }\n' + script, "test", *extra],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("\0")
    overrides = [arg for arg in args[args.index("scripts/train.py") + 1:] if arg]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=overrides)


@pytest.mark.parametrize("script", [
    "train_libero_vjepa_0_6b_dev.sh", "train_libero_vjepa_1_3b_dev.sh",
    "train_libero_vjepa_5b_dev.sh", "train_libero_vjepa_1_3b_dev_share_dir.sh",
])
def test_launcher_environment_and_cli_overrides(script):
    cfg = launcher_config(script, {
        "ADD_INPUT_BOTTLENECK": "false", "BOTTLENECK_DIM": "512",
    }).model.video_backbone.encoder
    assert cfg.add_input_bottleneck is False and cfg.bottleneck_dim == 512
    env = {"ADD_INPUT_BOTTLENECK": "true", "BOTTLENECK_DIM": "256"}
    cfg = launcher_config(script, env).model.video_backbone.encoder
    assert cfg.add_input_bottleneck is True and cfg.bottleneck_dim == 256
    cfg = launcher_config(script, env, (
        "model.video_backbone.encoder.add_input_bottleneck=false",
        "model.video_backbone.encoder.bottleneck_dim=128",
    )).model.video_backbone.encoder
    assert cfg.add_input_bottleneck is False and cfg.bottleneck_dim == 128


@pytest.mark.parametrize("size,width", [("0_6b", 1024), ("1_3b", 1536), ("5b", 3072)])
@pytest.mark.parametrize("channels", [1024, 1408])
def test_full_backbone_construction_on_meta(tmp_path, monkeypatch, size, width, channels):
    from openwam.model.video_backbone import build_video_backbone
    from tests.test_wan_text_embedding_cache import make_cache

    cfg = launcher_config(f"train_libero_vjepa_{size}_dev.sh", {"ADD_INPUT_BOTTLENECK": "true"})
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    enc = VJEPA21VideoEncoder(
        nn.Identity(), history_num_tubelets=0, embed_dim=channels, variant="test",
        add_input_bottleneck=True, seperate_encode_view=True,
    )
    source = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
    if size == "5b":
        from openwam.model.video_backbone.wan.shared.configs.model_configs import wan_series
        entry = next(item for item in wan_series if item.get("extra_kwargs", {}).get("dim") == 3072)
        source["components"] = [{
            "attr": "dit", "model_class": entry["model_class"], "extra_kwargs": entry["extra_kwargs"],
        }]
    def unexpected(*args, **kwargs):
        raise AssertionError("Meta construction must not load pretrained assets")
    monkeypatch.setattr("openwam.model.video_backbone.wan.pipeline_builder.discover_model_files", unexpected)
    backbone = build_video_backbone(
        cfg.model.video_backbone.name, {}, source=source, device="meta", external_encoder=enc,
    )
    with torch.device("meta"):
        backbone.reinit_for_from_scratch(external_encoder=enc, source=cfg)
    projection = backbone.dit.patch_embedding
    assert isinstance(projection, BottleneckPatchEmbed)
    assert projection.clean_proj.weight.shape == (width, channels, 1, 2, 2)
    assert projection.clean_proj.weight.is_meta
    assert projection.proj1.weight.shape == (512, channels, 1, 2, 2)
    assert projection.proj2.weight.shape == (width, 512, 1, 1, 1)
    assert projection.proj1.weight.is_meta and projection.proj2.weight.is_meta
    assert backbone.dit.head.head.weight.shape == (channels * 4, width)
    assert backbone.dit.view_embedding.weight.shape == (2, width)
    assert backbone.dit.view_embedding.weight.is_meta


@pytest.mark.parametrize("patch_size", [(1, 1, 1), (1, 2, 2)])
@pytest.mark.parametrize("clean_frames", [0, 1, 2, 3])
def test_split_preserves_order_and_uses_independent_projections(patch_size, clean_frames):
    layer = BottleneckPatchEmbed(8, 24, patch_size, 4).double()
    x = torch.randn(2, 8, 3, 4, 6, dtype=torch.float64)
    clean = F.conv3d(x, layer.clean_proj.weight, layer.clean_proj.bias, stride=patch_size)
    combined = (layer.proj2.weight.flatten(1) @ layer.proj1.weight.flatten(1)).reshape(24, 8, *patch_size)
    future = F.conv3d(x, combined, layer.proj2.bias, stride=patch_size)
    expected = torch.cat([clean[:, :, :clean_frames], future[:, :, clean_frames:]], dim=2)
    torch.testing.assert_close(layer(x, num_clean_condition_latents=clean_frames), expected)


@pytest.mark.parametrize("loss_on_clean", [False, True])
def test_clean_and_future_gradients_are_independent(loss_on_clean):
    layer = BottleneckPatchEmbed(8, 24, (1, 2, 2), 4)
    x = torch.randn(1, 8, 3, 4, 4, requires_grad=True)
    y = layer(x, num_clean_condition_latents=1)
    selected = y[:, :, :1] if loss_on_clean else y[:, :, 1:]
    selected.square().mean().backward()
    clean_grad = layer.clean_proj.weight.grad.abs().sum()
    future_grad = layer.proj1.weight.grad.abs().sum() + layer.proj2.weight.grad.abs().sum()
    assert (clean_grad > 0) if loss_on_clean else (clean_grad == 0)
    assert (future_grad == 0) if loss_on_clean else (future_grad > 0)
    excluded = x.grad[:, :, 1:] if loss_on_clean else x.grad[:, :, :1]
    assert torch.count_nonzero(excluded) == 0


@pytest.mark.parametrize("clean_frames", [-1, 4])
def test_invalid_clean_prefix_is_rejected(clean_frames):
    layer = BottleneckPatchEmbed(8, 24, (1, 2, 2), 4)
    with pytest.raises(ValueError, match="clean prefix"):
        layer(torch.randn(1, 8, 3, 4, 4), num_clean_condition_latents=clean_frames)


@pytest.fixture
def split_backbone(tmp_path, encoder_source):
    from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
    from tests.test_wan22_style_capacity import launcher_config as capacity_config
    from tests.test_wan_text_embedding_cache import make_cache

    cfg = capacity_config()
    cfg.model.video_backbone.encoder.model_path = str(encoder_source)
    cfg.model.video_backbone.encoder.add_input_bottleneck = True
    cfg.model.video_backbone.encoder.bottleneck_dim = 4
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in {"dim": 96, "ffn_dim": 192, "num_heads": 4, "num_layers": 1, "freq_dim": 32}.items():
        cfg.model.video_backbone.dit_config[key] = value
    enc = build_video_encoder(cfg.model.video_backbone.encoder)
    backbone = Wan22Style.from_pretrained(cfg, external_encoder=enc, device="cpu")
    backbone.reinit_for_from_scratch(external_encoder=enc)
    return backbone.float()


@pytest.mark.parametrize("prefix,reference,fuse,expected_clean", [
    (0, 0, False, 0), (0, 0, True, 1), (0, 1, False, 1),
    (2, 0, False, 2), (1, 2, True, 2), (2, 1, True, 2), (3, 0, True, 3),
])
@pytest.mark.parametrize("timestep", [0.0, 500.0])
def test_backbone_routes_declared_context_not_diffusion_time(
    split_backbone, prefix, reference, fuse, expected_clean, timestep,
):
    latents = torch.randn(1, 8, 3, 4, 4)
    layer = split_backbone.dit.patch_embedding
    inputs = dict(
        latents=latents, context=torch.randn(1, 4, 4096), timestep=torch.tensor([timestep]),
        num_clean_condition_latents=prefix, fuse_vae_embedding_in_latents=fuse,
    )
    if reference:
        inputs["clean_condition_latents"] = latents[:, :, :reference]
    state = split_backbone.prepare(**inputs)
    clean = layer.clean_proj(latents)
    future = layer.proj2(layer.proj1(latents))
    expected = torch.cat([clean[:, :, :expected_clean], future[:, :, expected_clean:]], dim=2)
    torch.testing.assert_close(state.hidden_states, expected.flatten(2).transpose(1, 2))
    # Repeated denoising calls update only future values, keeping clean tokens fixed.
    inputs["latents"] = latents.clone()
    inputs["latents"][:, :, expected_clean:] += 1.0
    updated = split_backbone.prepare(**inputs)
    clean_tokens = expected_clean * 4
    torch.testing.assert_close(updated.hidden_states[:, :clean_tokens], state.hidden_states[:, :clean_tokens])
    if expected_clean < 3:
        assert not torch.allclose(updated.hidden_states[:, clean_tokens:], state.hidden_states[:, clean_tokens:])


def test_model_config_defaults_preserve_shared_projection():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["model/video_backbone/encoder=vjepa21"])
    assert cfg.model.video_backbone.encoder.add_input_bottleneck is False
    assert cfg.model.video_backbone.encoder.bottleneck_dim == 512
