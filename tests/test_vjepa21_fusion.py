"""CPU checks for fusion math, branch isolation and checkpoint reconstruction."""

import copy
import json
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from openwam.model.video_backbone.encoder.registry import build_video_encoder
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.encoder.vjepa21_fusion import (
    DEFAULT_FUSION_LAYERS,
    FUSION_METHODS,
    fuse_layer_features,
    resolve_fusion_layers,
)
from openwam.model.video_backbone.encoder.vjepa21_src import loader
from openwam.model.video_backbone.encoder.vjepa21_src.vision_transformer import VisionTransformer

SINGLE_SOURCE_METHODS = tuple(method for method in FUSION_METHODS if method not in ("concat_before_bottleneck", "adafuse"))


class GridViT(nn.Module):
    """Distinct layers with full-clip interaction and a non-identity native norm."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.proj = nn.Linear(3, 8)
        self.norm = nn.LayerNorm(8)
        self.native_calls = self.intermediate_calls = 0
        with torch.no_grad():
            self.norm.weight.copy_(torch.linspace(0.5, 2, 8))
            self.norm.bias.copy_(torch.linspace(-1, 1, 8))

    def raw_layers(self, pixels):
        tubelet = 1 if pixels.shape[2] == 1 else 2
        # CPU pooling needs FP32; preserve the toy patchifier's output dtype.
        grid = F.avg_pool3d(pixels.float(), (tubelet, 16, 16)).to(pixels.dtype)
        grid = self.proj(grid.permute(0, 2, 3, 4, 1))
        for i in range(len(self.blocks)):
            grid = grid + 0.2 * grid.mean((1, 2, 3), keepdim=True) + 0.3 * torch.tanh(grid.roll(i + 1, -1))
            yield grid

    def forward(self, pixels):
        self.native_calls += 1
        for grid in self.raw_layers(pixels):
            pass
        return self.norm(grid).flatten(1, 3)

    def iter_intermediate_layers(self, pixels, layer_indices):
        self.intermediate_calls += 1
        for i, grid in enumerate(self.raw_layers(pixels)):
            if i in layer_indices:
                yield grid


def reference_fusion(layers, method):
    normalized = [F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5) for x in layers]
    result = torch.stack(normalized).mean(0)
    if method == "mean_plus_global":
        result = result + normalized[-1].mean((2, 3), keepdim=True)
    return result.to(layers[0].dtype)


@pytest.mark.parametrize("method", ["mean", "mean_plus_global"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fusion_matches_reference_and_preserves_time_and_view(method, dtype):
    torch.manual_seed(13)
    layers = [torch.randn(2, 4, 2, 3, 8).to(dtype) for _ in range(7)]
    actual = fuse_layer_features(iter(layers), method)
    torch.testing.assert_close(actual, reference_fusion(layers, method))
    # Changing another view or time must not change this time's spatial mean.
    altered = [x.clone() for x in layers]
    for x in altered:
        x[1] = torch.randn_like(x[1])
        x[0, 0] = torch.randn_like(x[0, 0])
    torch.testing.assert_close(fuse_layer_features(iter(altered), method)[0, 1:], actual[0, 1:])


def test_streaming_fusion_does_not_retain_all_raw_layers():
    refs = []

    def stream():
        for _ in range(40):
            feature = torch.randn(1, 2, 2, 3, 8)
            refs.append(weakref.ref(feature))
            assert sum(ref() is not None for ref in refs) <= 2
            yield feature

    assert torch.isfinite(fuse_layer_features(stream(), "mean_plus_global")).all()


@pytest.mark.parametrize("layers", [[0, 2, 3], "all_layers"])
def test_layer_selection(layers):
    assert resolve_fusion_layers(layers, 4) == ((0, 1, 2, 3) if isinstance(layers, str) else tuple(layers))
    assert resolve_fusion_layers("all_layers", 40) == tuple(range(40))
    assert resolve_fusion_layers(DEFAULT_FUSION_LAYERS, 40) == DEFAULT_FUSION_LAYERS


@pytest.mark.parametrize("layers", [[], [-1], [4], [1, 1], [2, 0], [True], [1.5], "all"])
def test_invalid_layer_selection(layers):
    with pytest.raises(ValueError, match="fusion_layers"):
        resolve_fusion_layers(layers, 4)


@pytest.mark.parametrize("frames", [1, 6])
def test_real_vit_intermediates_match_block_hooks_and_keep_native_norm(frames):
    torch.manual_seed(9)
    vit = VisionTransformer(
        img_size=32, patch_size=16, num_frames=6, tubelet_size=2,
        embed_dim=48, depth=12, num_heads=2, use_rope=True, img_temporal_dim_size=1,
    ).eval().requires_grad_(False)
    captured = []
    hooks = [block.register_forward_hook(lambda _m, _args, output: captured.append(output[0].clone()))
             for block in vit.blocks]
    pixels = torch.randn(2, 3, frames, 32, 48)
    native = vit(pixels)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(native, vit.norms_block[-1](captured[-1]), rtol=0, atol=0)
    # Includes non-hierarchical layers, which the old out_layers API cannot select.
    selected = (0, 1, 7, 11)
    for i, grid in zip(selected, vit.iter_intermediate_layers(pixels, selected), strict=True):
        assert grid.shape == (2, 1 if frames == 1 else 3, 2, 3, 48)
        torch.testing.assert_close(grid.flatten(1, 3), captured[i], rtol=0, atol=0)
    assert len(list(vit.iter_intermediate_layers(pixels, range(12)))) == 12


def reference_clip(vit, pixels, method, layers, start=0):
    # Reference handles cameras one at a time, unlike the encoder's packed batch.
    if method == "last_layer":
        grid = vit(pixels).reshape(pixels.shape[0], pixels.shape[2] // 2, 14, 14, 8)[:, start:]
    else:
        selected = [x[:, start:] for i, x in enumerate(vit.raw_layers(pixels)) if i in layers]
        grid = reference_fusion(selected, method)
    return grid.permute(0, 4, 1, 2, 3)


@pytest.mark.parametrize("condition", SINGLE_SOURCE_METHODS)
@pytest.mark.parametrize("future", SINGLE_SOURCE_METHODS)
@pytest.mark.parametrize("pool", [False, True])
def test_nine_branch_combinations_match_separate_camera_reference(condition, future, pool):
    torch.manual_seed(7)
    vit = GridViT()
    reference_vit = copy.deepcopy(vit)
    encoder = VJEPA21VideoEncoder(
        vit, embed_dim=8, variant="tiny", seperate_encode_view=True,
        history_num_tubelets=1, use_avg_t_pooling=pool,
        condition_fusion_method=condition, future_fusion_method=future, fusion_layers=[0, 2, 3],
    )
    pixels = torch.randn(2, 3, 6, 224, 448)
    views = []
    for view in pixels.chunk(2, dim=-1):
        cond = reference_clip(reference_vit, view[:, :, :2], condition, [0, 2, 3])
        target = reference_clip(reference_vit, view, future, [0, 2, 3], start=1)
        if pool:
            target = target.mean(2, keepdim=True)
        views.append(torch.cat([cond, target], dim=2))
    expected = torch.cat(views, dim=-1)
    expected = encoder.feature_norm(expected.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
    torch.testing.assert_close(encoder.batch_encode(pixels), expected)
    assert vit.native_calls == (condition == "last_layer") + (future == "last_layer")
    assert vit.intermediate_calls == (condition != "last_layer") + (future != "last_layer")

    changed = pixels.clone()
    changed[:, :, 2:] = torch.randn_like(changed[:, :, 2:]) * 10
    before = encoder.batch_encode(pixels)
    after = encoder.batch_encode(changed)
    torch.testing.assert_close(before[:, :, :1], after[:, :, :1], rtol=0, atol=0)
    assert not torch.allclose(before[:, :, 1:], after[:, :, 1:])


@pytest.mark.parametrize("method", SINGLE_SOURCE_METHODS)
@pytest.mark.parametrize("history,forward", [(False, "mixed"), (False, "video"), (True, "video")])
def test_condition_only_does_not_run_future(method, history, forward):
    vit = GridViT()
    encoder = VJEPA21VideoEncoder(
        vit, embed_dim=8, variant="tiny", history_num_tubelets=int(history),
        vjepa21_forward=forward, condition_fusion_method=method,
        future_fusion_method="mean_plus_global", fusion_layers="all_layers",
    )
    result = encoder.batch_encode(torch.randn(1, 3, 2 if history else 1, 32, 48))
    assert result.shape == (1, 8, 1, 2, 3)
    assert vit.native_calls + vit.intermediate_calls == 1


def test_last_layer_ignores_selection_and_preserves_native_affine():
    vit = GridViT()
    encoder = VJEPA21VideoEncoder(vit, history_num_tubelets=0, embed_dim=8, variant="tiny")
    pixels = torch.randn(1, 3, 2, 32, 48)
    expected = vit(pixels).transpose(1, 2).reshape(1, 8, 1, 2, 3)
    torch.testing.assert_close(encoder._vit_grid(pixels), expected, rtol=0, atol=0)
    assert encoder.fusion_layers == ()
    assert vit.intermediate_calls == 0
    with pytest.raises(ValueError, match="condition_fusion_method"):
        VJEPA21VideoEncoder(vit, history_num_tubelets=0, embed_dim=8, variant="tiny", condition_fusion_method="typo")


@pytest.fixture
def encoder_source(tmp_path, monkeypatch):
    source = tmp_path / "encoder"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "patch": 16, "tubelet": 2, "embed_dim": 8, "variant": "tiny", "arch_name": "tiny",
    }))
    monkeypatch.setattr(loader, "prepare_vjepa_imports_and_patch", lambda: None)
    monkeypatch.setattr(loader, "build_vit_from_manifest", lambda *_: GridViT())
    monkeypatch.setattr(loader, "load_vit_weights", lambda *_: None)
    return source


@pytest.mark.parametrize("layers", [[0, 2, 3], "all_layers"])
@pytest.mark.parametrize("cfg_type", [dict, OmegaConf.create, SimpleNamespace])
def test_training_to_deploy_roundtrip_and_mismatch_rejection(tmp_path, encoder_source, layers, cfg_type):
    cfg = OmegaConf.create({"model": {"video_backbone": {"encoder": {
        "name": "vjepa21", "model_path": str(encoder_source), "vjepa21_forward": "video",
        "condition_fusion_method": "mean", "future_fusion_method": "mean_plus_global",
        "fusion_layers": layers,
    }}}})
    encoder = build_video_encoder(cfg.model.video_backbone.encoder)
    output = tmp_path / "checkpoint"
    encoder.save_deploy_assets(str(output), cfg)
    OmegaConf.save(cfg, output / "config.yaml")
    save_file(encoder.state_dict(), str(output / "weights.safetensors"))
    saved = OmegaConf.to_container(OmegaConf.load(output / "config.yaml"), resolve=True)["model"]["video_backbone"]["encoder"]
    saved["model_path"] = "/unavailable/training/path"
    enc_cfg = cfg_type(**saved) if cfg_type is SimpleNamespace else cfg_type(saved)
    restored = VJEPA21VideoEncoder.from_skeleton({}, encoder_cfg=enc_cfg, ckpt_dir=str(output))
    restored.load_state_dict(load_file(str(output / "weights.safetensors")), strict=True)
    pixels = torch.randn(1, 3, 9, 32, 48)
    torch.testing.assert_close(restored.batch_encode(pixels), encoder.batch_encode(pixels), rtol=0, atol=0)
    metadata = json.loads((output / "manifest.json").read_text())["feature_fusion"]
    assert metadata["fusion_layers"] == list(encoder.fusion_layers)
    assert metadata["condition_fusion_method"] == "mean"
    assert metadata["future_fusion_method"] == "mean_plus_global"
    saved["future_fusion_method"] = "last_layer"
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        VJEPA21VideoEncoder.from_skeleton({}, encoder_cfg=saved, ckpt_dir=str(output))


def test_legacy_skeleton_defaults_to_native_path(encoder_source):
    encoder = VJEPA21VideoEncoder.from_skeleton(
        {}, encoder_cfg={"vjepa21_forward": "video"}, ckpt_dir=str(encoder_source),
    )
    assert encoder.condition_fusion_method == encoder.future_fusion_method == "last_layer"
    assert encoder.fusion_layers == ()


def test_saving_last_layer_replaces_inherited_fusion_metadata(tmp_path, encoder_source):
    path = encoder_source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["feature_fusion"] = {"condition_fusion_method": "mean", "future_fusion_method": "mean",
                                  "fusion_layers": [0, 2, 3]}
    path.write_text(json.dumps(manifest))
    encoder = VJEPA21VideoEncoder(GridViT(), history_num_tubelets=0, embed_dim=8, variant="tiny")
    cfg = OmegaConf.create({"model": {"video_backbone": {"encoder": {"model_path": str(encoder_source)}}}})
    output = tmp_path / "last_layer_checkpoint"
    encoder.save_deploy_assets(str(output), cfg)
    restored = VJEPA21VideoEncoder.from_skeleton({}, encoder_cfg={}, ckpt_dir=str(output))
    assert restored.condition_fusion_method == restored.future_fusion_method == "last_layer"


@pytest.mark.parametrize("condition,future", [("mean_plus_global", "last_layer"),
                                            ("last_layer", "mean_plus_global"), ("mean", "mean")])
def test_fused_latents_train_and_deploy_with_joint_dit(tmp_path, monkeypatch, encoder_source, condition, future):
    # Exercise the existing end-to-end DiT/backward/strict checkpoint test with
    # real, nonzero fused latents instead of its shape-only encoder stub.
    from tests import test_vjepa_patch_grid as integration

    original_config = integration.launcher_config

    def fused_config():
        cfg = original_config()
        cfg.model.video_backbone.encoder.condition_fusion_method = condition
        cfg.model.video_backbone.encoder.future_fusion_method = future
        cfg.model.video_backbone.encoder.fusion_layers = "all_layers"
        return cfg

    monkeypatch.setattr(integration, "launcher_config", fused_config)
    integration.test_joint_forward_backward_and_saved_checkpoint(
        tmp_path, encoder_source, wan_grid=True, temporal_pooling=False,
        history=True, separate_views=True, input_bottleneck=True,
    )
