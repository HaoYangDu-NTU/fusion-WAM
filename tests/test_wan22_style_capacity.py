"""Capacity-only construction checks; full parameter geometry uses meta tensors."""

import shlex
import re
import shutil
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.video_backbone import build_video_backbone
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from tests.test_wan_text_embedding_cache import make_cache

ROOT = Path(__file__).resolve().parents[1]


def launcher_config(small=True, backbone="wan22_style_1_3b"):
    suffix = "_wan22_style_1_3b" if small else ""
    script = ROOT / f"scripts/train_libero_vjepa21_cached_text{suffix}.sh"
    source = script.read_text()
    defaults = dict(re.findall(r'^([A-Z_]+)="\$\{[^:}]+:-([^}]*)\}"', source, re.MULTILINE))
    source = re.sub(r'\$\{([A-Z_]+):-([^}]*)\}', lambda m: defaults.get(m[1], m[2]), source)
    source = re.sub(r'\$([A-Z_]+)', lambda m: defaults.get(m[1], m[0]), source)
    tokens = shlex.split(source.replace("\\\n", " "), comments=True)
    overrides = tokens[tokens.index("scripts/train.py") + 1:]
    if small:
        overrides = [arg.replace("model/video_backbone=wan22_style_1_3b", f"model/video_backbone={backbone}") for arg in overrides]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=overrides)


def test_launcher_changes_only_capacity_and_output():
    small = OmegaConf.to_container(launcher_config(), resolve=True)
    baseline = OmegaConf.to_container(launcher_config(False), resolve=True)
    dit = small["model"]["video_backbone"].pop("dit_config")
    assert (dit["dim"], dit["ffn_dim"], dit["num_heads"], dit["num_layers"]) == (1536, 7168, 12, 30)
    assert dit["dim"] // dit["num_heads"] == 128
    small["model"]["video_backbone"]["name"] = baseline["model"]["video_backbone"]["name"]
    small["training"]["output_path"] = baseline["training"]["output_path"]
    # Hydra's wall-clock timestamp may cross a second between compositions.
    small["project"]["output_dir"] = baseline["project"]["output_dir"]
    assert small == baseline


def test_processing_methods_are_inherited_unchanged():
    for method in (
        "prepare", "finalize", "pre_attn_at_layer", "post_attn_at_layer",
        "preprocess_input_for_train", "preprocess_input_for_inference",
        "reinit_for_from_scratch",
    ):
        assert getattr(Wan22Style, method) is getattr(Wan22Ti2v, method)


@pytest.mark.parametrize("backbone,z_dim,dim,heads,action_dim,action_ffn,video_params,action_params", [
    ("wan22_style_1_3b", 1408, 1536, 12, 1024, 4096, 1_270_902_784, 643_087_440),
    ("wan22_style_0_5b", 1408, 1024, 8, 320, 1280, 497_010_176, 105_937_040),
    ("wan22_style_0_5b", 1024, 1024, 8, 320, 1280, 493_862_912, 105_937_040),
])
@pytest.mark.parametrize("source_shape", ["full", "normalized", "deploy"])
def test_full_geometry_and_action_attention(
    tmp_path, monkeypatch, source_shape, backbone, z_dim, dim, heads, action_dim, action_ffn, video_params, action_params,
):
    cfg = launcher_config(backbone=backbone)
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    # A bogus old model directory must never be scanned or loaded.
    cfg.model.video_backbone.model_path = "/must-not-load-the-5b-checkpoint"
    def unexpected(*args, **kwargs):
        raise AssertionError("The capacity ablation tried to discover pretrained Wan assets")
    monkeypatch.setattr("openwam.model.video_backbone.wan.pipeline_builder.discover_model_files", unexpected)
    encoder = VJEPA21VideoEncoder(torch.nn.Linear(1, 1), history_num_tubelets=0, embed_dim=z_dim, variant="vitg")
    source = {"full": cfg, "normalized": {"video_backbone": cfg.model.video_backbone},
              "deploy": OmegaConf.to_container(cfg.model.video_backbone, resolve=True)}[source_shape]
    vb = build_video_backbone(backbone, {}, source=source, device="meta", external_encoder=encoder)
    with torch.device("meta"):
        action = ActionDiT(
            action_dim=80, dim=action_dim, ffn_dim=action_ffn, num_heads=vb.num_heads,
            num_layers=vb.num_layers, video_dim=vb.dim, bridge_layers=tuple(range(vb.num_layers)),
            variant="joint_self_attn", attn_head_dim=vb.head_dim, text_dim=4096, shift_action=5.0,
        )
    assert sum(p.numel() for p in vb.dit.parameters()) == video_params
    assert sum(p.numel() for p in action.parameters()) == action_params
    assert tuple(vb.dit.patch_embedding.weight.shape) == (dim, z_dim, 1, 2, 2)
    assert tuple(vb.dit.head.head.weight.shape) == (z_dim * 4, dim)
    assert vb.num_heads == action.num_heads == heads
    assert vb.head_dim == action.head_dim == 128
    assert vb.num_layers == action.num_layers == 30
    assert vb._shift_video == 12.0
    assert vb.dit.seperated_timestep and vb.dit.fuse_vae_embedding_in_latents
    assert not vb.dit.has_image_input and not vb.dit.require_vae_embedding and not vb.dit.require_clip_embedding
    assert vb.text_encoder is None and vb.vae is None


@pytest.mark.parametrize("invalid,match", [
    ({"from_scratch": False}, "from_scratch"),
    ({"text_encoder_type": "invalid"}, "text_encoder_type"),
    ({"components": [{"attr": "dit", "extra_kwargs": {"dim": 3072}}]}, "does not match"),
])
def test_rejects_incompatible_configuration(invalid, match):
    cfg = OmegaConf.to_container(launcher_config().model.video_backbone, resolve=True)
    cfg.update(invalid)
    encoder = VJEPA21VideoEncoder(torch.nn.Linear(1, 1), history_num_tubelets=0, embed_dim=1408, variant="vitg")
    with pytest.raises(ValueError, match=match):
        Wan22Style.from_pretrained(cfg, external_encoder=encoder, device="meta")


@pytest.mark.parametrize("backbone,z_dim,dim", [
    ("wan22_style_1_3b", 1408, 1536),
    ("wan22_style_0_5b", 1408, 1024),
    ("wan22_style_0_5b", 1024, 1024),
])
def test_saved_components_use_small_model_and_shared_cache(tmp_path, monkeypatch, backbone, z_dim, dim):
    cfg = launcher_config(backbone=backbone)
    cache_dir = make_cache(tmp_path)
    cfg.model.video_backbone.text_embedding_dir = str(cache_dir)
    # Real cache directories may link to another dataset's much larger cache.
    # Saving deployment metadata must not traverse or copy either cache.
    other_cache = tmp_path / "libero_plus"
    other_cache.mkdir()
    (other_cache / "unrelated.safetensors").write_bytes(b"unrelated cache")
    (cache_dir / "libero_plus").symlink_to(other_cache, target_is_directory=True)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Wan22Style must reference the shared text cache, not copy it")

    monkeypatch.setattr(shutil, "copytree", unexpected_copy)
    encoder = VJEPA21VideoEncoder(torch.nn.Linear(1, 1), history_num_tubelets=0, embed_dim=z_dim, variant="vitg")
    # Encoder sidecar implementation is unchanged and exercised by its own tests.
    monkeypatch.setattr(encoder, "save_deploy_assets", lambda *args: None)
    vb = Wan22Style.from_pretrained(cfg, external_encoder=encoder, device="meta")
    output = tmp_path / "checkpoint"
    vb.save_deploy_assets(str(output), cfg)
    saved = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
    assert saved["components"][0]["extra_kwargs"]["dim"] == dim
    assert saved["components"][1]["attr"] == "vae"
    assert saved["components"][1]["source"] == "state_dict"
    assert not (output / "text_embedding").exists()
    assert saved["text_embedding_dir"] == str(cache_dir.resolve())
    saved["model_path"] = "/unavailable/wan"
    restored = Wan22Style.from_pretrained(saved, external_encoder=encoder, device="meta", ckpt_dir=str(output))
    assert restored.dim == dim
    context, lengths = restored._text_embedding_cache.encode(["open drawer"], device="cpu")
    assert context[0, 0, 0].item() == 1 and lengths.item() == 3
    assert {k: tuple(v.shape) for k, v in vb.state_dict().items()} == {
        k: tuple(v.shape) for k, v in restored.state_dict().items()
    }
