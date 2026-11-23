import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from safetensors.numpy import load_file, save_file

from benchmarks.vla_harness.libero_plus_protocol import check_text_cache
from benchmarks.vla_harness.prepare_libero_plus_svae import ENCODER_MARKER, prepare_checkpoint


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    cfg = {
        "model": {
            "video_backbone": {
                "encoder": {"name": "vjepa21", "svae_target_dim": 4},
                "from_scratch": True,
                "text_embedding_dir": "/original/libero/cache",
                "components": [{"attr": "dit"}],
            }
        },
        "dataloader": {"action_mode": "eef", "multiview": True},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    (root / "manifest.json").write_text('{"embed_dim": 8}')
    (root / "svae_config.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "model_config": {"input_dim": 8, "latent_dim": 4},
            }
        )
    )
    np.save(root / "normalization_stats.npy", {"test": 1})
    prefix = "video_backbone.video_encoder."
    tensors = {
        prefix + "_svae.enc_proj.weight": np.zeros((8, 8), np.float32),
        prefix + "_svae.input_mean": np.zeros(8, np.float32),
        prefix + "_svae.input_std": np.ones(8, np.float32),
        prefix + "feature_norm.weight": np.ones(4, np.float32),
        "video_backbone.dit.patch_embedding.weight": np.zeros((8, 4, 1, 2, 2), np.float32),
    }
    save_file(tensors, root / "checkpoint_step_1.safetensors")
    return root


@pytest.mark.parametrize("already_prepared", [False, True])
def test_prepare_preserves_model_and_includes_svae_assets(source, tmp_path, already_prepared):
    config = source / "config.yaml"
    original = yaml.safe_load(config.read_text())
    if already_prepared:
        original["model"]["video_backbone"]["components"].append(dict(ENCODER_MARKER))
        config.write_text(yaml.safe_dump(original))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output, cache = tmp_path / "deploy", tmp_path / "plus-cache"
    prepare_checkpoint(source, "checkpoint_step_1.safetensors", output, cache)
    actual = yaml.safe_load((output / "config.yaml").read_text())
    if not already_prepared:
        original["model"]["video_backbone"]["components"].append(dict(ENCODER_MARKER))
    original["model"]["video_backbone"]["text_embedding_dir"] = str(cache)
    assert actual == original
    for name in ["manifest.json", "svae_config.json", "normalization_stats.npy", "checkpoint_step_1.safetensors"]:
        assert (output / name).resolve() == source / name
    assert (output / "text_embedding").resolve() == cache
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert prepare_checkpoint(source, "checkpoint_step_1.safetensors", output, cache) == output
    with pytest.raises(FileNotFoundError, match="text cache is not ready"):
        check_text_cache(output, [], [])
    (output / "config.yaml").write_text("changed: true\n")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_checkpoint(source, "checkpoint_step_1.safetensors", output, cache)


@pytest.mark.parametrize("bad", ["missing_sidecar", "wrong_dim", "missing_weights"])
def test_invalid_svae_assets_fail_before_creating_deployment(source, tmp_path, bad):
    if bad == "missing_sidecar":
        (source / "svae_config.json").unlink()
    elif bad == "wrong_dim":
        cfg = yaml.safe_load((source / "config.yaml").read_text())
        cfg["model"]["video_backbone"]["encoder"]["svae_target_dim"] = 9
        (source / "config.yaml").write_text(yaml.safe_dump(cfg))
    else:
        save_file({"unrelated": np.zeros(1, np.float32)}, source / "checkpoint_step_1.safetensors")
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_checkpoint(source, "checkpoint_step_1.safetensors", tmp_path / "deploy", tmp_path / "cache")
    assert not (tmp_path / "deploy").exists()


def raw_checkpoint(source, separate, wan_grid):
    cfg = yaml.safe_load((source / "config.yaml").read_text())
    enc = cfg["model"]["video_backbone"]["encoder"]
    enc["svae_target_dim"] = None
    enc["use_wan_style_patch_grid"] = wan_grid
    if separate is not None:
        enc["seperate_encode_view"] = separate
    (source / "config.yaml").write_text(yaml.safe_dump(cfg))
    (source / "svae_config.json").unlink()
    grid = 2 if wan_grid else 1
    tensors = {
        "video_backbone.video_encoder.feature_norm.weight": np.ones(8, np.float32),
        "video_backbone.dit.patch_embedding.weight": np.zeros((8, 8, 1, grid, grid), np.float32),
    }
    if separate:
        tensors["video_backbone.dit.view_embedding.weight"] = np.zeros((2, 8), np.float32)
    save_file(tensors, source / "checkpoint_step_1.safetensors")


@pytest.mark.parametrize("separate", [None, False, True])
@pytest.mark.parametrize("wan_grid", [False, True])
def test_raw_vjepa_preparation_needs_no_svae_assets(source, tmp_path, separate, wan_grid):
    from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint as prepare_raw

    raw_checkpoint(source, separate, wan_grid)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output, cache = tmp_path / "raw-deploy", tmp_path / "plus-cache"
    prepare_raw(source, "checkpoint_step_1.safetensors", output, cache)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert not (output / "svae_config.json").exists()
    assert (output / "checkpoint_step_1.safetensors").resolve() == source / "checkpoint_step_1.safetensors"
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    expected = yaml.safe_load(before["config.yaml"])
    expected["model"]["video_backbone"]["text_embedding_dir"] = str(cache)
    expected["model"]["video_backbone"]["components"].append(dict(ENCODER_MARKER))
    assert cfg == expected
    record = json.loads((output / "deployment_preparation.json").read_text())
    assert record["svae_latent_dim"] is None and record["encoder_latent_dim"] == 8
    assert prepare_raw(source, "checkpoint_step_1.safetensors", output, cache) == output


@pytest.mark.parametrize("bad", ["missing_view", "wrong_view", "wrong_channels"])
def test_raw_vjepa_bad_geometry_fails_before_creating_deployment(source, tmp_path, bad):
    raw_checkpoint(source, True, False)
    path = source / "checkpoint_step_1.safetensors"
    tensors = load_file(path)
    if bad == "missing_view":
        del tensors["video_backbone.dit.view_embedding.weight"]
    elif bad == "wrong_view":
        tensors["video_backbone.dit.view_embedding.weight"] = np.zeros((2, 16), np.float32)
    else:
        tensors["video_backbone.dit.patch_embedding.weight"] = np.zeros((8, 4, 1, 1, 1), np.float32)
    save_file(tensors, path)
    with pytest.raises(ValueError):
        prepare_checkpoint(source, path.name, tmp_path / "deploy", tmp_path / "cache")
    assert not (tmp_path / "deploy").exists()


def test_harness_prepare_runs_real_svae_and_matches_individual_encoding(monkeypatch):
    from benchmarks.vla_harness import batch_engine
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    class TinyViT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 8)

        def forward(self, pixels):
            batch, _, frames, height, width = pixels.shape
            token = self.proj(pixels.mean(dim=(2, 3, 4)))
            count = (frames // 2) * (height // 16) * (width // 16)
            return token[:, None].expand(batch, count, 8)

    encoder = VJEPA21VideoEncoder(
        TinyViT(), history_num_tubelets=0,
        embed_dim=8,
        variant="test",
        svae_config={
            "input_dim": 8,
            "latent_dim": 4,
            "num_heads": 2,
            "num_layers": 1,
            "intermediate_size": 16,
            "dropout": 0.0,
        },
    ).eval()
    obs = [{"image": Image.fromarray(np.full((32, 32, 3), v, np.uint8)), "prompt": str(v)} for v in [50, 180]]
    expected = torch.cat([encoder.batch_encode(encoder.preprocess_video([o["image"]])) for o in obs])
    calls = []
    original = encoder._svae.encode_mean

    def encode_mean(z):
        calls.append(tuple(z.shape))
        return original(z)

    monkeypatch.setattr(encoder._svae, "encode_mean", encode_mean)
    monkeypatch.setattr(batch_engine, "make_schedule", lambda *a, **kw: [(1000, 1000), (0, 0)])
    vb = SimpleNamespace(video_encoder=encoder, _is_ti2v=True, _has_vace=False, shift_video=12.0,
                         encode_video=encoder.batch_encode)
    vb.preprocess_input_for_inference = lambda **kw: {
        "context": torch.zeros(1, 1, 8),
        "seq_lens": torch.tensor([1]),
        "noise": torch.zeros(1, 4, 1, 2, 2),
        "latents": torch.zeros(1, 4, 1, 2, 2),
    }
    arch = SimpleNamespace(
        video_backbone=vb,
        action_backbone=SimpleNamespace(shift_action=5.0),
        video_scheduler=None,
        action_scheduler=None,
        uses_proprioception=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        action_dim=10,
        eval=lambda: None,
    )
    inf = SimpleNamespace(num_frames=3, video_num_frames=1, denoise_steps=10, height=32, width=32, denoise_mode="sync")
    engine = batch_engine.BatchedOpenWAMEngine(
        SimpleNamespace(
            architecture=arch,
            cfg=SimpleNamespace(inference=inf),
            _cfg_scale=1.0,
            _decode_video=False,
            _prompt_embed_cache={},
        )
    )
    inputs, actions, _ = engine._prepare(obs)
    assert calls == [(2, 8, 1, 2, 2)]
    assert inputs["latents"].shape == (2, 4, 1, 2, 2)
    torch.testing.assert_close(inputs["clean_condition_latents"], expected)
    torch.testing.assert_close(inputs["latents"], expected)
    torch.testing.assert_close(actions[0], actions[1])


def test_repreparing_plus_preserves_explicit_matching_cache(source, tmp_path):
    cache = tmp_path / 'custom-plus-cache'
    cache.mkdir()
    (cache/'manifest.json').write_text(json.dumps(dict(correct_instruction_suffix=True, prompts={})))
    first = tmp_path/'first'
    prepare_checkpoint(source, 'checkpoint_step_1.safetensors', first, cache)
    second = tmp_path/'second'
    prepare_checkpoint(first, 'checkpoint_step_1.safetensors', second)
    cfg = yaml.safe_load((second/'config.yaml').read_text())
    assert cfg['model']['video_backbone']['text_embedding_dir'] == str(cache)
