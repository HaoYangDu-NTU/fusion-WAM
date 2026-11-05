"""Concat source semantics, real joint training and checkpoint deployment."""

import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.encoder.vjepa21_src import loader
from openwam.train.utils.training_utils import reduce_step_metrics
from tests import test_vjepa_patch_grid as integration
from tests.test_vjepa21_fusion import encoder_source  # noqa: F401
from tests.test_vjepa21_repa import RepaGridViT

METHOD = "concat_before_bottleneck"


def make_encoder(vit, layers=(1, 3), **kwargs):
    kwargs.setdefault("history_num_tubelets", 0)
    return VJEPA21VideoEncoder(
        vit, embed_dim=8, variant="tiny", add_input_bottleneck=True,
        condition_fusion_method=METHOD, future_fusion_method=METHOD,
        fusion_layers=list(layers) if not isinstance(layers, str) else layers, **kwargs,
    ).eval().requires_grad_(False)


@pytest.fixture
def concat_source(encoder_source, monkeypatch):
    monkeypatch.setattr(loader, "build_vit_from_manifest", lambda *_: RepaGridViT())
    return encoder_source


@pytest.mark.parametrize("layers", [[3], [1, 3], "all_layers"])
@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("views", [False, True])
def test_concat_matches_independent_sources_and_preserves_native_final(layers, pool, views):
    torch.manual_seed(29)
    vit = RepaGridViT()
    encoder = make_encoder(vit, layers, history_num_tubelets=1,
                           use_avg_t_pooling=pool, seperate_encode_view=views)
    # Test a non-identity OpenWAM affine as well as the pretrained JEPA affine.
    with torch.no_grad():
        encoder.feature_norm.weight.copy_(torch.linspace(0.8, 1.4, 8))
        encoder.feature_norm.bias.copy_(torch.linspace(-0.2, 0.2, 8))
    native = VJEPA21VideoEncoder(
        copy.deepcopy(vit), embed_dim=8, variant="tiny", history_num_tubelets=1,
        use_avg_t_pooling=pool, seperate_encode_view=views,
    ).eval().requires_grad_(False)
    native.feature_norm.load_state_dict(encoder.feature_norm.state_dict())
    pixels = torch.randn(2, 3, 6, 224 if views else 32, 448 if views else 48)
    selected = list(range(4)) if layers == "all_layers" else layers
    expected_views = []
    for view in pixels.chunk(2, -1) if views else [pixels]:
        cond_layers = list(vit.raw_layers(view[:, :, :2]))
        future_layers = list(vit.raw_layers(view))
        sources = []
        for index in selected:
            norm = vit.norm if index == 3 else lambda x: F.layer_norm(x.float(), (8,), eps=1e-5)
            cond = norm(cond_layers[index])
            future = norm(future_layers[index][:, 1:])
            if pool:
                future = future.mean(1, keepdim=True)
            source = encoder.feature_norm(torch.cat([cond, future], 1))
            sources.append(source)
        expected_views.append(torch.cat(sources, -1).permute(0, 4, 1, 2, 3))
    expected = torch.cat(expected_views, -1)
    actual = encoder.batch_encode(pixels)
    assert encoder.properties.z_dim == 8 * len(selected)
    assert encoder.latent_channel_groups == tuple((f"layer_{i}", 8) for i in selected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, -8:], native.batch_encode(pixels), rtol=0, atol=0)
    assert vit.intermediate_calls == 2 and vit.native_calls == 0
    changed = pixels.clone()
    changed[:, :, 2:] = torch.randn_like(changed[:, :, 2:]) * 10
    updated = encoder.batch_encode(changed)
    torch.testing.assert_close(updated[:, :, :1], actual[:, :, :1], rtol=0, atol=0)
    for before, after in zip(actual.split(8, 1), updated.split(8, 1)):
        assert not torch.allclose(before[:, :, 1:], after[:, :, 1:])
    # Deploy's condition-only call must reproduce the training clean prefix.
    torch.testing.assert_close(encoder.batch_encode(pixels[:, :, :2]), actual[:, :, :1], rtol=0, atol=0)


@pytest.mark.parametrize("override,match", [
    ({"future_fusion_method": "last_layer"}, "both condition and future"),
    ({"condition_fusion_method": "last_layer"}, "both condition and future"),
    ({"add_input_bottleneck": False}, "add_input_bottleneck=true"),
    ({"fusion_layers": [0, 2]}, "final V-JEPA block"),
    ({"svae_target_dim": 4}, "without S-VAE"),
])
def test_incompatible_concat_configuration_fails_early(override, match):
    cfg = dict(embed_dim=8, variant="tiny", add_input_bottleneck=True,
               condition_fusion_method=METHOD, future_fusion_method=METHOD, fusion_layers=[1, 3])
    cfg.update(override)
    with pytest.raises(ValueError, match=match):
        VJEPA21VideoEncoder(RepaGridViT(), **cfg)


def test_concat_rejects_repa():
    encoder = make_encoder(RepaGridViT())
    with pytest.raises(ValueError, match="repa_enabled=false"):
        encoder.resolve_repa_target_layers("last_layer", [3])


@pytest.mark.parametrize("layers", [[29, 39], [9, 19, 29, 39]])
def test_full_1_3b_projection_sizes_on_meta(tmp_path, layers):
    from openwam.model.video_backbone import build_video_backbone
    from tests.test_vjepa_dev_launchers import config
    from tests.test_wan_text_embedding_cache import make_cache

    cfg = config("1_3b", env_overrides={
        "CONDITION_FUSION_METHOD": METHOD, "FUTURE_FUSION_METHOD": METHOD,
        "FUSION_LAYERS": str(layers), "REPA_ENABLED": "false",
        # Use the tiny test cache; text width is the same 4096 as Cosmos3 avg.
        "TEXT_ENCODER_TYPE": "umt5-xxl",
    })
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    vit = torch.nn.Module()
    vit.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(40)])
    encoder = VJEPA21VideoEncoder(
        vit, history_num_tubelets=0, embed_dim=1408, variant="test", add_input_bottleneck=True, bottleneck_dim=256,
        condition_fusion_method=METHOD, future_fusion_method=METHOD, fusion_layers=layers,
    )
    source = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
    backbone = build_video_backbone(
        cfg.model.video_backbone.name, {}, source=source, device="meta", external_encoder=encoder,
    )
    backbone.reinit_for_from_scratch(external_encoder=encoder, source=source)
    channels = len(layers) * 1408
    projection = backbone.dit.patch_embedding
    assert projection.clean_proj.weight.shape == (1536, channels, 1, 2, 2)
    assert projection.proj1.weight.shape == (256, channels, 1, 2, 2)
    assert projection.proj2.weight.shape == (1536, 256, 1, 1, 1)
    assert backbone.dit.head.head.weight.shape == (4 * channels, 1536)


@pytest.mark.parametrize("layers,pool,views", [([1, 3], False, True), ("all_layers", True, False)])
def test_joint_training_checkpoint_and_plus_deployment(tmp_path, concat_source, monkeypatch, layers, pool, views):
    original = integration.launcher_config

    def config():
        cfg = original()
        cfg.model.video_backbone.encoder.condition_fusion_method = METHOD
        cfg.model.video_backbone.encoder.future_fusion_method = METHOD
        cfg.model.video_backbone.encoder.fusion_layers = layers
        cfg.model.architecture.repa_enabled = False
        return cfg

    monkeypatch.setattr(integration, "launcher_config", config)
    integration.test_joint_forward_backward_and_saved_checkpoint(
        tmp_path, concat_source, wan_grid=True, temporal_pooling=pool,
        history=True, separate_views=views, input_bottleneck=True,
    )
    checkpoint = tmp_path / "checkpoint"
    np.save(checkpoint / "normalization_stats.npy", {"test": 1})
    expected_layers = list(range(4)) if layers == "all_layers" else layers
    metadata = json.loads((checkpoint / "manifest.json").read_text())["feature_fusion"]
    assert metadata["source_norms"] == ["ln0"] * (len(expected_layers) - 1) + ["pretrained_final"]
    assert metadata["feature_norm"] == "per_source"
    assert metadata["source_channels"] == 8
    assert metadata["latent_channels"] == 8 * len(expected_layers)
    prepared = prepare_checkpoint(checkpoint, "model.safetensors", tmp_path / "deploy", tmp_path / "plus-cache")
    record = json.loads((prepared / "deployment_preparation.json").read_text())
    assert record["encoder_latent_dim"] == 8 * len(expected_layers)
    cfg = OmegaConf.load(prepared / "config.yaml")
    restored = VJEPA21VideoEncoder.from_skeleton(
        {}, encoder_cfg=cfg.model.video_backbone.encoder, ckpt_dir=str(prepared),
    )
    assert restored.fusion_layers == tuple(expected_layers)
    train_cfg = OmegaConf.load(checkpoint / "config.yaml")
    backbone = integration.Wan22Style.from_pretrained(
        OmegaConf.to_container(train_cfg.model.video_backbone, resolve=True),
        external_encoder=restored, device="cpu", ckpt_dir=str(checkpoint),
    )
    backbone.reinit_for_from_scratch(external_encoder=restored, source=train_cfg.model.video_backbone)
    backbone.set_dtype_device(torch.float32, torch.device("cpu"))
    inference = backbone.preprocess_input_for_inference(
        prompt="open drawer", num_frames=5, height=64, width=64, num_inference_steps=4,
    )
    noise = inference["noise"]
    assert noise.shape[1] == 8 * len(expected_layers)
    assert inference["latents"].shape == noise.shape
    for source in noise.split(8, 1)[1:]:
        assert not torch.equal(source, noise[:, :8])
    # These errors can otherwise evade shape-only loading, notably layer changes
    # with the same source count and changing the original layer's norm policy.
    manifest_path = checkpoint / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    for bad in ("missing_metadata", "wrong_norm", "wrong_depth", "wrong_channels", "wrong_layers"):
        manifest = copy.deepcopy(original_manifest)
        fusion = manifest["feature_fusion"]
        if bad == "missing_metadata":
            del manifest["feature_fusion"]
        elif bad == "wrong_norm":
            fusion["source_norms"][-1] = "ln0"
        elif bad == "wrong_depth":
            fusion["encoder_depth"] += 1
        elif bad == "wrong_channels":
            fusion["latent_channels"] += 8
        else:
            fusion["fusion_layers"][0] = 99
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            VJEPA21VideoEncoder.from_skeleton({}, encoder_cfg=cfg.model.video_backbone.encoder, ckpt_dir=str(checkpoint))
        with pytest.raises(ValueError):
            prepare_checkpoint(checkpoint, "model.safetensors", tmp_path / bad, tmp_path / "plus-cache")
        assert not (tmp_path / bad).exists()
    manifest_path.write_text(json.dumps(original_manifest))
    weights = load_file(str(checkpoint / "model.safetensors"))
    weights["video_backbone.dit.head.head.weight"] = weights["video_backbone.dit.head.head.weight"][:-1]
    save_file(weights, str(checkpoint / "bad_head.safetensors"))
    with pytest.raises(ValueError):
        prepare_checkpoint(checkpoint, "bad_head.safetensors", tmp_path / "bad-head", tmp_path / "plus-cache")


@pytest.mark.parametrize("distributed", [False, True])
def test_source_losses_survive_metric_reduction(distributed):
    accelerator = SimpleNamespace(num_processes=2, gather=lambda x: x.repeat(2, 1)) if distributed else None
    losses = {"total": torch.tensor(5.), "video": torch.tensor(2.), "action": torch.tensor(3.),
              "video_layer_1": torch.tensor(1.), "video_layer_3": torch.tensor(3.)}
    metrics = reduce_step_metrics(accelerator, losses, torch.tensor(4.))
    assert metrics["loss_video_layer_1"] == 1.
    assert metrics["loss_video_layer_3"] == 3.
    assert (metrics["loss_video_layer_1"] + metrics["loss_video_layer_3"]) / 2 == metrics["loss_video"]


def test_source_losses_reach_trainer_logs_and_csv(tmp_path):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    result = {key: torch.tensor(value) for key, value in {
        "loss": 5., "loss_video": 2., "loss_action": 3.,
        "loss_video_layer_1": 1., "loss_video_layer_3": 3.,
    }.items()}
    trainer = SimpleNamespace(
        accelerator=None, lambda_video=1., lambda_action=1.,
        architecture=SimpleNamespace(prepare_inputs=lambda _: {"actions": torch.zeros(1)},
                                     compute_loss=lambda **_: result),
    )
    losses = OpenWAMTrainer.compute_loss(trainer, {})
    metrics = reduce_step_metrics(None, losses, torch.tensor(0.))
    records = []
    OpenWAMTrainer.log_step(
        trainer, metrics=metrics, global_step=1, opt_step=1, epoch=0, lr=1e-4,
        steps_per_sec=1., batch_size=1, pbar=None,
        wandb_run=SimpleNamespace(log=lambda record, **_: records.append(record)),
        debug=True, output_path=str(tmp_path),
    )
    with (tmp_path / "debug_loss_history.csv").open() as stream:
        row = next(csv.DictReader(stream))
    for layer, value in [(1, 1.), (3, 3.)]:
        assert records[0][f"train/loss_video_layer_{layer}"] == value
        assert float(row[f"loss_video_layer_{layer}"]) == value


@pytest.mark.parametrize("source_count", [2, 4])
@pytest.mark.parametrize("prediction_type", ["x", "v"])
def test_split_loss_matches_full_loss_and_gradients_with_padding(monkeypatch, source_count, prediction_type):
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    encoder = make_encoder(RepaGridViT(), [1, 3] if source_count == 2 else "all_layers")
    monkeypatch.setattr(type(arch.video_backbone), "video_encoder", property(lambda _: encoder), raising=False)
    arch.video_prediction_type = prediction_type
    arch.video_backbone.scheduler.linear_timesteps_weights = torch.linspace(0.2, 1.8, 1000)
    clean = torch.randn(2, 8 * source_count, 4, 2, 2)
    prediction = torch.randn_like(clean, requires_grad=True)
    seen = []

    def forward(*args, **inputs):
        seen.append(inputs["latents"].clone())
        return prediction, None

    monkeypatch.setattr(arch, "forward", forward)
    inputs = dict(input_latents=clean, clean_condition_latents=clean[:, :, :1],
                  video_is_pad=torch.tensor([[False, False, True], [False, True, True]]))
    torch.manual_seed(123)
    split = arch.compute_loss(lambda_action=0, lambda_video=0.7, **inputs)
    grad_split = torch.autograd.grad(split["loss"], prediction)[0]
    monkeypatch.setattr(type(arch.video_backbone), "video_encoder", property(lambda _: None))
    torch.manual_seed(123)
    full = arch.compute_loss(lambda_action=0, lambda_video=0.7, **inputs)
    grad_full = torch.autograd.grad(full["loss"], prediction)[0]
    torch.testing.assert_close(split["loss"], full["loss"])
    torch.testing.assert_close(grad_split, grad_full)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    torch.testing.assert_close(seen[0][:, :, :1], clean[:, :, :1], rtol=0, atol=0)
    assert grad_split[:, :, 0].count_nonzero() == 0
    assert grad_split[:, :, -1].count_nonzero() == 0
    assert grad_split[1, :, -2].count_nonzero() == 0
    for group in grad_split.split(8, 1):
        assert group[:, :, 1].abs().sum() > 0


@pytest.mark.parametrize("source_count", [2, 4])
@pytest.mark.parametrize("prediction_type", ["x", "v"])
@pytest.mark.parametrize("batched", [False, True])
def test_inference_updates_all_sources_and_clamps_clean_prefix(monkeypatch, source_count, prediction_type, batched):
    from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    arch.video_prediction_type = prediction_type
    channels = 8 * source_count
    initial = torch.randn(1, channels, 3, 2, 2)
    ref = torch.randn_like(initial[:, :, :1])
    initial[:, :, :1] = ref
    # Each source has its own target/velocity. This catches only updating the
    # original layer, or mistakenly reusing one source's prediction for all.
    pred = torch.arange(1, source_count + 1, dtype=torch.float32).repeat_interleave(8).view(1, channels, 1, 1, 1)
    seen = []

    def forward(actions, *args, **inputs):
        seen.append(inputs["latents"].clone())
        return pred.expand_as(initial), torch.zeros_like(actions)

    monkeypatch.setattr(arch, "forward", forward)
    inputs = {"latents": initial.clone(), "clean_condition_latents": ref}
    schedule = [(1000, 1000), (750, 750), (500, 500), (250, 250), (0, 0)]
    if batched:
        # Exercise the actual harness loop; model/shape reconstruction is covered
        # by the real joint checkpoint test above, independent of this oracle.
        engine = object.__new__(BatchedOpenWAMEngine)
        engine.architecture = arch
        engine.native = SimpleNamespace(_dit_cache=None)
        monkeypatch.setattr(engine, "_prepare", lambda _: (inputs, torch.randn(1, 4, arch.action_dim), schedule))
        engine.generate_batch([{}])
    else:
        monkeypatch.setattr(arch.video_backbone, "preprocess_input_for_inference", lambda **_: inputs, raising=False)
        arch.generate(schedule, "test", num_frames=5, decode_video=False)
    assert len(seen) == 4
    for step, state in enumerate(seen):
        torch.testing.assert_close(state[:, :, :1], ref, rtol=0, atol=0)
        expected = initial - step * 0.25 * pred if prediction_type == "v" else initial + step * 0.25 * (pred - initial)
        torch.testing.assert_close(state[:, :, 1:], expected[:, :, 1:])
        if step:
            for before, after in zip(initial.split(8, 1), state.split(8, 1)):
                assert not torch.allclose(before[:, :, 1:], after[:, :, 1:])


@pytest.mark.parametrize("layers", ["[29,39]", "[9,19,29,39]", "all_layers"])
def test_dev_launcher_exposes_concat_without_starting_training(layers):
    from tests.test_vjepa_dev_launchers import config

    cfg = config("1_3b", env_overrides={
        "CONDITION_FUSION_METHOD": METHOD, "FUTURE_FUSION_METHOD": METHOD,
        "FUSION_LAYERS": layers, "REPA_ENABLED": "false",
    })
    enc = cfg.model.video_backbone.encoder
    assert enc.condition_fusion_method == enc.future_fusion_method == METHOD
    assert enc.fusion_layers == (layers if layers == "all_layers" else json.loads(layers))
    assert enc.add_input_bottleneck is True and enc.bottleneck_dim == 256
    assert cfg.model.architecture.repa_enabled is False
