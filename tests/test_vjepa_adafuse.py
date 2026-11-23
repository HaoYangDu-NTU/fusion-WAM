"""AdaFuse source semantics, input-only gradients, and train/deploy parity."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file, save_file

from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
from benchmarks.vla_harness.run_eval import make_plan
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.model import build_architecture, resolve_architecture_config
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.encoder.vjepa21_fusion import AdaFuse
from openwam.train.utils.checkpointing import save_config
from tests.test_vjepa21_repa import RepaGridViT
from tests.test_vjepa_concat import concat_source, encoder_source  # noqa: F401
from tests.benchmarks.vla_harness.test_vla_harness_concat import libero_args
from tests.test_wan22_style_capacity import launcher_config
from tests.test_wan_text_embedding_cache import make_cache


def tiny_model(tmp_path, source, *, condition="adafuse", future="adafuse", pool=False, views=False, layers=(1, 3)):
    cfg = launcher_config()
    vb = cfg.model.video_backbone
    vb.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in dict(model_path=str(source), add_input_bottleneck=True, bottleneck_dim=4,
                           condition_fusion_method=condition, future_fusion_method=future,
                           fusion_layers=list(layers) if not isinstance(layers, str) else layers,
                           use_avg_t_pooling=pool, history_num_tubelets=2,
                           seperate_encode_view=views).items():
        vb.encoder[key] = value
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        vb.dit_config[key] = value
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    cfg.model.architecture.repa_enabled = False
    cfg.model.architecture.video_prediction_type = cfg.model.architecture.action_prediction_type = "x"
    cfg.dataloader.normalize_mode = None
    cfg.training.mixed_precision = "bf16"
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    model.init_training_schedulers(1000)
    model.freeze_modules(["video_backbone.video_encoder"])
    model.video_prediction_type = model.action_prediction_type = "x"
    return model, cfg


@pytest.mark.parametrize("count", [2, 4, 40])
@pytest.mark.parametrize("pool", [1, 2])
def test_channelwise_layer_weights_and_norm_pool_order(count, pool):
    torch.manual_seed(7)
    fuser = AdaFuse(count, 8, temporal_pool_stride=pool)
    with torch.no_grad():
        fuser.post_fusion_norm.weight.copy_(torch.linspace(.7, 1.3, 8))
        fuser.post_fusion_norm.bias.copy_(torch.linspace(-.3, .3, 8))
    features = [torch.randn(2, 8, 4, 2, 3) * (i + 1) + i for i in range(count)]
    # Independent oracle: explicit stack and reductions, no production helpers.
    weights = torch.full((count, 8), .2 / (count - 1))
    weights[-1] = .8
    torch.testing.assert_close(fuser.layer_logits.softmax(0), weights)
    normalized = torch.stack([F.layer_norm(x.movedim(1, -1), (8,), eps=1e-5) for x in features])
    mixed = (normalized * weights[:, None, None, None, None, :]).sum(0)
    if pool == 2:
        mixed = mixed.reshape(2, 2, 2, 2, 3, 8).mean(2)
    expected = F.layer_norm(mixed, (8,), fuser.post_fusion_norm.weight, fuser.post_fusion_norm.bias)
    actual = fuser(features)
    torch.testing.assert_close(actual, expected.movedim(-1, 1))
    (actual * torch.randn_like(actual)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in fuser.parameters())
    assert all(x.grad is None for x in features)
    # Channel-specific weights, not one scalar shared by the whole layer.
    with torch.no_grad():
        fuser.layer_logits[0, 0] += 2
    assert not torch.allclose(fuser(features), actual)


@pytest.mark.parametrize("layers", [[0, 2], [1, 3], "all_layers"])
@pytest.mark.parametrize("views", [False, True])
def test_sources_preserve_native_final_and_clean_future_view_boundaries(layers, views):
    torch.manual_seed(8)
    encoder = VJEPA21VideoEncoder(RepaGridViT(), embed_dim=8, variant="test", history_num_tubelets=2,
                                 seperate_encode_view=views, condition_fusion_method="adafuse",
                                 future_fusion_method="adafuse", fusion_layers=layers, use_avg_t_pooling=False)
    video = torch.randn(2, 3, 8, 224 if views else 32, 448 if views else 48)
    condition, future = encoder.batch_encode_fusion_features(video)
    indices = list(range(4)) if layers == "all_layers" else layers
    for j, layer in enumerate(indices):
        cond_views, future_views = [], []
        for view in video.chunk(2, -1) if views else [video]:
            c = list(encoder._m.raw_layers(view[:, :, :4]))[layer]
            f = list(encoder._m.raw_layers(view))[layer][:, 2:]
            if layer == 3:
                c, f = encoder._m.norm(c), encoder._m.norm(f)
            cond_views.append(c.movedim(-1, 1))
            future_views.append(f.movedim(-1, 1))
        torch.testing.assert_close(condition[j], torch.cat(cond_views, -1))
        torch.testing.assert_close(future[j], torch.cat(future_views, -1))
        assert not condition[j].requires_grad and not future[j].requires_grad
    assert encoder._m.intermediate_calls == 2
    condition_only, absent = encoder.batch_encode_fusion_features(video[:, :, :4])
    assert absent is None and encoder._m.intermediate_calls == 3
    for actual, expected in zip(condition_only, condition):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = video.clone()
    changed[:, :, 4:] *= 10
    new_condition, new_future = encoder.batch_encode_fusion_features(changed)
    for before, after in zip(condition, new_condition):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.allclose(new_future[0], future[0])
    with pytest.raises(RuntimeError, match="backbone"):
        encoder.batch_encode(video)


@pytest.mark.parametrize("condition,future", [("adafuse", "last_layer"), ("mean", "adafuse"),
                                                ("adafuse", "adafuse")])
@pytest.mark.parametrize("pool", [False, True])
def test_real_training_both_fuser_gradients_and_static_branch_compatibility(tmp_path, concat_source, condition, future, pool):
    model, _ = tiny_model(tmp_path, concat_source, condition=condition, future=future, pool=pool)
    vb = model.video_backbone
    rng = np.random.default_rng(31)
    frames = [Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for _ in range(9)]
    sample = dict(video=frames, clean_condition_images=frames[:4], prompt="open drawer",
                  proprio=torch.randn(model.proprio_encoder.in_features),
                  action=torch.randn(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                  video_mask=torch.tensor([True] * 5 + [False] * 4))
    inputs = model.prepare_inputs([sample])
    # Cached text is BF16; the tiny CPU training model uses FP32 without AMP.
    inputs["context"] = inputs["context"].float()
    assert inputs["input_latents"] is None
    assert inputs["video_is_pad"].tolist() == ([[False, True]] if pool else [[False, False, True, True]])
    features = inputs["video_layer_features"]
    encoder = vb.video_encoder
    static = copy.deepcopy(encoder)
    static._condition_fusion_method = condition if condition != "adafuse" else "last_layer"
    static._future_fusion_method = future if future != "adafuse" else "last_layer"
    pixels = encoder.preprocess_video_batch([[*frames[:4], *frames[1:]]])
    static_latents = static.batch_encode(pixels)
    fused = vb.fuse_video_features(features)
    assert fused.shape == static_latents.shape
    if condition != "adafuse":
        torch.testing.assert_close(fused[:, :, :2], static_latents[:, :, :2], rtol=0, atol=0)
    if future != "adafuse":
        torch.testing.assert_close(fused[:, :, 2:], static_latents[:, :, 2:], rtol=0, atol=0)
    with torch.no_grad():
        inference = vb.encode_video(encoder.preprocess_video_batch([frames[:4]]))
    torch.testing.assert_close(inference, fused[:, :, :2])
    fusers = [m for m in (vb.condition_fuser, vb.future_fuser) if m is not None]
    if len(fusers) == 2:
        assert fusers[0].layer_logits is not fusers[1].layer_logits
    assert not any(p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for m in fusers for p in m.parameters())
    before = [p.detach().clone() for m in fusers for p in m.parameters()]
    from openwam.train.utils.optimizer_groups import build_trainable_parameters
    parameters = build_trainable_parameters(SimpleNamespace(architecture=model))
    assert all(any(p is candidate for candidate in parameters) for m in fusers for p in m.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    inputs["use_gradient_checkpointing"] = True
    loss = model.compute_loss(**inputs)
    loss["loss"].backward()
    for module in fusers:
        for p in module.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert all(p.grad is None for p in encoder.parameters())
    assert vb.dit.patch_embedding.proj1.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(not torch.equal(old, p) for old, p in zip(before, [p for m in fusers for p in m.parameters()]))


class LossHarness(torch.nn.Module):
    """Exercise the real loss graph with an analytically differentiable predictor."""
    compute_loss = BaseWAMArchitecture.compute_loss

    def __init__(self, prediction_type, coefficient):
        super().__init__()
        self.video_prediction_type = prediction_type
        self.coefficient = coefficient
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        scheduler = SimpleNamespace(timesteps=torch.tensor([300.]), sigmas=torch.tensor([.3]))
        self.video_backbone = SimpleNamespace(scheduler=scheduler)
        self.action_backbone = SimpleNamespace(scheduler=scheduler)
        self.dtype, self.device = torch.float32, torch.device("cpu")
        self._noise_schedule_distribution = "uniform"
        self.repa_enabled = False

    def forward(self, *args, **inputs):
        self.noisy = inputs["latents"]
        return self.coefficient * self.noisy + self.anchor, None

    def _compute_video_loss(self, prediction, target, *args):
        assert not target.requires_grad
        self.prediction, self.target = prediction, target
        return (prediction - target).square().mean()

    def _add_repa_loss(self, result, *args):
        return result


@pytest.mark.parametrize("prediction", ["x", "v"])
@pytest.mark.parametrize("coefficient", [0., .25])
def test_future_gradient_only_through_prediction_not_target_or_conversion_reference(prediction, coefficient):
    model = LossHarness(prediction, coefficient)
    latent = torch.randn(1, 8, 2, 2, 2, requires_grad=True)
    loss = model.compute_loss(input_latents=latent, lambda_action=0)
    loss["loss"].backward()
    derivative = coefficient * .7 * (-1 / .3 if prediction == "x" else 1)
    expected = 2 * (model.prediction.detach() - model.target) / latent.numel() * derivative
    torch.testing.assert_close(latent.grad, expected)
    assert (latent.grad.abs().sum() == 0) == (coefficient == 0)


@pytest.mark.parametrize("condition,future,layers", [("adafuse", "adafuse", [1, 3]),
    ("last_layer", "adafuse", "all_layers"), ("adafuse", "last_layer", [0, 2])])
def test_checkpoint_libero_plus_and_native_batched_inference(tmp_path, concat_source, condition, future, layers):
    model, cfg = tiny_model(tmp_path, concat_source, condition=condition, future=future, views=True, layers=layers)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    # Non-initial weights must survive deployment, including output affine.
    with torch.no_grad():
        for fuser in (model.video_backbone.condition_fuser, model.video_backbone.future_fuser):
            if fuser is not None:
                fuser.layer_logits.add_(torch.randn_like(fuser.layer_logits))
                fuser.post_fusion_norm.weight.mul_(1.4)
                fuser.post_fusion_norm.bias.add_(.2)
    checkpoint = tmp_path / "checkpoint"
    model.video_backbone.save_deploy_assets(str(checkpoint), cfg)
    cfg.model.video_backbone.encoder.model_path = "/unavailable/encoder"
    save_config(str(checkpoint), cfg)
    save_file(model.state_dict(), str(checkpoint / "model.safetensors"))
    plan = make_plan(libero_args(checkpoint))
    assert len(plan["tasks"]) * plan["num_trials"] == 2000
    assert plan["feature_fusion"]["latent_channels"] == 8
    _, deployed = load_from_checkpoint_dir(str(checkpoint), "cpu", "model.safetensors")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(deployed.state_dict()[key], value, rtol=0, atol=0)
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (224, 448, 3), dtype=np.uint8))
    refs = [image] * 4
    native = SimpleNamespace(architecture=deployed, _cfg_scale=1., _decode_video=False, _dit_cache=None,
        _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {"denoise_mode": "sync", "denoise_steps": 2,
        "num_frames": 5, "video_num_frames": 5, "height": 224, "width": 448}}))
    engine = BatchedOpenWAMEngine(native)
    observations = [dict(prompt=p, image=image, clean_condition_images=refs, state=np.zeros(10, np.float32))
                    for p in ("open drawer", "close drawer")]
    calls = []
    future_fuser = deployed.video_backbone.future_fuser
    handle = future_fuser.register_forward_hook(lambda *args: calls.append(1)) if future_fuser is not None else None
    with torch.no_grad():
        batched, _, _ = engine._prepare(observations)
        single = deployed.video_backbone.preprocess_input_for_inference(prompt="open drawer", clean_condition_images=refs,
            num_frames=5, height=224, width=448, num_inference_steps=2)
        original = model.video_backbone.encode_video(model.video_backbone.video_encoder.preprocess_video_batch([refs]))
    torch.testing.assert_close(batched["clean_condition_latents"][:1], single["clean_condition_latents"])
    torch.testing.assert_close(original, single["clean_condition_latents"])
    chunks = engine.generate_batch(observations)
    assert len(chunks) == 2 and all(chunk.shape == (4, 10) and np.isfinite(chunk).all() for chunk in chunks)
    assert not calls  # Future state is denoised directly in the learned space.
    if handle is not None:
        handle.remove()
    np.save(checkpoint / "normalization_stats.npy", {"test": 1})
    prepared = prepare_checkpoint(checkpoint, "model.safetensors", tmp_path / "plus", tmp_path / "plus-cache")
    prepared_cfg = OmegaConf.load(prepared / "config.yaml")
    assert prepared_cfg.model.video_backbone.encoder.future_fusion_method == future
    assert (prepared / "model.safetensors").resolve() == checkpoint / "model.safetensors"
    record = json.loads((prepared / "deployment_preparation.json").read_text())
    assert record["encoder_latent_dim"] == 8
    # Saved indices and normalization cannot be silently changed, even when D/K match.
    manifest_path = checkpoint / "manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    manifest["feature_fusion"]["adafuse"]["source_norms"][0] = "wrong_norm"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        make_plan(libero_args(checkpoint))
    with pytest.raises(ValueError, match="differs"):
        load_from_checkpoint_dir(str(checkpoint), "cpu", "model.safetensors")
    manifest_path.write_text(manifest_text)
    # Both header-only launch validation and strict deploy loading reject missing weights.
    weights = load_file(str(checkpoint / "model.safetensors"))
    branch = "future" if future == "adafuse" else "condition"
    del weights[f"video_backbone.{branch}_fuser.layer_logits"]
    save_file(weights, str(checkpoint / "model.safetensors"))
    with pytest.raises(ValueError, match="fuser weights"):
        make_plan(libero_args(checkpoint))
    with pytest.raises((RuntimeError, ValueError), match="(?i)missing"):
        load_from_checkpoint_dir(str(checkpoint), "cpu", "model.safetensors")


def test_unsupported_combinations_fail_before_training():
    common = dict(embed_dim=8, variant="test", condition_fusion_method="adafuse", fusion_layers=[1, 3])
    for extra in (dict(svae_target_dim=4), dict(fusion_layers=[3]), dict(future_fusion_method="concat_before_bottleneck")):
        with pytest.raises(ValueError):
            VJEPA21VideoEncoder(RepaGridViT(), **(common | extra))
    encoder = VJEPA21VideoEncoder(RepaGridViT(), **common)
    with pytest.raises(ValueError, match="repa_enabled=false"):
        encoder.resolve_repa_target_layers("last_layer", [])


@pytest.mark.parametrize("condition,future,layers", [("adafuse", "last_layer", "[29,39]"),
    ("last_layer", "adafuse", "all_layers"), ("adafuse", "adafuse", "[9,19,29,39]")])
def test_real_launcher_accepts_independent_adafuse_options(condition, future, layers):
    from tests.test_vjepa_dev_launchers import config
    cfg = config("1_3b", env_overrides={"CONDITION_FUSION_METHOD": condition,
        "FUTURE_FUSION_METHOD": future, "FUSION_LAYERS": layers})
    assert cfg.model.video_backbone.encoder.condition_fusion_method == condition
    assert cfg.model.video_backbone.encoder.future_fusion_method == future
    assert cfg.model.video_backbone.encoder.bottleneck_dim == 256
