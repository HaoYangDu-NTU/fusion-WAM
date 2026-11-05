"""Spatial cosine math, gradient ownership, real MoT training and serialization."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.utils.repa import (
    masked_spatial_cosine_loss,
    spatial_normalize,
    video_grid_to_view_maps,
)
from openwam.model.video_backbone.encoder.bottleneck import BottleneckPatchEmbed
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.training_utils import reduce_step_metrics
from tests.test_vjepa21_repa import RepaGridViT
from tests.test_vjepa_dev_launchers import config, launcher
from tests.test_wan22_style_capacity import launcher_config
from tests.test_wan_text_embedding_cache import make_cache


def test_spatial_math_and_view_layout_match_independent_reference():
    grid = torch.randn(2, 8, 3, 4, 10)
    maps = video_grid_to_view_maps(grid, 2)
    expected = torch.stack([grid[b, :, t, :, v * 5:(v + 1) * 5]
                            for b in range(2) for t in range(3) for v in range(2)])
    torch.testing.assert_close(maps, expected, rtol=0, atol=0)
    tokens = maps.flatten(2)
    reference = (tokens - 0.8 * tokens.mean(-1, keepdim=True)) / (tokens.std(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(spatial_normalize(maps).flatten(2), reference)
    altered = maps.clone()
    altered[1:] *= 30
    torch.testing.assert_close(spatial_normalize(altered)[0], spatial_normalize(maps)[0])
    assert torch.isfinite(spatial_normalize(torch.ones(2, 8, 1, 1))).all()


def test_cosine_padding_and_all_padded_backward():
    student = torch.randn(3, 8, 2, 2, requires_grad=True)
    target = torch.randn_like(student)
    mask = torch.tensor([True, False, True])
    expected = 1 - torch.nn.functional.cosine_similarity(student[[0, 2]], target[[0, 2]], dim=1).mean()
    loss = masked_spatial_cosine_loss(student, target, mask)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(student.grad[1]) == 0
    student.grad = None
    empty = masked_spatial_cosine_loss(student, target, torch.zeros(3, dtype=torch.bool))
    empty.backward()
    assert empty.item() == 0 and torch.count_nonzero(student.grad) == 0


def tiny_model(tmp_path, *, bottleneck=True, views=False, enabled=True, history=1):
    cfg = launcher_config()
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in {"dim": 96, "ffn_dim": 192, "num_heads": 4, "num_layers": 2, "freq_dim": 32}.items():
        cfg.model.video_backbone.dit_config[key] = value
    encoder = VJEPA21VideoEncoder(
        RepaGridViT(), embed_dim=8, variant="test", use_avg_t_pooling=False,
        history_num_tubelets=history, seperate_encode_view=views,
        add_input_bottleneck=bottleneck, bottleneck_dim=4,
    )
    backbone = Wan22Style.from_pretrained(cfg, external_encoder=encoder, device="cpu")
    backbone.reinit_for_from_scratch(external_encoder=encoder)
    arch_cfg = dict(
        framework="dual_system", variant="joint_self_attn", dim=48, ffn_dim=96,
        action_dim=7, num_heads=4, attn_head_dim=24, text_dim=4096, use_proprioception=False, bridge_interval=1,
        video_prediction_type="x", action_prediction_type="x", attention_mask_mode="mutual",
        repa_enabled=enabled, repa_layer_depth=1, repa_coeff=0.05,
        repa_target_fusion_method="mean", repa_target_fusion_layers=[0, 2, 3],
    )
    with patch.object(BaseWAMArchitecture, "_init_video_backbone", lambda self, _: setattr(self, "video_backbone", backbone)):
        model = DualSystemSelfAttnArchitecture(arch_cfg)
    model.to(dtype=torch.float32)
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    model.init_training_schedulers(1000)
    model.freeze_modules(["video_backbone.video_encoder"])
    return model


@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("bottleneck,views", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_real_preprocess_joint_loss_gradients_and_checkpoint(tmp_path, bottleneck, views, checkpoint, dtype, history):
    torch.manual_seed(29)
    model = tiny_model(tmp_path, bottleneck=bottleneck, views=views, history=history)
    encoder = model.video_backbone.video_encoder
    h, w = (224, 448) if views else (32, 32)
    rng = np.random.default_rng(29)
    frames = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for _ in range(5)]
    inputs = model.preprocess(frames=[frames], text=["open drawer"], ref_images=[[frames[0]] * (2 * history)])
    assert inputs["input_latents"].shape[2] == history + 2 and inputs["repa_target_latents"].shape[2] == 2
    assert not inputs["repa_target_latents"].requires_grad
    # Turning off the auxiliary request changes neither the native latent nor RNG.
    model.repa_enabled = False
    baseline = model.preprocess(frames=[frames], text=["open drawer"], ref_images=[[frames[0]] * (2 * history)])
    torch.testing.assert_close(inputs["input_latents"], baseline["input_latents"], rtol=0, atol=0)
    assert "repa_target_latents" not in baseline
    model.repa_enabled = True
    # The encoder fixture uses CPU avg_pool3d, which lacks bf16 support.
    # Exercise the actual DiT, shared projection and loss in both training dtypes.
    model.to(dtype=dtype)
    model.set_dtype_device(dtype, torch.device("cpu"))
    for batch in (inputs, baseline):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                batch[key] = value.to(dtype)
    actions = torch.randn(1, 4, 7, dtype=dtype)
    inputs["video_is_pad"] = torch.tensor([[False, True]])
    result = model.compute_loss(actions=actions, use_gradient_checkpointing=checkpoint, **inputs)
    assert torch.isfinite(result["loss"]) and result["loss_repa"] > 0
    result["loss"].backward()
    assert model.repa_projector.weight.grad.abs().sum() > 0
    projection = model.video_backbone.dit.patch_embedding
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projection.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    params = build_trainable_parameters(SimpleNamespace(architecture=model))
    assert len({id(p) for p in params}) == len(params)
    assert any(p is model.repa_projector.weight for p in params)
    # Inference uses the ordinary two-result forward, with no target/head call.
    forward_inputs = {k: v for k, v in baseline.items() if k != "input_latents"}
    timestep = torch.tensor([500.0], dtype=dtype)
    forward_inputs.update(latents=inputs["input_latents"], timestep=timestep)
    model.eval()
    with torch.no_grad(), patch.object(model.repa_projector, "forward", side_effect=AssertionError("inference head")):
        expected = model(actions, timestep, **forward_inputs)
    assert len(expected) == 2
    model.save_checkpoint(str(tmp_path / "model.safetensors"))
    restored = copy.deepcopy(model)
    with torch.no_grad():
        restored.repa_projector.weight.zero_()
    restored.load_checkpoint(str(tmp_path / "model.safetensors"), strict=True)
    torch.testing.assert_close(restored.repa_projector.weight, model.repa_projector.weight)
    with torch.no_grad():
        actual = restored(actions, timestep, **forward_inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("history", [1, 2])
def test_teacher_has_no_gradient_but_student_updates_shared_future_projection(tmp_path, history):
    model = tiny_model(tmp_path, history=history)
    vb = model.video_backbone
    projection = vb.dit.patch_embedding
    assert isinstance(projection, BottleneckPatchEmbed)
    inputs = {"input_latents": torch.randn(1, 8, history + 2, 4, 4),
              "clean_condition_latents": torch.randn(1, 8, history, 4, 4),
              "repa_target_latents": torch.randn(1, 8, 2, 4, 4, requires_grad=True)}
    # A leaf hidden state isolates the teacher branch from the input projection.
    hidden = torch.randn(1, (history + 2) * 4, 96, requires_grad=True)
    model._compute_repa_loss(hidden, inputs).backward()
    assert hidden.grad.abs().sum() > 0
    assert inputs["repa_target_latents"].grad is None
    assert all(p.grad is None for p in projection.parameters())
    assert torch.count_nonzero(hidden.grad[:, :4 * history]) == 0
    model.zero_grad(set_to_none=True)
    # Through the student input, the same parameters do receive auxiliary gradients.
    projected = projection(inputs["input_latents"], num_clean_condition_latents=history)
    hidden = projected.permute(0, 2, 3, 4, 1).flatten(1, 3)
    model._compute_repa_loss(hidden, inputs).backward()
    assert projection.proj1.weight.grad.abs().sum() > 0
    assert projection.proj2.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(projection.clean_proj.weight.grad) == 0


def test_disabled_checkpoint_has_no_auxiliary_parameters(tmp_path):
    model = tiny_model(tmp_path, enabled=False)
    assert model.repa_projector is None
    assert not any(name.startswith("repa_") for name in model.state_dict())


def test_video_only_training_still_applies_auxiliary_loss(tmp_path):
    model = tiny_model(tmp_path)
    latents = torch.randn(1, 8, 3, 4, 4)
    kwargs = dict(input_latents=latents, clean_condition_latents=latents[:, :, :1],
                  repa_target_latents=latents[:, :, 1:], context=torch.randn(1, 4, 4096),
                  seq_lens=torch.tensor([4]), fuse_vae_embedding_in_latents=True, lambda_action=0)
    result = model.compute_loss(**kwargs)
    torch.testing.assert_close(result["loss"].detach(), result["loss_video"] + result["loss_repa"])
    assert result["loss_action"].item() == 0
    result["loss"].backward()
    assert model.repa_projector.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("field,value", [("repa_layer_depth", 0), ("repa_layer_depth", 3),
                                       ("repa_layer_depth", 1.5), ("repa_coeff", float("nan")),
                                       ("repa_coeff", -1), ("repa_spatial_norm_gamma", 1.1),
                                       ("repa_target_fusion_method", "concat"),
                                       ("repa_target_fusion_layers", [39])])
def test_invalid_alignment_config_fails_before_training(tmp_path, field, value):
    model = tiny_model(tmp_path, enabled=False)
    setattr(model, field, value)
    with pytest.raises(ValueError):
        model._init_repa_projector()


def test_head_width_is_inferred_and_initialization_preserves_rng(tmp_path):
    model = tiny_model(tmp_path, enabled=False)
    rng = torch.get_rng_state()
    model._init_repa_projector()
    assert torch.equal(torch.get_rng_state(), rng)
    for width in (1024, 1536, 3072):
        model.video_backbone.dit.dim = width
        with torch.device("meta"):
            model._init_repa_projector()
        assert model.repa_projector.weight.shape == (width, width, 3, 3)


@pytest.mark.parametrize("distributed", [False, True])
def test_repa_metric_is_reduced_with_existing_losses(distributed):
    losses = {"total": torch.tensor(1.5), "video": torch.tensor(1.0),
              "action": torch.tensor(0.25), "repa": torch.tensor(0.25)}
    accelerator = SimpleNamespace(num_processes=2, gather=lambda values: values.repeat(2, 1)) if distributed else None
    metrics = reduce_step_metrics(accelerator, losses, torch.tensor(2.0))
    assert metrics["loss_repa"] == 0.25
    assert metrics["loss_total"] == metrics["loss_video"] + metrics["loss_action"] + metrics["loss_repa"]


@pytest.mark.parametrize("layers,expected", [("[39]", [39]), ("all_layers", "all_layers")])
def test_launcher_keeps_main_and_target_selection_independent(layers, expected):
    cfg = config("1_3b", env_overrides={
        "REPA_ENABLED": "true", "REPA_LAYER_DEPTH": "9", "REPA_COEFF": "0.03",
        "REPA_SPATIAL_NORM_GAMMA": "0.7", "REPA_TARGET_FUSION_METHOD": "mean",
        "REPA_TARGET_FUSION_LAYERS": layers,
        "CONDITION_FUSION_METHOD": "last_layer", "FUTURE_FUSION_METHOD": "last_layer",
    })
    arch = cfg.model.architecture
    assert arch.repa_enabled and arch.repa_layer_depth == 9 and arch.repa_coeff == 0.03
    assert arch.repa_spatial_norm_gamma == 0.7 and arch.repa_target_fusion_method == "mean"
    assert arch.repa_target_fusion_layers == expected
    assert cfg.model.video_backbone.encoder.future_fusion_method == "last_layer"
    assert cfg.model.video_backbone.encoder.condition_fusion_method == "last_layer"
    assert config("1_3b").model.architecture.repa_enabled is False
    override = config("1_3b", env_overrides={"REPA_ENABLED": "true"},
                      extra=("model.architecture.repa_enabled=false",))
    assert override.model.architecture.repa_enabled is False


@pytest.mark.parametrize("name,value", [("REPA_ENABLED", "yes"), ("REPA_TARGET_FUSION_METHOD", "concat")])
def test_invalid_launcher_repa_options_fail_before_training(name, value):
    result = launcher("1_3b", env_overrides={name: value})
    assert result.returncode == 2 and "scripts/train.py" not in result.stdout
