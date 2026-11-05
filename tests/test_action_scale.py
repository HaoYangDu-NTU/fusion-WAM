"""Action scaling at training, proprio, checkpoint and deployment boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
from openwam.deploy import model_loader
from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture
from openwam.train.utils import ckpt_model_loader
from tests.test_architecture_variants import _build_arch, _build_shared_moe_arch
from tests.test_dual_system_idm import _GenerateVideoBackbone, _make_idm_with_video
from tests.test_finetune_config_override import _live_cfg, _write_ckpt_cfg
from tests.test_flow_prediction import TinyArchitecture
from tests.test_openwam_trainer import _make_fake_loss_inputs
from tests.test_vla_harness_batch_engine import CPUArchitecture, Encoder


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), -float("inf"), None, "invalid"])
def test_invalid_scale_rejected_before_building_backbone(scale):
    with patch.object(TinyArchitecture, "_init_video_backbone") as build:
        with pytest.raises(ValueError, match="action_scale"):
            TinyArchitecture({"action_scale": scale})
    build.assert_not_called()


def test_legacy_scale_is_identity():
    arch = TinyArchitecture()
    value = torch.randn(2, 3)
    assert arch.action_scale == 1.0
    assert arch._scale_action_input(value) is value
    assert arch._scale_action_input(None) is None


def test_proprio_scaled_once_with_mask_and_gradient_isolation():
    arch = TinyArchitecture({"action_scale": 2.0})
    arch._init_proprio_context({"use_proprioception": True, "state_dim": 3}, text_dim=3)
    with torch.no_grad():
        arch.proprio_encoder.weight.copy_(torch.eye(3))
        arch.proprio_encoder.bias.fill_(0.4)
    proprio = torch.tensor([[0.1, -0.2, 0.0], [0.3, 0.4, 0.5]], requires_grad=True)
    original = proprio.detach().clone()
    inputs = {"context": torch.zeros(2, 2, 3), "_proprio_sample_mask": torch.tensor([True, False])}
    for _ in range(2):
        out = arch._append_proprio_context_token(inputs, proprio)
        torch.testing.assert_close(out["context"][:, -1], (original * 2 + 0.4) * torch.tensor([[1.0], [0.0]]))
        assert out["context_mask"][:, -1].tolist() == [True, False]
    out["context"].sum().backward()
    torch.testing.assert_close(proprio.grad, torch.tensor([[2.0] * 3, [0.0] * 3]))
    torch.testing.assert_close(proprio.detach(), original)


@pytest.mark.parametrize("variant", ["vanilla", "moe"])
def test_single_system_proprio_scaled_at_encoder_without_mutation(variant):
    cfg = {"framework": "single_system", "variant": variant, "action_dim": 7,
           "use_proprioception": True, "state_dim": 7, "action_scale": 2.0,
           "expert_ffn_dim": 64, "bridge_layers": [0]}
    arch = (_build_shared_moe_arch(cfg, num_layers=1) if variant == "moe"
            else _build_arch("single_system_vanilla", cfg, num_layers=1))
    arch.init_training_schedulers(10)
    inputs = _make_fake_loss_inputs(B=1, action_dim=7, T_action=5, video_dim=64)
    proprio = torch.linspace(-1, 1, 7).view(1, 7)
    original = proprio.clone()
    with patch.object(arch.action_backbone, "encode_state", wraps=arch.action_backbone.encode_state) as encode:
        for _ in range(2):
            result = arch.compute_loss(**inputs, proprio=proprio, actions=torch.randn(1, 5, 7))
            assert torch.isfinite(result["loss"])
        for call in encode.call_args_list:
            torch.testing.assert_close(call.args[0], original * 2)
    torch.testing.assert_close(proprio, original)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_idm_loss_scales_clean_action_without_changing_weights(scale):
    class TinyIDM(TinyArchitecture):
        compute_loss = DualSystemIDMArchitecture.compute_loss
        video_cond_noise_prob = 0.0

    arch = TinyIDM({"action_scale": scale})
    actions = torch.tensor([[[-0.3, 0.0, 0.7], [0.2, -0.1, 0.4]]])
    video = torch.zeros(1, 1, 2, 1, 1)
    noise = torch.full_like(actions, 0.6)
    with patch("torch.randn_like", side_effect=[torch.zeros_like(video), noise]):
        result = arch.compute_loss(actions=actions, input_latents=video, lambda_video=0, lambda_action=1.2)
    clean = actions * scale
    torch.testing.assert_close(arch.noisy_actions, 0.98 * clean + 0.02 * noise)
    expected = (arch.outputs[1] - (noise - clean)).square().mean() * 1.7 * 1.2
    torch.testing.assert_close(result["loss"], expected)
    expected_grad = torch.autograd.grad(expected, arch.weight, retain_graph=True)[0]
    result["loss"].backward()
    torch.testing.assert_close(arch.weight.grad, expected_grad)


@pytest.mark.parametrize("with_normalizer", [False, True])
def test_idm_unscales_before_normalizer(monkeypatch, with_normalizer):
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    arch = _make_idm_with_video(_GenerateVideoBackbone())
    kwargs = dict(schedule=[(1000., 1000.), (0., 500.), (0., 0.)], prompt="",
                  num_frames=3, action_num_frames=5, decode_video=False, seed=42)
    torch.manual_seed(42)
    raw = arch.generate(**kwargs)["actions"]
    arch.action_scale = 2.0
    if with_normalizer:
        arch.normalizer = SimpleNamespace(unnormalize=lambda x: x * 3.0 + 100.0)
    torch.manual_seed(42)
    scaled = arch.generate(**kwargs)["actions"]
    expected = raw / 2.0
    if with_normalizer:
        expected = expected * 3.0 + 100.0
    np.testing.assert_allclose(scaled, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("with_normalizer", [False, True])
def test_perfect_x_native_and_batch_return_physical_actions(scale, with_normalizer):
    clean = torch.linspace(-0.8, 0.1, 10)
    seen_noise = []

    class PerfectX(CPUArchitecture):
        def forward(self, actions, action_time, *, latents, **kwargs):
            seen_noise.append(actions.clone())
            return torch.zeros_like(latents), (clean * self.action_scale).expand_as(actions)

    arch = PerfectX(inactive=False)
    arch.action_scale, arch.action_prediction_type = scale, "x"
    arch.video_backbone.video_encoder = Encoder()
    arch.binary_command_dims = (9,)
    arch.normalizer = (SimpleNamespace(unnormalize=lambda x: x * 3.0 + 0.25)
                       if with_normalizer else None)
    obs = [{"image": i + 2, "prompt": "0", "state": np.zeros(10)} for i in range(2)]
    expected = clean.numpy()[None].repeat(4, axis=0)
    if with_normalizer:
        expected = expected * 3.0 + 0.25
    native = arch.generate([(1000., 1000.), (0., 0.)], "0", clean_condition_images=[2],
                           num_frames=3, action_num_frames=5, decode_video=False,
                           seed=42, proprio=torch.zeros(10))["actions"]
    np.testing.assert_allclose(native, expected, atol=2e-6, rtol=0)
    unit_noise = torch.randn(1, 4, 10, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(seen_noise[0], unit_noise, rtol=0, atol=0)
    cfg = SimpleNamespace(inference=SimpleNamespace(
        num_frames=5, video_num_frames=3, denoise_steps=1, height=4, width=4,
    ))
    engine = BatchedOpenWAMEngine.__new__(BatchedOpenWAMEngine)
    engine.architecture, engine.cfg = arch, cfg
    engine.native = SimpleNamespace(_prompt_embed_cache={}, _dit_cache=None)
    actual = engine.generate_batch(obs)
    expected[:, 9] = 1.0 if with_normalizer else -1.0
    for result in actual:
        np.testing.assert_allclose(result, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(seen_noise[-1], unit_noise.expand(2, -1, -1), rtol=0, atol=0)


@pytest.mark.parametrize("saved_scale", [None, 2.0])
def test_checkpoint_deploy_resume_and_finetune_scale(tmp_path, monkeypatch, saved_scale):
    # Exercise real config resolution/loaders; stub only heavyweight weights/backbones.
    ckpt_dir = _write_ckpt_cfg(tmp_path, **(
        {} if saved_scale is None else {"architecture__action_scale": saved_scale}
    ))
    monkeypatch.setattr("openwam.model.build_architecture", lambda name, params: TinyArchitecture(params))
    monkeypatch.setattr(TinyArchitecture, "load_checkpoint", lambda self, path: None)
    monkeypatch.setattr(TinyArchitecture, "set_dtype_device", lambda self, dtype, device: None)
    monkeypatch.setattr(model_loader, "_find_latest_checkpoint", lambda path: "unused.safetensors")
    monkeypatch.setattr(model_loader, "_build_normalizer", lambda cfg, path: None)
    monkeypatch.setattr(ckpt_model_loader, "find_latest_weights", lambda path: "unused.safetensors")
    original = (Path(ckpt_dir) / "config.yaml").read_bytes()
    _, deployed = model_loader.load_from_checkpoint_dir(ckpt_dir, device="cpu")
    _, resumed, _ = ckpt_model_loader.build_architecture_from_ckpt_dir(
        ckpt_dir, weights_required=False, override_cfg=None,
    )
    expected = 1.0 if saved_scale is None else saved_scale
    assert deployed.action_scale == resumed.action_scale == expected
    live = _live_cfg(architecture__action_scale=2.0)
    _, finetuned, _ = ckpt_model_loader.build_architecture_from_ckpt_dir(
        ckpt_dir, weights_required=True, override_cfg=live,
    )
    assert finetuned.action_scale == live.model.architecture.action_scale == 2.0
    assert (Path(ckpt_dir) / "config.yaml").read_bytes() == original
