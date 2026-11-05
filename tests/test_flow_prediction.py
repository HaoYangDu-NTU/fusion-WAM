"""Independent x/v switches: actual training loss, gradients and saved config."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import resolve_architecture_config
from openwam.model.architectures.utils.flow_prediction import prediction_to_velocity
from openwam.train.utils.checkpointing import save_config


class TinyArchitecture(BaseWAMArchitecture):
    def _init_video_backbone(self, cfg):
        pass

    def __init__(self, cfg=None, dtype=torch.float32):
        super().__init__(cfg)
        self._device = torch.device("cpu")
        self._dtype = dtype
        self.weight = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        schedulers = []
        for sigma in (0.2, 0.02):  # independent times, including JiT's clamp region
            scheduler = ActionScheduler()
            scheduler.sigmas = torch.tensor([sigma])
            scheduler.timesteps = scheduler.sigmas * 1000
            scheduler.linear_timesteps_weights = torch.tensor([1.7])
            schedulers.append(scheduler)
        self.video_backbone = SimpleNamespace(scheduler=schedulers[0])
        self.action_backbone = SimpleNamespace(scheduler=schedulers[1])

    def forward(self, actions, action_time, *, latents, **kwargs):
        self.noisy_actions = actions.detach().clone()
        self.outputs = (
            self.weight[0].to(self.dtype).expand_as(latents),
            self.weight[1].to(self.dtype).expand_as(actions),
        )
        return self.outputs


@pytest.mark.parametrize("video_mode,action_mode", [("v", "v"), ("x", "v"), ("v", "x"), ("x", "x")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("action_scale", [1.0, 2.0])
def test_actual_loss_and_gradients(video_mode, action_mode, dtype, action_scale):
    arch = TinyArchitecture(
        {"video_prediction_type": video_mode, "action_prediction_type": action_mode,
         "action_scale": action_scale}, dtype
    )
    video = torch.linspace(-1, 1, 24).reshape(2, 1, 3, 2, 2).to(dtype)
    actions = torch.linspace(-0.5, 0.5, 12).reshape(2, 2, 3).to(dtype)
    vn, an = torch.full_like(video, 0.7), torch.full_like(actions, -0.6)
    video_pad = torch.tensor([[False, True], [False, False]])
    action_pad = torch.tensor([[[False, True, False], [True, True, True]],
                               [[False, False, False], [False, True, False]]])
    with patch("torch.randn_like", side_effect=[vn, an]):
        result = arch.compute_loss(
            actions=actions, input_latents=video, clean_condition_latents=video[:, :, :1],
            video_is_pad=video_pad, action_is_pad=action_pad,
            lambda_video=0.8, lambda_action=1.2,
        )
    scaled_actions = actions * action_scale
    sigma_a = arch.action_scheduler.sigmas.to(dtype).view(1, 1, 1)
    torch.testing.assert_close(arch.noisy_actions, (1 - sigma_a) * scaled_actions + sigma_a * an)
    # Algebraically independent oracle: unchanged weights, scaled clean target, unit Gaussian noise.
    errors = []
    for mode, pred, clean, noise, scheduler in zip(
        (video_mode, action_mode), arch.outputs, (video, scaled_actions), (vn, an),
        (arch.video_scheduler, arch.action_scheduler),
    ):
        if mode == "x":
            sigma = scheduler.sigmas.to(dtype).float().clamp_min(0.05)
            errors.append((pred.float() - clean.float()).square() / sigma.square())
        else:
            errors.append((pred.float() - (noise - clean).float()).square())
    ev = errors[0][:, :, 1:].mean(dim=(1, 3, 4))
    mv, ma = (~video_pad).float(), (~action_pad).float()
    lv = ((ev * mv).sum(1) / mv.sum(1)).mean() * 1.7
    la = ((errors[1] * ma).sum((1, 2)) / ma.sum((1, 2))).mean() * 1.7
    expected = 0.8 * lv + 1.2 * la
    torch.testing.assert_close(result["loss"], expected)
    torch.testing.assert_close(result["loss_video"], 0.8 * lv)
    torch.testing.assert_close(result["loss_action"], 1.2 * la)
    expected_grad = torch.autograd.grad(expected, arch.weight, retain_graph=True)[0]
    result["loss"].backward()
    torch.testing.assert_close(arch.weight.grad, expected_grad)
    assert torch.isfinite(arch.weight.grad).all()


@pytest.mark.parametrize("sigma", [0.0, 0.001, 0.05, 0.6, 1.0])
def test_x_velocity_sign_floor_and_gradient(sigma):
    x = torch.tensor([0.25, -0.5])
    noise = torch.tensor([-0.5, 1.0])
    noisy = (1 - sigma) * x + sigma * noise
    prediction = (x + 0.125).requires_grad_()
    v = prediction_to_velocity(prediction, noisy, sigma, "x")
    expected = (sigma * (noise - x) - 0.125) / max(sigma, 0.05)
    torch.testing.assert_close(v, expected)
    v.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(x, -1 / max(sigma, 0.05)))


def test_v_is_exact_identity_and_legacy_config_defaults():
    pred = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
    assert prediction_to_velocity(pred, None, None, "v") is pred
    arch = TinyArchitecture()
    assert arch.video_prediction_type == arch.action_prediction_type == "v"


@pytest.mark.parametrize("field", ["video_prediction_type", "action_prediction_type"])
def test_bad_modes_fail_at_construction(field):
    with pytest.raises(ValueError, match=field):
        TinyArchitecture({field: "epsilon"})


def test_idm_rejects_unimplemented_x_path():
    with pytest.raises(ValueError, match="IDM"):
        TinyArchitecture({"variant": "idm", "video_prediction_type": "x"})


@pytest.mark.parametrize("video_mode,action_mode", [("v", "v"), ("x", "v"), ("v", "x"), ("x", "x")])
@pytest.mark.parametrize("action_scale", [1.0, 2.0])
def test_hydra_checkpoint_config_roundtrip(tmp_path, video_mode, action_mode, action_scale):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "model/video_backbone=wan22_style_1_3b", "model/video_backbone/encoder=vjepa21",
            f"model.architecture.video_prediction_type={video_mode}",
            f"model.architecture.action_prediction_type={action_mode}",
            f"model.architecture.action_scale={action_scale}",
        ])
    save_config(str(tmp_path), cfg)
    loaded = OmegaConf.load(tmp_path / "config.yaml")
    params = resolve_architecture_config(loaded.model).params
    arch = TinyArchitecture(params)
    assert (arch.video_prediction_type, arch.action_prediction_type) == (video_mode, action_mode)
    assert arch.action_scale == action_scale
