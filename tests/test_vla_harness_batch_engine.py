"""Compare batched denoising with the real native generate loop using a CPU model."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
from openwam.deploy.denoise_schedule import make_schedule
from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from tests.test_vjepa_patch_grid import encoder_source


class Encoder:
    history_num_tubelets = 0

    def preprocess_video_batch(self, videos):
        assert all(len(v) == max(1, 2 * self.history_num_tubelets) for v in videos)
        values = [[float(np.mean(v[i:i+2])) for i in range(0, len(v), 2)] for v in videos]
        return torch.tensor(values)[:, None, :, None, None].expand(-1, 2, -1, 2, 2)

    def batch_encode(self, pixels):
        return pixels / 10


class VideoBackbone:
    shift_video = 12
    video_encoder = Encoder()

    def encode_video(self, pixels):
        return self.video_encoder.batch_encode(pixels)

    def preprocess_input_for_inference(self, prompt, clean_condition_images, **kwargs):
        noise = torch.randn(1, 2, 1 + max(1, self.video_encoder.history_num_tubelets), 2, 2, generator=torch.Generator().manual_seed(kwargs["seed"]))
        return {
            "context": torch.tensor([[float(prompt)]]),
            "seq_lens": torch.tensor([4]),
            "noise": noise,
            "latents": noise.clone(),
            "clean_condition_latents": None
            if clean_condition_images is None
            else self.video_encoder.batch_encode(self.video_encoder.preprocess_video_batch([clean_condition_images])),
            "fuse_vae_embedding_in_latents": clean_condition_images is not None,
            "num_clean_condition_latents": 0,
        }


class CPUArchitecture:
    generate = BaseWAMArchitecture.generate
    device = torch.device("cpu")
    dtype = torch.float32
    action_dim = 10
    uses_proprioception = True
    binary_command_dims = ()

    def __init__(self, inactive):
        self.video_backbone = VideoBackbone()
        self.action_backbone = SimpleNamespace(shift_action=5.0)
        self.video_scheduler = FlowMatchScheduler("Wan")
        self.action_scheduler = ActionScheduler()
        self.normalizer = SimpleNamespace(unnormalize=lambda values: values * 0.4 + 0.1)
        self.inactive = torch.tensor([False] * 8 + [True] * 2) if inactive else None

    def eval(self):
        return self

    def _resolve_inactive_action_dims(self, mask, device):
        return self.inactive

    def normalize_deploy_proprio(self, value):
        return torch.as_tensor(value, dtype=torch.float32) * 0.5

    def forward(self, actions, action_time, *, latents, context, proprio, clean_condition_latents, timestep, **kwargs):
        # Detect conditioning drift, sample mixing and mismatched clean prefixes.
        torch.testing.assert_close(latents[:, :, :clean_condition_latents.shape[2]], clean_condition_latents, rtol=0, atol=0)
        phase = context[:, 0] * action_time / 1000 * 25
        video_pattern = torch.arange(latents[0].numel()).reshape(1, *latents.shape[1:])
        action_pattern = torch.arange(actions[0].numel()).reshape(1, *actions.shape[1:])
        vv = torch.cos(video_pattern + phase.reshape(-1, 1, 1, 1, 1))
        aa = torch.cos(action_pattern + phase.reshape(-1, 1, 1))
        # Context 0 stays stable (cache hits); context 1 changes (cache misses).
        aa = aa + proprio.reshape(-1, 1, 10) / 10
        if getattr(self, "video_prediction_type", "v") == "x":
            vv = latents - (timestep / 1000).reshape(-1, 1, 1, 1, 1) * vv
        if getattr(self, "action_prediction_type", "v") == "x":
            aa = actions - (action_time / 1000).reshape(-1, 1, 1) * aa
        return vv, aa


@pytest.mark.parametrize("video_mode,action_mode", [("v", "v"), ("x", "v"), ("v", "x"), ("x", "x")])
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
@pytest.mark.parametrize("history", [0, 1, 2, 3])
@pytest.mark.parametrize("action_scale", [1.0, 2.0])
def test_batch_matches_native_loop_and_individual_cache_decisions(
    cache_enabled, inactive, video_mode, action_mode, history, action_scale,
):
    arch = CPUArchitecture(inactive)
    arch.action_scale = action_scale
    arch.video_backbone.video_encoder = Encoder()
    arch.video_backbone.video_encoder.history_num_tubelets = int(history)
    arch.video_prediction_type = video_mode
    arch.action_prediction_type = action_mode
    cfg = SimpleNamespace(
        inference=SimpleNamespace(num_frames=5, video_num_frames=3, denoise_steps=10, height=4, width=4)
    )
    native = SimpleNamespace(
        architecture=arch, cfg=cfg, _prompt_embed_cache={}, _dit_cache=DiTVelocityCache() if cache_enabled else None
    )
    engine = BatchedOpenWAMEngine.__new__(BatchedOpenWAMEngine)
    engine.native, engine.architecture, engine.cfg = native, arch, cfg
    obs = [{"clean_condition_images": list(range(i, i + 2 * history)), "image": 2 + i, "prompt": str(i),
            "state": np.arange(10, dtype=np.float32) + i} for i in range(2)]
    expected, cache_stats = [], []
    for item in obs:
        cache = DiTVelocityCache() if cache_enabled else None
        schedule = make_schedule(
            "sync",
            video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler,
            num_steps=10,
            shift=5,
            shift_video=12,
        )
        result = arch.generate(
            schedule,
            item["prompt"],
            clean_condition_images=item["clean_condition_images"] if history else [item["image"]],
            num_frames=3,
            action_num_frames=5,
            height=4,
            width=4,
            seed=42,
            num_inference_steps=10,
            shift=5,
            dit_cache=cache,
            decode_video=False,
            proprio=arch.normalize_deploy_proprio(item["state"]),
        )
        expected.append(result["actions"])
        cache_stats.append(cache.stats if cache else None)
    actual = engine.generate_batch(obs)
    for reference, result in zip(expected, actual):
        np.testing.assert_array_equal(result, reference)
    assert engine.last_stats["cache"] == cache_stats
    if cache_enabled:
        assert cache_stats[0]["total_skips"] > cache_stats[1]["total_skips"]
        assert set(engine.last_stats["forward_batch_sizes"]) == {1, 2}
    reversed_result = engine.generate_batch(obs[::-1])
    for reference, result in zip(expected[::-1], reversed_result):
        np.testing.assert_array_equal(result, reference)
    if inactive:
        # The zero-padding noise path ends at zero before unnormalization.
        np.testing.assert_allclose(actual[0][:, -2:], 0.1)


@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("wan_grid", [False, True])
@pytest.mark.parametrize("action_scale", [1.0, 2.0])
def test_independent_views_real_joint_model_batch_matches_native(tmp_path, encoder_source, wan_grid, action_scale, monkeypatch, history):
    from PIL import Image

    from openwam.dataloader.transforms.multiview import assemble_two_view_layout
    from tests.test_separate_encode_view import _backbone
    from tests.test_vjepa_patch_grid import _joint_model

    torch.manual_seed(17)
    vb = _backbone(tmp_path, encoder_source, wan_grid, history=history)
    arch = _joint_model(vb)
    arch.action_scale = action_scale
    arch._init_proprio_context({"use_proprioception": True, "state_dim": 80}, text_dim=4096)
    arch.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    shift = arch.action_backbone.shift_action
    shift = 5.0 if shift is None else shift
    inf = SimpleNamespace(num_frames=33, video_num_frames=9, denoise_steps=10,
                          height=384, width=320, denoise_mode="sync")
    native = SimpleNamespace(
        architecture=arch, cfg=SimpleNamespace(inference=inf), _prompt_embed_cache={},
        _dit_cache=None, _cfg_scale=1.0, _decode_video=False,
    )
    engine = BatchedOpenWAMEngine(native)

    def canvas(head, wrist):
        return assemble_two_view_layout(
            Image.new("RGB", (224, 224), (head,) * 3),
            Image.new("RGB", (224, 224), (wrist,) * 3),
        )

    observations = [
        {"prompt": prompt, "image": canvas(30 + i * 10, 100 + i * 10),
         "clean_condition_images": [canvas(50 + j + i * 10, 160 + j + i * 10) for j in range(2 * history - 1)]
                                   + [canvas(30 + i * 10, 100 + i * 10)], "state": np.full(80, i, np.float32)}
        for i, prompt in enumerate(["open drawer", "close drawer"])
    ]
    expected = []
    for obs in observations:
        schedule = make_schedule("sync", video_scheduler=arch.video_scheduler,
                                 action_scheduler=arch.action_scheduler, num_steps=10,
                                 shift=shift, shift_video=vb.shift_video)
        expected.append(arch.generate(
            schedule, obs["prompt"], clean_condition_images=obs["clean_condition_images"],
            num_frames=9, action_num_frames=33, height=384, width=320, seed=42,
            num_inference_steps=10, shift=shift, decode_video=False,
            proprio=arch.normalize_deploy_proprio(obs["state"]),
        )["actions"])
    from openwam.model.video_backbone.wan_backbone import wan_conditioning
    original_noise = wan_conditioning.build_deploy_noise
    noise_calls = []
    def build_noise(**kwargs):
        noise_calls.append(kwargs)
        return original_noise(**kwargs)
    monkeypatch.setattr(wan_conditioning, "build_deploy_noise", build_noise)
    actual = engine.generate_batch(observations)
    assert len(noise_calls) == 1  # same seed/grid across requests, independent trajectories

    assert vb.video_encoder._m.inputs[-1].shape == (4, 3, 2 * history, 224, 224)
    assert engine.last_stats["forward_batch_sizes"] == [2] * 10
    for a, b in zip(actual, expected):
        assert a.shape == (32, 80) and np.isfinite(a).all()
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
