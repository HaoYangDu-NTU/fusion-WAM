import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.experimental.latent_space.wan22_stride2_encoder import (
    WAN22_STRIDE2_PATCHES_PER_STEP,
    WAN22_STRIDE2_LATENT_FRAMES,
    WAN22_STRIDE2_LOGICAL_TEMPORAL_IDS,
    _FASTWAM_CHECKPOINT,
    _Wan22VAE38Stride2EncoderCore,
    Wan22Stride2T5LatentEncoder,
    build_wan22_stride2_policy_mask,
    build_wan22_stride2_predictor_token_ids,
    make_wan22_stride2_video_offsets,
)


class FakeWanStride2Core(nn.Module):
    """Emit identifiable native temporal slots and record VAE inputs."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, video):
        self.calls.append(video.detach().clone())
        frames = (video.shape[2] + 3) // 4
        flat_batch = (
            (video[:, 0, 0, 0, 0] + 1.0)
            .mul(100)
            .round()
            .view(-1, 1, 1, 1, 1)
        )
        temporal = torch.arange(frames, device=video.device).view(1, 1, -1, 1, 1)
        rows = torch.arange(14, device=video.device).view(1, 1, 1, -1, 1)
        columns = torch.arange(14, device=video.device).view(1, 1, 1, 1, -1)
        values = flat_batch * 1_000_000 + temporal * 10_000 + rows * 100 + columns
        return values.expand(-1, 48, -1, -1, -1).float()


class _TinyCausalConv3d(nn.Conv3d):
    pass


class _TinyEncoder3d38(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, feat_cache=None, feat_idx=None):
        pooled = F.adaptive_avg_pool3d(x.mean(dim=1, keepdim=True), (1, 14, 14))
        return pooled.expand(-1, 96, -1, -1, -1), feat_cache, feat_idx


class _TinyWanImplementation:
    CausalConv3d = _TinyCausalConv3d
    Encoder3d_38 = _TinyEncoder3d38

    @staticmethod
    def patchify(x, patch_size):
        return x


def make_clips(frames=17, *, num_views=None):
    time = torch.arange(frames).float().mul(0.1).sub(1.0)
    if num_views is None:
        return time.view(1, 1, frames, 1, 1).expand(1, 3, frames, 224, 224).clone()
    views = torch.arange(num_views).float().mul(0.01)
    return (
        time.view(1, 1, 1, frames, 1, 1)
        + views.view(1, -1, 1, 1, 1, 1)
    ).expand(1, num_views, 3, frames, 224, 224).clone()


def expected_flat_tokens(flat_batch_id, frames=5):
    flat_batch = torch.as_tensor([flat_batch_id]).view(1, 1, 1, 1)
    temporal = torch.arange(frames).view(-1, 1, 1, 1)
    rows = torch.arange(14).view(1, 1, -1, 1)
    columns = torch.arange(14).view(1, 1, 1, -1)
    return (
        (flat_batch * 1_000_000 + temporal * 10_000 + rows * 100 + columns)
        .reshape(-1)
        .float()
    )


def test_offsets_and_dense_mask_use_five_native_time_slots():
    assert make_wan22_stride2_video_offsets() == tuple(range(0, 33, 2))
    assert WAN22_STRIDE2_LOGICAL_TEMPORAL_IDS == (0, 1, 2, 3, 4)
    context_ids, future_ids = build_wan22_stride2_predictor_token_ids(num_views=2)
    assert context_ids.numel() == 2 * WAN22_STRIDE2_PATCHES_PER_STEP
    assert future_ids.numel() == 4 * 2 * WAN22_STRIDE2_PATCHES_PER_STEP
    mask = build_wan22_stride2_policy_mask(image_size=(224, 224), num_views=2)
    torch.testing.assert_close(mask.ctx_idx, context_ids)
    torch.testing.assert_close(mask.tgt_idx, future_ids)


def test_stride2_core_accepts_seventeen_frames_and_emits_five_latents():
    core = _Wan22VAE38Stride2EncoderCore(
        _TinyWanImplementation,
        checkpoint_format=_FASTWAM_CHECKPOINT,
    )
    output = core(torch.zeros(2, 3, 17, 32, 32))
    assert output.shape == (2, 48, 5, 14, 14)


def test_target_uses_last_seventeen_rgb_frames_and_returns_five_latents():
    core = FakeWanStride2Core()
    encoder = Wan22Stride2T5LatentEncoder(core)
    clips = make_clips(frames=19)

    output = encoder(clips)

    assert output.shape == (1, WAN22_STRIDE2_LATENT_FRAMES * 14 * 14, 48)
    assert [tuple(call.shape) for call in core.calls] == [(1, 3, 17, 224, 224)]
    # The two leading history frames are dropped; frame 2 is physical t0.
    torch.testing.assert_close(
        core.calls[0][0, 0, :, 0, 0],
        torch.arange(2, 19).float().mul(0.1).sub(1.0),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        output[0, :, 0], expected_flat_tokens(20, frames=5), rtol=0, atol=0
    )


def test_context_mask_encodes_only_current_rgb_frame():
    core = FakeWanStride2Core()
    encoder = Wan22Stride2T5LatentEncoder(core)
    context_ids, _ = build_wan22_stride2_predictor_token_ids(num_views=1)

    output = encoder(make_clips(frames=19), context_ids.unsqueeze(0))

    assert output.shape == (1, WAN22_STRIDE2_PATCHES_PER_STEP, 48)
    assert [tuple(call.shape) for call in core.calls] == [(1, 3, 1, 224, 224)]
    torch.testing.assert_close(
        core.calls[0][0, 0, 0, 0, 0], torch.tensor(-0.8), rtol=0, atol=1e-7
    )


def test_two_view_target_keeps_time_then_view_token_order():
    core = FakeWanStride2Core()
    encoder = Wan22Stride2T5LatentEncoder(core)
    output = encoder(make_clips(frames=17, num_views=2))

    assert output.shape == (1, 5 * 2 * 14 * 14, 48)
    # View 0 starts at -1.0 and view 1 at -0.99; both expose all five times.
    first_view = output.reshape(1, 5, 2, 14, 14, 48)[0, :, 0, :, :, 0].reshape(-1)
    second_view = output.reshape(1, 5, 2, 14, 14, 48)[0, :, 1, :, :, 0].reshape(-1)
    torch.testing.assert_close(first_view, expected_flat_tokens(0, frames=5), rtol=0, atol=0)
    torch.testing.assert_close(second_view, expected_flat_tokens(1, frames=5), rtol=0, atol=0)
