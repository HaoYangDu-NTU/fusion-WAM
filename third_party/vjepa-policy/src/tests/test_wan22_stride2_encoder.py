import torch
import torch.nn as nn

from vjepa_policy.experimental.latent_space.geometry_policy import (
    encode_geometry_video_views,
)
from vjepa_policy.experimental.latent_space.wan22_stride2_encoder import (
    WAN22_STRIDE2_PATCHES_PER_STEP,
    Wan22Stride2T5LatentEncoder,
    build_wan22_stride2_policy_mask,
    make_wan22_stride2_video_offsets,
)


class FakeWanStride2Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, video):
        self.calls.append(video.detach().clone())
        latent_frames = 1 + (video.shape[2] - 1) // 4
        temporal = torch.arange(latent_frames, device=video.device).view(1, 1, -1, 1, 1)
        rows = torch.arange(14, device=video.device).view(1, 1, 1, -1, 1)
        cols = torch.arange(14, device=video.device).view(1, 1, 1, 1, -1)
        batch_id = (video[:, 0, 0, 0, 0] + 0.9).mul(100).round().view(-1, 1, 1, 1, 1)
        values = batch_id * 1_000_000 + temporal * 10_000 + rows * 100 + cols
        return values.expand(-1, 48, -1, -1, -1).float()


def make_clips(frames=17, *, views=2):
    time = torch.arange(frames).float().mul(0.1).sub(1.0)
    if views == 1:
        return time.view(1, 1, frames, 1, 1).expand(1, 3, frames, 224, 224)
    view = torch.arange(views).float().mul(0.01)
    return (
        time.view(1, 1, 1, frames, 1, 1)
        + view.view(1, views, 1, 1, 1, 1)
    ).expand(1, views, 3, frames, 224, 224)


def test_stride2_offsets_and_target_have_five_real_latents():
    assert make_wan22_stride2_video_offsets() == tuple(range(0, 33, 2))
    core = FakeWanStride2Core()
    encoder = Wan22Stride2T5LatentEncoder(core)

    output = encoder(make_clips())

    assert output.shape == (1, 5 * 2 * 14 * 14, 48)
    assert core.calls[0].shape == (2, 3, 17, 224, 224)
    assert torch.count_nonzero(output) > 0
    assert encoder._vjepa_policy_latent_frames == 5


def test_stride2_context_mask_encodes_only_current_rgb_frame():
    core = FakeWanStride2Core()
    encoder = Wan22Stride2T5LatentEncoder(core, video_microbatch_size=1)
    mask = build_wan22_stride2_policy_mask(image_size=(224, 224), num_views=2)

    output = encode_geometry_video_views(encoder, make_clips(), mask.ctx_idx.unsqueeze(0))

    assert output.shape == (1, 2 * WAN22_STRIDE2_PATCHES_PER_STEP, 48)
    assert [call.shape for call in core.calls] == [(1, 3, 1, 224, 224), (1, 3, 1, 224, 224)]


def test_stride2_supports_leading_history_without_changing_current_window():
    core = FakeWanStride2Core()
    encoder = Wan22Stride2T5LatentEncoder(core)
    clips = make_clips(frames=19, views=1)
    encoder(clips)

    assert core.calls[0].shape == (1, 3, 17, 224, 224)
    torch.testing.assert_close(core.calls[0][:, 0, 0, 0, 0], clips[:, 0, 2, 0, 0])


def test_stride2_policy_mask_is_t0_to_all_four_future_steps():
    mask = build_wan22_stride2_policy_mask(image_size=(224, 224), num_views=2)
    assert mask.n_ctx == 2 * 14 * 14
    assert mask.tgt_idx.numel() == 4 * 2 * 14 * 14
    assert mask.tgt_idx[0] == 2 * 14 * 14
    assert mask.tgt_idx[-1] == 5 * 2 * 14 * 14 - 1
