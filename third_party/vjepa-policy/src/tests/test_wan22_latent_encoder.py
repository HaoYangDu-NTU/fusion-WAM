from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vjepa_policy.experimental.latent_space.wan22_encoder import (
    WAN22_PATCHES_PER_STEP,
    WAN22_LOGICAL_TEMPORAL_IDS,
    Wan22LatentEncoder,
    _FASTWAM_CHECKPOINT,
    _Wan22VAE38EncoderCore,
    _load_fastwam_vae_module,
    build_wan22_policy_mask,
    build_wan22_predictor_token_ids,
    load_wan22_latent_encoder,
)
from vjepa_policy.models.vjepa_policy import encode_video_views


class FakeWanCore(nn.Module):
    def __init__(self, output_shape=(48, None, 14, 14)):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.output_shape = output_shape
        self.calls = []

    def forward(self, video):
        self.calls.append(video.detach().clone())
        channels, frames, height, width = self.output_shape
        if frames is None:
            frames = (video.shape[2] + 3) // 4
        flat_batch = (video[:, 0, 0, 0, 0] + 0.9).mul(100).round().view(-1, 1, 1, 1, 1)
        temporal = torch.arange(frames, device=video.device).view(1, 1, -1, 1, 1)
        rows = torch.arange(height, device=video.device).view(1, 1, 1, -1, 1)
        columns = torch.arange(width, device=video.device).view(1, 1, 1, 1, -1)
        values = flat_batch * 1_000_000 + temporal * 10_000 + rows * 100 + columns
        return values.expand(-1, channels, -1, -1, -1).float()


def make_clips(*, num_views=None, requires_grad=False):
    time = torch.arange(10).float().mul(0.1).sub(1.0)
    if num_views is None:
        clips = time.view(1, 1, 10, 1, 1).expand(1, 3, 10, 224, 224)
    else:
        views = torch.arange(num_views).float().mul(0.01)
        clips = (time.view(1, 1, 1, 10, 1, 1) + views.view(1, -1, 1, 1, 1, 1)).expand(
            1, num_views, 3, 10, 224, 224
        )
    return clips.clone().requires_grad_(requires_grad)


def expected_flat_tokens(flat_batch_ids):
    flat_batch = torch.as_tensor(flat_batch_ids).view(1, -1, 1, 1)
    temporal = torch.arange(3).view(-1, 1, 1, 1)
    rows = torch.arange(14).view(1, 1, -1, 1)
    columns = torch.arange(14).view(1, 1, 1, -1)
    return (
        (flat_batch * 1_000_000 + temporal * 10_000 + rows * 100 + columns)
        .reshape(-1)
        .float()
    )


def test_two_view_adapter_selects_frames_without_renormalizing_and_orders_tokens():
    core = FakeWanCore()
    encoder = Wan22LatentEncoder(core)
    clips = make_clips(num_views=2, requires_grad=True)

    output = encoder(clips)

    assert output.shape == (1, 5 * 2 * 14 * 14, 48)
    assert core.calls[0].shape == (2, 3, 9, 224, 224)
    expected_pixels = torch.stack(
        (
            torch.arange(1, 10).float().mul(0.1).sub(1.0),
            torch.arange(1, 10).float().mul(0.1).sub(0.99),
        )
    )
    torch.testing.assert_close(
        core.calls[0][:, 0, :, 0, 0], expected_pixels, rtol=0, atol=1e-7
    )
    context_ids, future_ids = build_wan22_predictor_token_ids(num_views=2)
    valid_ids = torch.cat((context_ids, future_ids))
    torch.testing.assert_close(
        output[0, valid_ids, 0], expected_flat_tokens([0, 1]), rtol=0, atol=0
    )
    invalid = output.reshape(1, 5, 2, 14, 14, 48)[:, [1, 3]]
    assert torch.count_nonzero(invalid) == 0

    assert core.training is False
    assert all(not parameter.requires_grad for parameter in core.parameters())
    encoder.train()
    assert core.training is False
    assert output.requires_grad is False
    assert clips.grad is None


def test_single_view_adapter_orders_tokens_as_time_height_width():
    core = FakeWanCore()
    encoder = Wan22LatentEncoder(core)

    output = encoder(make_clips())

    assert output.shape == (1, 5 * 14 * 14, 48)
    context_ids, future_ids = build_wan22_predictor_token_ids(num_views=1)
    torch.testing.assert_close(
        output[0, torch.cat((context_ids, future_ids)), 0],
        expected_flat_tokens([0]),
        rtol=0,
        atol=0,
    )


def test_masked_context_encodes_only_current_frame_and_supports_microbatching():
    core = FakeWanCore()
    encoder = Wan22LatentEncoder(core, video_microbatch_size=1)
    clips = make_clips(num_views=2)
    context_ids, _ = build_wan22_predictor_token_ids(num_views=2)

    output = encode_video_views(encoder, clips, context_ids.unsqueeze(0))

    assert output.shape == (1, 2 * WAN22_PATCHES_PER_STEP, 48)
    assert [call.shape for call in core.calls] == [
        (1, 3, 1, 224, 224),
        (1, 3, 1, 224, 224),
    ]
    torch.testing.assert_close(
        output[0, :, 0], expected_flat_tokens([0, 1])[: 2 * 14 * 14]
    )


def test_sparse_policy_mask_keeps_baseline_rope_lattice():
    mask = build_wan22_policy_mask(image_size=(224, 224), num_views=2)
    context_ids, future_ids = build_wan22_predictor_token_ids(num_views=2)

    torch.testing.assert_close(mask.ctx_idx, context_ids)
    torch.testing.assert_close(mask.tgt_idx, future_ids)
    assert mask.n_ctx == 2 * 14 * 14

    with pytest.raises(ValueError, match="exactly 2 views"):
        build_wan22_policy_mask(image_size=(224, 224), num_views=1)


def test_adapter_rejects_masks_for_missing_logical_times():
    encoder = Wan22LatentEncoder(FakeWanCore())
    unavailable_t1 = torch.arange(
        WAN22_PATCHES_PER_STEP,
        2 * WAN22_PATCHES_PER_STEP,
    ).unsqueeze(0)

    with pytest.raises(ValueError, match="no latents"):
        encoder(make_clips(), unavailable_t1)


def test_predictor_ids_keep_the_original_five_step_temporal_lattice():
    context_ids, future_ids = build_wan22_predictor_token_ids(num_views=2)

    assert WAN22_LOGICAL_TEMPORAL_IDS == (0, 2, 4)
    torch.testing.assert_close(context_ids, torch.arange(0, 392), rtol=0, atol=0)
    torch.testing.assert_close(
        future_ids[:392], torch.arange(784, 1176), rtol=0, atol=0
    )
    torch.testing.assert_close(
        future_ids[392:], torch.arange(1568, 1960), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "output_shape",
    [
        (47, 3, 14, 14),
        (48, 2, 14, 14),
        (48, 3, 13, 14),
    ],
)
def test_adapter_rejects_wrong_vae_geometry(output_shape):
    encoder = Wan22LatentEncoder(FakeWanCore(output_shape))

    with pytest.raises(ValueError, match="must return"):
        encoder(make_clips())


_TINY_FASTWAM_MODULE = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Conv3d):
    pass


class Encoder3d_38(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, feat_cache=None, feat_idx=None):
        x = F.adaptive_avg_pool3d(x.mean(dim=1, keepdim=True), (1, 14, 14))
        return x.expand(-1, 96, -1, -1, -1) * self.scale, feat_cache, feat_idx


class Decoder3d_38(nn.Module):
    def __init__(self, **kwargs):
        raise AssertionError("the decoder must not be constructed")


def patchify(x, patch_size):
    return x
"""


def make_tiny_fastwam_source(tmp_path: Path) -> Path:
    source_dir = tmp_path / "fastwam_src"
    module_path = source_dir / "fastwam" / "models" / "wan22" / "wan_video_vae.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(_TINY_FASTWAM_MODULE)
    return source_dir


def test_loader_reads_encoder_only_safetensors_without_constructing_decoder(tmp_path):
    source_dir = make_tiny_fastwam_source(tmp_path)
    implementation = _load_fastwam_vae_module(source_dir)
    reference_core = _Wan22VAE38EncoderCore(
        implementation,
        checkpoint_format=_FASTWAM_CHECKPOINT,
    )
    state_dict = {
        f"model.{key}": value.detach().contiguous()
        for key, value in reference_core.state_dict().items()
    }
    state_dict["model.decoder.weight"] = torch.ones(1)
    checkpoint = tmp_path / "wan22_vae.safetensors"
    save_file(state_dict, checkpoint)

    encoder = load_wan22_latent_encoder(checkpoint, fastwam_src_dir=source_dir)

    assert isinstance(encoder, Wan22LatentEncoder)
    assert not hasattr(encoder.encoder_core, "decoder")
    assert all(not parameter.is_meta for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    output = encoder(torch.zeros(1, 3, 10, 224, 224))
    assert output.shape == (1, 5 * 14 * 14, 48)


def test_loader_rejects_non_safetensors_and_missing_paths(tmp_path):
    with pytest.raises(ValueError, match="explicit .safetensors"):
        load_wan22_latent_encoder(tmp_path / "wan22_vae.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_wan22_latent_encoder(tmp_path / "missing.safetensors")
