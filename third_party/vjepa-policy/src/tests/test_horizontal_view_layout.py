import argparse
from unittest import mock

import torch

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.datasets import combine_video_clips
from vjepa_policy.models.wm_predictor import build_world_model_predictor
from vjepa_policy.models.vision_transformer import VisionTransformer
from vjepa_policy.train_common import (
    add_common_args,
    build_dataset_collator,
    build_patch_valid,
    get_model_image_size,
)


def make_horizontal_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args(
        [
            "--dataset-dirs",
            "/tmp/dataset",
            "--text-cache-dir",
            "/tmp/text",
            "--checkpoint",
            "/tmp/vjepa2.pt",
            "--views",
            "observation.images.image",
            "observation.images.wrist_image",
            "--view-layout",
            "horizontal",
            "--crop-size",
            "224",
            "--patch-size",
            "16",
        ]
    )


def test_horizontal_views_keep_full_resolution_and_have_no_invalid_patches():
    args = make_horizontal_args()
    assert get_model_image_size(args) == (224, 448)
    assert build_patch_valid(args) is None

    fake_dataset = mock.Mock(action_dim=7, state_dim=8)
    with mock.patch(
        "vjepa_policy.train_common.LeRobotVideoDataset", return_value=fake_dataset
    ) as dataset_class:
        dataset, collator = build_dataset_collator(args, action_chunk_size=32)
    assert dataset is fake_dataset
    dataset_kwargs = dataset_class.call_args.kwargs
    assert dataset_kwargs["video_transform"].size == (224, 224)
    assert dataset_kwargs["view_combiner"] == "horizontal"
    assert collator.n_ctx == 14 * 28

    transform = dataset_kwargs["video_transform"]
    views = [transform(torch.zeros(10, 3, 64, 64)) for _ in range(2)]
    video = combine_video_clips(views, "horizontal")
    assert video.shape == (3, 10, 224, 448)


def test_horizontal_mask_and_rope_grid_are_14_by_28():
    image_size = (224, 448)
    mask = CausalPatchMask(
        image_size=image_size,
        patch_size=16,
        tubelet_size=2,
        video_frames=10,
        context_tubelets=1,
    )
    assert mask.n_ctx == 14 * 28
    assert mask.ctx_idx[-1].item() == 14 * 28 - 1
    assert mask.tgt_idx[0].item() == 14 * 28
    assert mask.tgt_idx[-1].item() == 5 * 14 * 28 - 1

    encoder = VisionTransformer(
        img_size=image_size,
        patch_size=16,
        num_frames=10,
        tubelet_size=2,
        embed_dim=48,
        depth=1,
        num_heads=3,
        use_rope=True,
        interpolate_rope=True,
    )
    assert (encoder.img_height, encoder.img_width) == image_size
    assert encoder.blocks[0].attn.grid_size == 14
    assert encoder.blocks[0].attn.interpolate_rope is True
    assert encoder.blocks[0].attn.corrected_frequency_pairing is False

    predictor = build_world_model_predictor(
        img_size=image_size,
        patch_size=16,
        num_frames=10,
        tubelet_size=2,
        embed_dim=48,
        predictor_embed_dim=48,
        depth=1,
        num_heads=3,
        num_mask_tokens=2,
        lang_dim=8,
        proprio_dim=5,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=True,
    )
    assert (predictor.grid_height, predictor.grid_width) == (14, 28)
    rope = predictor.predictor_blocks[0].self_attn
    assert rope.interpolate_rope is True
    assert rope.corrected_frequency_pairing is True
    assert rope.canonical_spatial_grid == (16, 16)

    ids = torch.tensor([0, 27, 13 * 28, 14 * 28 - 1])
    _, height, width = rope.separate_positions(ids, H_patches=14, W_patches=28)
    scaled_height, scaled_width = rope._scale_spatial_positions(
        height, width, H_patches=14, W_patches=28
    )
    assert torch.equal(height, torch.tensor([0, 0, 13, 13]))
    assert torch.equal(width, torch.tensor([0, 27, 0, 27]))
    assert torch.allclose(scaled_height, torch.tensor([0.0, 0.0, 15.0, 15.0]))
    assert torch.allclose(scaled_width, torch.tensor([0.0, 15.0, 0.0, 15.0]))
