import argparse
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.datasets import combine_video_clips, discover_video_keys
from vjepa_policy.policy_serving.libero import PolicyServingConfig
from vjepa_policy.train_common import (
    VJEPABatchCollator,
    add_common_args,
    build_dataset_collator,
    build_patch_valid,
    get_model_image_size,
    get_camera_keys,
)


def make_independent_args():
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
            "independent",
            "--crop-size",
            "224",
            "--patch-size",
            "16",
        ]
    )


def test_independent_views_preserve_view_dimension():
    views = [torch.full((10, 3, 224, 224), value) for value in (1.0, 2.0)]
    video = combine_video_clips(views, "independent")

    assert video.shape == (2, 3, 10, 224, 224)
    assert torch.equal(video[0], views[0].permute(1, 0, 2, 3))
    assert torch.equal(video[1], views[1].permute(1, 0, 2, 3))


@pytest.mark.parametrize("num_views", [1, 3])
def test_independent_views_support_dynamic_view_count(num_views):
    views = [
        torch.full((4, 3, 32, 32), float(index))
        for index in range(num_views)
    ]

    video = combine_video_clips(views)

    assert video.shape == (num_views, 3, 4, 32, 32)
    for index, view in enumerate(views):
        assert torch.equal(video[index], view.permute(1, 0, 2, 3))


def write_info(root, features):
    metadata_dir = root / "meta"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "info.json").write_text(
        json.dumps({"features": features}), encoding="utf-8"
    )


def test_video_key_discovery_is_sorted_and_shared_across_roots(tmp_path):
    features = {
        "observation.images.wrist": {"dtype": "video"},
        "action": {"dtype": "float32"},
        "observation.images.main": {"dtype": "image"},
    }
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        write_info(root, features)

    keys = discover_video_keys(roots)

    assert keys == ["observation.images.main", "observation.images.wrist"]


def test_video_key_discovery_rejects_inconsistent_roots(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    write_info(roots[0], {"observation.images.main": {"dtype": "video"}})
    write_info(roots[1], {"observation.images.wrist": {"dtype": "video"}})

    with pytest.raises(ValueError, match="same ordered camera features"):
        discover_video_keys(roots)


def test_default_layout_discovers_all_dataset_views():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--dataset-dirs",
            "/tmp/dataset",
            "--text-cache-dir",
            "/tmp/text",
            "--checkpoint",
            "/tmp/vjepa2.pt",
        ]
    )
    discovered = ["observation.images.main", "observation.images.wrist"]
    with mock.patch(
        "vjepa_policy.train_common.discover_video_keys", return_value=discovered
    ):
        keys = get_camera_keys(args)

    assert args.view_layout == "independent"
    assert args.crop_size == 224
    assert keys == discovered
    assert get_model_image_size(args) == (224, 224)


def test_libero_deployment_defaults_use_independent_224_views():
    config = PolicyServingConfig()
    assert config.crop_size == 224
    assert config.view_layout == "independent"
    assert config.encoder_interpolate_rope is True

    script = (
        Path(__file__).parents[1] / "examples" / "libero" / "evaluate_policy.sh"
    ).read_text(encoding="utf-8")
    assert "POLICY_CROP_SIZE=${POLICY_CROP_SIZE:-224}" in script
    assert "POLICY_VIEW_LAYOUT=${POLICY_VIEW_LAYOUT:-independent}" in script
    assert (
        "POLICY_ENCODER_INTERPOLATE_ROPE=${POLICY_ENCODER_INTERPOLATE_ROPE:-1}"
        in script
    )


def test_independent_layout_uses_single_view_model_grid_and_two_view_mask():
    args = make_independent_args()
    assert get_model_image_size(args) == (224, 224)
    assert build_patch_valid(args) is None

    fake_dataset = mock.Mock(action_dim=7, state_dim=8)
    with mock.patch(
        "vjepa_policy.train_common.LeRobotVideoDataset", return_value=fake_dataset
    ) as dataset_class:
        dataset, collator = build_dataset_collator(args, action_chunk_size=32)

    assert dataset is fake_dataset
    dataset_kwargs = dataset_class.call_args.kwargs
    assert dataset_kwargs["video_transform"].size == (224, 224)
    assert dataset_kwargs["view_combiner"] == "independent"
    assert dataset_kwargs["require_full_future_window"] is False
    assert collator.num_views == 2
    assert collator.n_ctx == 2 * 14 * 14


def test_full_future_window_flag_is_forwarded_to_dataset():
    args = make_independent_args()
    args.require_full_future_window = True
    fake_dataset = mock.Mock(action_dim=7, state_dim=8)
    with mock.patch(
        "vjepa_policy.train_common.LeRobotVideoDataset", return_value=fake_dataset
    ) as dataset_class:
        build_dataset_collator(args, action_chunk_size=32)

    assert dataset_class.call_args.kwargs["require_full_future_window"] is True


def test_independent_mask_and_collator_are_temporal_then_view_major():
    mask = CausalPatchMask(
        image_size=(224, 224),
        patch_size=16,
        tubelet_size=2,
        video_frames=10,
        context_tubelets=1,
        num_views=2,
    )
    patches_per_view = 14 * 14
    assert torch.equal(mask.ctx_idx[:patches_per_view], torch.arange(patches_per_view))
    assert torch.equal(
        mask.ctx_idx[patches_per_view:],
        torch.arange(patches_per_view, 2 * patches_per_view),
    )
    assert mask.tgt_idx[0].item() == 2 * patches_per_view
    assert mask.tgt_idx[-1].item() == 5 * 2 * patches_per_view - 1

    collator = VJEPABatchCollator(
        image_size=(224, 224),
        patch_size=16,
        tubelet_size=2,
        video_frames=10,
        num_views=2,
    )
    sample = {
        "video": torch.zeros(2, 3, 10, 224, 224),
        "context": torch.zeros(4, 8),
        "context_mask": torch.ones(4, dtype=torch.bool),
    }
    clips, _, _, masks_enc, masks_pred = collator([sample, sample])
    assert clips.shape == (2, 2, 3, 10, 224, 224)
    assert masks_enc[0].shape == (2, 2 * patches_per_view)
    assert masks_pred[0].shape == (2, 4 * 2 * patches_per_view)


def test_three_view_mask_is_temporal_then_view_major():
    mask = CausalPatchMask(
        image_size=(32, 32),
        patch_size=16,
        tubelet_size=2,
        video_frames=4,
        context_tubelets=1,
        num_views=3,
    )

    assert torch.equal(mask.ctx_idx, torch.arange(12))
    assert torch.equal(mask.tgt_idx, torch.arange(12, 24))
