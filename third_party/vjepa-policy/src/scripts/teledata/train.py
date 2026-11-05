"""Train the canonical policy on a local LeRobot v3 TeleData dataset."""

import json
import os

import torch

from scripts.train_vjepa_policy import main, parse_args
from vjepa_policy.datasets.lerobot_video_v3 import LeRobotVideoV3Dataset
from vjepa_policy.datasets.lerobot_video_teledata import TeleDataEpisodeDataset
from vjepa_policy.datasets.transforms import VideoClipTransform
from vjepa_policy.train_common import (
    VJEPABatchCollator,
    get_camera_keys,
    get_model_image_size,
    get_model_video_frames,
)


def parse_teledata_args(argv=None):
    args = parse_args(argv)
    args.image_resize_mode = "letterbox"
    return args


def build_v3_dataset_collator(args, action_chunk_size):
    if len(args.dataset_dirs) != 1:
        raise ValueError("The TeleData v3 entry point accepts exactly one dataset root")
    if args.view_layout != "independent":
        raise ValueError("The four TeleData cameras must use independent view layout")
    if args.instruction_field != "task":
        raise ValueError("LeRobot v3 TeleData provides task instructions, not remarks")
    expected_action_chunk_size = args.num_frames - 1
    if action_chunk_size != expected_action_chunk_size:
        raise ValueError(
            f"action_chunk_size must equal num_frames - 1 ({expected_action_chunk_size})"
        )

    camera_keys = get_camera_keys(args)
    image_size = get_model_image_size(args)
    dataset_kwargs = dict(
        video_keys=camera_keys,
        num_frames=args.num_frames,
        past_frames=args.past_frames,
        video_stride=args.video_frame_stride,
        action_indices=args.action_indices,
        state_indices=args.state_indices,
        relative_action_indices=args.relative_action_indices,
        video_transform=VideoClipTransform(
            (args.crop_size, args.crop_size), resize_mode=args.image_resize_mode
        ),
        view_combiner="independent",
        text_cache_dir=args.text_cache_dir,
        context_len=args.context_len,
        normalization_stats_path=args.dataset_stats,
        action_normalization=args.action_normalization,
        state_normalization=args.state_normalization,
        clip_normalized=args.clip_normalized,
        video_backend=args.video_backend or "torchcodec",
        require_full_future_window=args.require_full_future_window,
    )
    info_path = os.path.join(args.dataset_dirs[0], "meta", "info.json")
    with open(info_path, encoding="utf-8") as handle:
        codebase_version = json.load(handle).get("codebase_version")
    if codebase_version == "v2.1":
        dataset = TeleDataEpisodeDataset(
            dataset_dir=args.dataset_dirs[0], **dataset_kwargs
        )
    else:
        dataset = LeRobotVideoV3Dataset(
            dataset_dir=args.dataset_dirs[0], **dataset_kwargs
        )
    if dataset.action_dim != args.action_dim:
        raise ValueError(
            f"Configured action_dim={args.action_dim}, dataset has {dataset.action_dim}"
        )
    if dataset.state_dim != args.proprio_dim:
        raise ValueError(
            f"Configured proprio_dim={args.proprio_dim}, dataset has {dataset.state_dim}"
        )
    if args.max_state_dim and args.proprio_dim > args.max_state_dim:
        raise ValueError(
            f"proprio_dim={args.proprio_dim} exceeds max_state_dim={args.max_state_dim}"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        dataset.normalizer.save(os.path.join(args.output_dir, "dataset_stats.json"))

    collator = VJEPABatchCollator(
        image_size=image_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        video_frames=get_model_video_frames(args),
        context_tubelets=args.context_tubelets,
        include_action=True,
        num_views=len(camera_keys),
    )
    return dataset, collator


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main(
        dataset_collator_builder=build_v3_dataset_collator,
        args=parse_teledata_args(),
    )
