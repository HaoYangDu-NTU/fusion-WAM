"""Train the isolated DINOv2-native16 or Wan-stride2 geometry experiment."""

from __future__ import annotations

import argparse
import os
from functools import partial

from scripts.train_vjepa_policy import main, parse_args
from vjepa_policy.experimental.latent_space.geometry_builders import (
    DINOV2_NATIVE16,
    GEOMETRY_EXPERIMENTS,
    build_geometry_dataset_collator,
    build_geometry_encoder,
    build_geometry_policy_for_experiment,
    build_geometry_predictor,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
)


def parse_experiment_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--geometry-experiment",
        choices=GEOMETRY_EXPERIMENTS,
        default=os.environ.get("GEOMETRY_EXPERIMENT"),
    )
    parser.add_argument(
        "--latent-checkpoint",
        default=os.environ.get("LATENT_CHECKPOINT"),
    )
    parser.add_argument(
        "--image-microbatch-size",
        type=int,
        default=int(os.environ.get("IMAGE_MICROBATCH_SIZE", "32")),
    )
    parser.add_argument(
        "--video-microbatch-size",
        type=int,
        default=int(os.environ.get("VIDEO_MICROBATCH_SIZE", "4")),
    )
    parser.add_argument(
        "--fastwam-src-dir",
        default=os.environ.get("FASTWAM_SRC_DIR", str(DEFAULT_FASTWAM_SRC_DIR)),
    )
    parser.add_argument(
        "--action-activation-checkpointing-blocks",
        type=int,
        default=int(os.environ.get("ACTION_ACTIVATION_CHECKPOINTING_BLOCKS", "0")),
    )
    parser.add_argument(
        "--action-microbatch-size",
        type=int,
        default=(
            int(os.environ["ACTION_MICROBATCH_SIZE"])
            if os.environ.get("ACTION_MICROBATCH_SIZE")
            else None
        ),
    )
    experiment, remaining = parser.parse_known_args(argv)
    args = parse_args(remaining)
    if experiment.geometry_experiment is None:
        parser.error("--geometry-experiment is required")
    if experiment.latent_checkpoint is None:
        parser.error("--latent-checkpoint is required")
    for name, value in vars(experiment).items():
        setattr(args, name, value)
    # Keep the legacy latent field explicit in metadata/debugging code while
    # ensuring no old builder is accidentally selected.
    args.latent_encoder = (
        "dinov2" if args.geometry_experiment == DINOV2_NATIVE16 else "wan2_2"
    )
    return args


def main_geometry(args):
    experiment = args.geometry_experiment
    main(
        args=args,
        encoder_builder=build_geometry_encoder,
        predictor_builder=partial(
            build_geometry_predictor,
            experiment=experiment,
        ),
        policy_builder=partial(
            build_geometry_policy_for_experiment,
            experiment,
            action_activation_checkpointing_blocks=(
                args.action_activation_checkpointing_blocks
            ),
            action_microbatch_size=args.action_microbatch_size,
        ),
        dataset_collator_builder=partial(
            build_geometry_dataset_collator,
            experiment=experiment,
        ),
    )


if __name__ == "__main__":
    main_geometry(parse_experiment_args())
