"""Train a policy with only its frozen latent encoder replaced."""

from __future__ import annotations

import argparse
import os
from functools import partial

from scripts.train_vjepa_policy import main, parse_args
from vjepa_policy.experimental.latent_space.builders import (
    build_latent_dataset_collator,
    build_latent_encoder,
    build_latent_predictor,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
)


def parse_experiment_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--latent-encoder",
        choices=("dinov2", "dinov3", "wan2_2", "internvideo3"),
        default=os.environ.get("LATENT_ENCODER"),
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
    experiment, remaining = parser.parse_known_args(argv)
    args = parse_args(remaining)
    if experiment.latent_encoder is None or experiment.latent_checkpoint is None:
        parser.error("--latent-encoder and --latent-checkpoint are required")
    for name, value in vars(experiment).items():
        setattr(args, name, value)
    return args


if __name__ == "__main__":
    args = parse_experiment_args()
    main(
        args=args,
        encoder_builder=build_latent_encoder,
        predictor_builder=partial(
            build_latent_predictor,
            latent_encoder=args.latent_encoder,
        ),
        dataset_collator_builder=build_latent_dataset_collator,
    )
