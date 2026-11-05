"""Recorded DINOv3 ViT-L entrypoint with the len-128 instruction cache.

The shared latent-ablation validator requires the canonical encoder-RoPE
interpolation setting.  The frozen DINO adapter does not use encoder RoPE, so
this setting has no effect on its latent tokens; all other latent-ablation
builders remain unchanged.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import torch

from scripts.train_vjepa_policy import main, parse_args
from vjepa_policy.experimental.latent_space.builders import (
    build_latent_dataset_collator,
    build_latent_predictor,
    _consume_baseline_encoder_initialization,
)
from vjepa_policy.experimental.latent_space.dino_encoder import (
    load_dino_latent_encoder,
)


def parse_experiment_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--latent-encoder",
        choices=("dinov3",),
        default=os.environ.get("LATENT_ENCODER", "dinov3"),
    )
    parser.add_argument(
        "--latent-checkpoint",
        default=os.environ.get("LATENT_CHECKPOINT"),
    )
    parser.add_argument(
        "--image-microbatch-size",
        type=int,
        default=int(os.environ.get("IMAGE_MICROBATCH_SIZE", "320")),
    )
    experiment, remaining = parser.parse_known_args(argv)
    args = parse_args(remaining)
    if experiment.latent_checkpoint is None:
        parser.error("--latent-checkpoint or LATENT_CHECKPOINT is required")
    for name, value in vars(experiment).items():
        setattr(args, name, value)
    return args


def build_recorded_dinov3_encoder(args):
    # Match the canonical baseline RNG advancement before loading DINO.
    _consume_baseline_encoder_initialization(args)
    with torch.random.fork_rng(devices=[]):
        return load_dino_latent_encoder(
            "dinov3",
            args.latent_checkpoint,
            image_microbatch_size=args.image_microbatch_size,
        )


if __name__ == "__main__":
    args = parse_experiment_args()
    main(
        args=args,
        encoder_builder=build_recorded_dinov3_encoder,
        predictor_builder=partial(build_latent_predictor, latent_encoder="dinov3"),
        dataset_collator_builder=build_latent_dataset_collator,
    )
