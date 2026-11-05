"""Serve a policy trained with a frozen alternate latent encoder."""

from __future__ import annotations

import dataclasses
import logging
from functools import partial
from typing import Literal

import tyro

from examples.libero.serve_policy import Args as BaseArgs
from examples.libero.serve_policy import main as serve_main
from vjepa_policy.experimental.latent_space.dino_encoder import (
    load_dino_latent_encoder,
)
from vjepa_policy.experimental.latent_space.internvideo3_encoder import (
    load_internvideo3_latent_encoder,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
    load_wan22_latent_encoder,
)


@dataclasses.dataclass
class Args(BaseArgs):
    encoder_family: Literal["dinov2", "dinov3", "wan2_2", "internvideo3"] = (
        "dinov2"
    )
    model_name: str = "dinov2_vitl"
    encoder_checkpoint_key: str = "local_pretrained"
    image_microbatch_size: int = 32
    video_microbatch_size: int = 4
    fastwam_src_dir: str = str(DEFAULT_FASTWAM_SRC_DIR)


def build_serving_encoder(
    config,
    pretrained_encoder,
    *,
    image_microbatch_size,
    video_microbatch_size,
    fastwam_src_dir,
):
    if config.encoder_family in ("dinov2", "dinov3"):
        return load_dino_latent_encoder(
            config.encoder_family,
            pretrained_encoder,
            image_microbatch_size=image_microbatch_size,
        )
    if config.encoder_family == "wan2_2":
        return load_wan22_latent_encoder(
            pretrained_encoder,
            fastwam_src_dir=fastwam_src_dir,
            video_microbatch_size=video_microbatch_size,
        )
    if config.encoder_family == "internvideo3":
        return load_internvideo3_latent_encoder(
            pretrained_encoder,
            video_microbatch_size=video_microbatch_size,
        )
    raise ValueError(f"unsupported latent encoder {config.encoder_family!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    serve_main(
        args,
        encoder_builder=partial(
            build_serving_encoder,
            image_microbatch_size=args.image_microbatch_size,
            video_microbatch_size=args.video_microbatch_size,
            fastwam_src_dir=args.fastwam_src_dir,
        ),
    )
