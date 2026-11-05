"""Serve a DINOv2-native16 or Wan-stride2 latent-geometry policy."""

from __future__ import annotations

import dataclasses
import logging
import socket
from typing import Literal

import numpy as np
from openpi_client import websocket_policy_server
import torch
import tyro

from examples.libero.serve_policy import Args as BaseArgs
from vjepa_policy.experimental.latent_space.geometry_serving import (
    DINOV2_NATIVE16_SERVING_FAMILY,
    GeometryVJEPAPolicyServing,
    WAN22_STRIDE2_SERVING_FAMILY,
    geometry_experiment_from_encoder,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
)
from vjepa_policy.policy_serving.libero import PolicyServingConfig


@dataclasses.dataclass
class Args(BaseArgs):
    encoder_family: Literal[
        "dinov2",
        "wan2_2_stride2_t5",
    ] = DINOV2_NATIVE16_SERVING_FAMILY
    model_name: str = "dinov2_vitl_native16"
    encoder_checkpoint_key: str = "local_pretrained"
    image_microbatch_size: int = 32
    video_microbatch_size: int = 4
    fastwam_src_dir: str = str(DEFAULT_FASTWAM_SRC_DIR)


def _policy_config(args: Args) -> PolicyServingConfig:
    values = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(PolicyServingConfig)
        if hasattr(args, field.name)
    }
    return PolicyServingConfig(**values)


def main(args: Args) -> None:
    experiment = geometry_experiment_from_encoder(
        args.encoder_family,
        args.model_name,
    )
    if args.encoder_family == WAN22_STRIDE2_SERVING_FAMILY:
        expected_key = "encoder_quant_conv"
    else:
        expected_key = "local_pretrained"
    if args.encoder_checkpoint_key != expected_key:
        raise ValueError(
            f"{args.encoder_family} requires encoder_checkpoint_key={expected_key!r}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = GeometryVJEPAPolicyServing(
        experiment=experiment,
        ckpt_path=args.ckpt,
        pretrained_encoder=args.pretrained_encoder,
        text_cache_dir=args.text_cache_dir,
        config=_policy_config(args),
        device=args.device,
        image_microbatch_size=args.image_microbatch_size,
        video_microbatch_size=args.video_microbatch_size,
        fastwam_src_dir=args.fastwam_src_dir,
    )
    hostname = socket.gethostname()
    logging.info(
        "Starting geometry policy server (experiment=%s host=%s ip=%s port=%d)",
        experiment,
        hostname,
        socket.gethostbyname(hostname),
        args.port,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={"geometry_experiment": experiment},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
