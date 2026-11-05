"""Serve a trained VJEPA-Policy checkpoint over the openpi websocket protocol."""

import dataclasses
import logging
import socket
from typing import Literal

import numpy as np
from openpi_client import websocket_policy_server
import torch
import tyro

from vjepa_policy.policy_serving import PolicyServingConfig, VJEPAPolicyServing


@dataclasses.dataclass
class Args:
    ckpt: str
    pretrained_encoder: str
    text_cache_dir: str
    dataset_stats: str | None = None
    host: str = "0.0.0.0"
    port: int = 10000
    seed: int = 7
    precision: Literal["bf16", "fp32"] = "bf16"
    device: str = "cuda"
    encoder_family: Literal["vjepa2", "vjepa2_1"] = "vjepa2"
    encoder_checkpoint_key: str = "encoder"
    model_name: str = "vit_giant_xformers"
    crop_size: int = 224
    patch_size: int = 16
    num_frames: int = 10
    tubelet_size: int = 2
    context_tubelets: int = 1
    pred_depth: int = 24
    pred_embed_dim: int = 1024
    pred_num_heads: int = 16
    num_mask_tokens: int = 10
    lang_dim: int = 4096
    t5_len: int = 32
    action_dim: int = 7
    proprio_dim: int = 8
    max_state_dim: int | None = None
    action_chunk_size: int = 32
    action_hidden_size: int = 512
    action_num_layers: int = 24
    action_num_inference_steps: int = 10
    condition_num_heads: int = 8
    corrected_predictor_rope: bool = True
    encoder_interpolate_rope: bool = True
    view_layout: Literal["quadrant", "independent"] = "independent"
    camera_keys: tuple[str, ...] = (
        "observation.images.image",
        "observation.images.wrist_image",
    )


def main(args, encoder_builder=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = VJEPAPolicyServing(
        ckpt_path=args.ckpt,
        pretrained_encoder=args.pretrained_encoder,
        text_cache_dir=args.text_cache_dir,
        config=PolicyServingConfig(
            precision=args.precision,
            encoder_family=args.encoder_family,
            encoder_checkpoint_key=args.encoder_checkpoint_key,
            model_name=args.model_name,
            crop_size=args.crop_size,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            context_tubelets=args.context_tubelets,
            pred_depth=args.pred_depth,
            pred_embed_dim=args.pred_embed_dim,
            pred_num_heads=args.pred_num_heads,
            num_mask_tokens=args.num_mask_tokens,
            lang_dim=args.lang_dim,
            t5_len=args.t5_len,
            action_dim=args.action_dim,
            proprio_dim=args.proprio_dim,
            max_state_dim=args.max_state_dim,
            action_chunk_size=args.action_chunk_size,
            action_hidden_size=args.action_hidden_size,
            action_num_layers=args.action_num_layers,
            action_num_inference_steps=args.action_num_inference_steps,
            condition_num_heads=args.condition_num_heads,
            corrected_predictor_rope=args.corrected_predictor_rope,
            encoder_interpolate_rope=args.encoder_interpolate_rope,
            view_layout=args.view_layout,
            camera_keys=args.camera_keys,
            dataset_stats=args.dataset_stats,
        ),
        device=args.device,
        encoder_builder=encoder_builder,
    )
    hostname = socket.gethostname()
    logging.info(
        "Starting policy server (host=%s, ip=%s, port=%d)",
        hostname,
        socket.gethostbyname(hostname),
        args.port,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
