"""CPU construction preflight for a latent-space ablation."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from vjepa_policy.experimental.latent_space.builders import (
    build_latent_encoder,
    build_latent_predictor,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
    build_wan22_policy_mask,
)
from vjepa_policy.data import CausalPatchMask
from vjepa_policy.models.vjepa_policy import build_vjepa_policy


def baseline_args(cli_args):
    return SimpleNamespace(
        encoder_family="vjepa2",
        model_name="vit_large",
        encoder_checkpoint_key="target_encoder",
        crop_size=224,
        patch_size=16,
        tubelet_size=2,
        num_frames=33,
        past_frames=4,
        video_frame_stride=4,
        context_tubelets=1,
        view_layout="independent",
        views=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        camera_key=None,
        dataset_dirs=[],
        pred_depth=24,
        pred_embed_dim=1024,
        pred_num_heads=16,
        action_hidden_size=512,
        action_num_layers=24,
        condition_num_heads=8,
        predictor_rope_frequency_pairing="corrected",
        encoder_interpolate_rope=True,
        latent_encoder=cli_args.latent_encoder,
        latent_checkpoint=cli_args.latent_checkpoint,
        image_microbatch_size=cli_args.image_microbatch_size,
        video_microbatch_size=cli_args.video_microbatch_size,
        fastwam_src_dir=cli_args.fastwam_src_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent-encoder",
        required=True,
        choices=("dinov2", "dinov3", "wan2_2", "internvideo3"),
    )
    parser.add_argument("--latent-checkpoint", required=True)
    parser.add_argument("--image-microbatch-size", type=int, default=32)
    parser.add_argument("--video-microbatch-size", type=int, default=4)
    parser.add_argument("--fastwam-src-dir", default=str(DEFAULT_FASTWAM_SRC_DIR))
    parser.add_argument("--forward-device", default=None)
    cli_args = parser.parse_args()
    args = baseline_args(cli_args)

    torch.manual_seed(7)
    encoder = build_latent_encoder(args)
    predictor = build_latent_predictor(
        latent_encoder=args.latent_encoder,
        img_size=(224, 224),
        patch_size=16,
        num_frames=10,
        tubelet_size=2,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        num_mask_tokens=10,
        lang_dim=4096,
        proprio_dim=8,
        activation_checkpointing_blocks=7,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=True,
        num_views=2,
    )
    policy = build_vjepa_policy(
        encoder,
        predictor,
        action_dim=7,
        action_chunk_size=32,
        context_len=2 * 14 * 14,
        proprio_dim=8,
        action_hidden_size=512,
        action_num_layers=24,
        condition_num_heads=8,
        action_num_inference_steps=10,
    )
    if args.latent_encoder == "wan2_2":
        mask = build_wan22_policy_mask(image_size=(224, 224), num_views=2)
        assert mask.n_ctx == 2 * 14 * 14
        assert mask.tgt_idx.numel() == 2 * 2 * 14 * 14
    else:
        mask = CausalPatchMask(
            image_size=(224, 224),
            patch_size=16,
            tubelet_size=2,
            video_frames=10,
            context_tubelets=1,
            num_views=2,
        )

    print(f"encoder={sum(p.numel() for p in policy.encoder.parameters()):,}")
    print(f"predictor={sum(p.numel() for p in policy.predictor.parameters()):,}")
    print(
        f"action_expert={sum(p.numel() for p in policy.action_expert.parameters()):,}"
    )
    print(
        "predictor_rope=corrected,endpoint_aligned "
        f"grid={predictor.grid_depth}x{predictor.grid_height}x{predictor.grid_width} "
        f"views={predictor.num_views}"
    )
    if cli_args.forward_device is not None:
        device = torch.device(cli_args.forward_device)
        policy = policy.to(device).train()
        batch_size = 1
        text_length = 128 if args.latent_encoder == "internvideo3" else 32
        batch = (
            torch.rand(batch_size, 2, 3, 10, 224, 224, device=device).mul(2).sub(1),
            torch.randn(batch_size, text_length, 4096, device=device),
            torch.ones(batch_size, text_length, dtype=torch.bool, device=device),
            [mask.ctx_idx.unsqueeze(0).to(device)],
            [mask.tgt_idx.unsqueeze(0).to(device)],
            torch.randn(batch_size, 32, 7, device=device),
            torch.zeros(batch_size, 32, dtype=torch.bool, device=device),
            torch.randn(batch_size, 8, device=device),
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite preflight loss: {metrics}")
        loss.backward()
        print(
            f"forward_loss={loss.item():.6f} "
            f"world={metrics['loss_world'].item():.6f} "
            f"action={metrics['loss_action'].item():.6f}"
        )
        if device.type == "cuda":
            print(
                f"cuda_peak_mib={torch.cuda.max_memory_allocated(device) / 1024**2:.0f}"
            )


if __name__ == "__main__":
    main()
