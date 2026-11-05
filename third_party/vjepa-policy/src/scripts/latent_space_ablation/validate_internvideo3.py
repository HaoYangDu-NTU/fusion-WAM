"""Compare the vectorized adapter path with the official visual forward."""

from __future__ import annotations

import argparse

import torch

from vjepa_policy.experimental.latent_space.internvideo3_encoder import (
    load_internvideo3_latent_encoder,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    encoder = load_internvideo3_latent_encoder(args.checkpoint).to(device)
    clips = torch.randn(1, 3, 10, 224, 224, device=device)

    with torch.no_grad():
        actual = encoder(clips)
        grid_thw = torch.tensor([[5, 14, 14]], device=device)
        block_major, deepstack = encoder.backbone(
            encoder._patchify(clips), grid_thw=grid_thw
        )
        if deepstack:
            raise RuntimeError("pinned InternVideo3 checkpoint unexpectedly uses deepstack")
        expected = encoder._restore_row_major(
            encoder.final_norm(block_major),
            batch_size=1,
            temporal_steps=5,
        ).flatten(1, 2)

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    difference = (actual.float() - expected.float()).abs()
    print(
        f"validated vectorized InternVideo3 forward: shape={tuple(actual.shape)} "
        f"max_abs={difference.max().item():.6g} "
        f"mean_abs={difference.mean().item():.6g}"
    )


if __name__ == "__main__":
    main()
