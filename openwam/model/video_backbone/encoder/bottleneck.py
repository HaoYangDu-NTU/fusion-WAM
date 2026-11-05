"""Separate full-rank clean-context and low-rank future patch embeddings."""

import torch
import torch.nn as nn


class BottleneckPatchEmbed(nn.Module):
    """Keep a direct clean-prefix projection; bottleneck only the future.

    The future uses two linear convolutions without an activation.

    Input/output stay BCTHW grids, preserving the host DiT's patchify contract.
    This module belongs to the trainable DiT, not the frozen feature encoder.
    """

    def __init__(self, in_channels, embed_dim, patch_size, bottleneck_dim=512):
        super().__init__()
        if (
            isinstance(bottleneck_dim, bool)
            or not isinstance(bottleneck_dim, int)
            or bottleneck_dim <= 0
        ):
            raise ValueError("bottleneck_dim must be a positive integer")
        self.clean_proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.proj1 = nn.Conv3d(
            in_channels, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.proj2 = nn.Conv3d(bottleneck_dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x, num_clean_condition_latents=0):
        n = num_clean_condition_latents
        if not 0 <= n <= x.shape[2] or n % self.proj1.kernel_size[0]:
            raise ValueError("clean prefix must fit the latent sequence and align with temporal patches")
        if n == 0:
            return self.proj2(self.proj1(x))
        clean = self.clean_proj(x[:, :, :n])
        if n == x.shape[2]:
            return clean
        future = self.proj2(self.proj1(x[:, :, n:]))
        return torch.cat([clean, future], dim=2)
