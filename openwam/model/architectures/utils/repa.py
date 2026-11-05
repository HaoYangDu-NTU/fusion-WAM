"""Spatial helpers for future-only, shared-projection REPA alignment."""

import torch.nn.functional as F
from torch import Tensor


def video_grid_to_view_maps(grid: Tensor, num_views: int) -> Tensor:
    """[B,D,T,H,V*W] -> [B*T*V,D,H,W], preserving t-h-view-w packing."""
    if grid.ndim != 5 or num_views < 1 or grid.shape[-1] % num_views:
        raise ValueError(f"Invalid REPA grid/views: {tuple(grid.shape)}, views={num_views}")
    b, d, t, h, width = grid.shape
    w = width // num_views
    return grid.reshape(b, d, t, h, num_views, w).permute(0, 2, 4, 1, 3, 5).reshape(b * t * num_views, d, h, w)


def spatial_normalize(features: Tensor, gamma: float = 0.8, eps: float = 1e-6) -> Tensor:
    """Per-channel spatial z-score, independently for each time and camera.

    gamma controls mean subtraction, not standard-deviation exponentiation.
    Match the iREPA sample std; a one-position grid uses correction=0.
    """
    features = features.float()
    mean = features.mean(dim=(-2, -1), keepdim=True)
    correction = int(features.shape[-2] * features.shape[-1] > 1)
    std = features.std(dim=(-2, -1), correction=correction, keepdim=True)
    return (features - gamma * mean) / (std + eps)


def masked_spatial_cosine_loss(student: Tensor, target: Tensor, valid_frames: Tensor) -> Tensor:
    """Mean (1 - cosine) over valid spatial tokens; inputs are [B*T*V,D,H,W]."""
    if student.shape != target.shape or student.ndim != 4:
        raise ValueError(f"REPA shape mismatch: {tuple(student.shape)} vs {tuple(target.shape)}")
    if valid_frames.shape != (student.shape[0],):
        raise ValueError(f"REPA frame mask shape mismatch: {tuple(valid_frames.shape)}")
    student = F.normalize(student.float(), dim=1, eps=1e-6)
    target = F.normalize(target.float(), dim=1, eps=1e-6)
    per_token = 1 - (student * target).sum(dim=1)
    valid = valid_frames[:, None, None].expand_as(per_token)
    return per_token.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
