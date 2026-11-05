"""V-JEPA layer selection, static reductions and trainable AdaFuse.

The encoder owns condition/future slicing and view layout. These functions
operate on single-view grids; neither reduction mixes time or camera views.
"""

from collections.abc import Iterable, Sequence
import math
from typing import Literal, get_args

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

FusionMethod = Literal["last_layer", "mean", "mean_plus_global", "concat_before_bottleneck", "adafuse"]
FusionLayers = Sequence[int] | Literal["all_layers"]
FUSION_METHODS = get_args(FusionMethod)
DEFAULT_FUSION_LAYERS = (27, 29, 31, 33, 35, 37, 39)


def normalize_layer_feature(feature: Tensor) -> Tensor:
    """Non-affine per-token LN0, accumulated in fp32."""
    return F.layer_norm(feature.float(), (feature.shape[-1],), eps=1e-5)


def concat_feature_metadata(layers: Sequence[int], embed_dim: int, depth: int) -> dict:
    """Channel/normalization contract shared with header-only Plus preparation."""
    return {
        "encoder_depth": depth,
        "source_channels": embed_dim,
        "latent_channels": len(layers) * embed_dim,
        "source_norms": ["ln0"] * (len(layers) - 1) + ["pretrained_final"],
        "feature_norm": "per_source",
    }


def adafuse_feature_metadata(layers: Sequence[int], embed_dim: int, depth: int) -> dict:
    """Saved normalization/gradient contract for independently learned branches."""
    return {
        "encoder_depth": depth,
        "source_channels": embed_dim,
        "latent_channels": embed_dim,
        "adafuse": {
            "source_norms": ["pretrained_final_then_ln0" if i == depth - 1 else "ln0" for i in layers],
            "softmax_axis": "layers_per_channel",
            "output_norm": "trainable_layer_norm_after_temporal_pool",
            "norm_eps": 1e-5,
            "independent_branches": True,
            "future_gradient": "input_only_detached_target",
        },
    }


class AdaFuse(nn.Module):
    """LeWAM's static channelwise fusion, with optional future temporal pooling.

    Inputs are frozen layer grids [B, D, T, H, W]. The encoder supplies the
    pretrained-norm output for the true final block, raw outputs otherwise.
    Softmax mixes layers independently per channel, never time/space/views.
    This module belongs to the trainable backbone, outside the frozen encoder.
    """

    def __init__(self, num_layers: int, dim: int, *, temporal_pool_stride: int = 1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("adafuse requires at least two fusion_layers")
        self.temporal_pool_stride = temporal_pool_stride
        self.layer_logits = nn.Parameter(torch.zeros(num_layers, dim))
        # Start with 80% on the deepest selected source, 20% on the others.
        with torch.no_grad():
            self.layer_logits[-1].fill_(math.log(4 * (num_layers - 1)))
        self.post_fusion_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != self.layer_logits.shape[0]:
            raise ValueError("AdaFuse source count differs from fusion_layers")
        weights = self.layer_logits.float().softmax(dim=0).to(features[0].dtype)
        fused = None
        for feature, weight in zip(features, weights):
            # LN0 has no trainable affine; retain the feature dtype like the
            # official LayerNorm, while computing its statistics in fp32.
            normalized = normalize_layer_feature(feature.movedim(1, -1)).to(feature.dtype)
            weighted = normalized * weight
            fused = weighted if fused is None else fused + weighted
        stride = self.temporal_pool_stride
        if stride > 1:
            b, t, h, w, d = fused.shape
            fused = fused.reshape(b, t // stride, stride, h, w, d).mean(dim=2)
        return self.post_fusion_norm(fused).movedim(-1, 1).contiguous()


def resolve_fusion_layers(layers: FusionLayers, depth: int) -> tuple[int, ...]:
    """Resolve block indices once; the default seven-layer selection targets G."""
    if isinstance(layers, str):
        if layers != "all_layers":
            raise ValueError("fusion_layers must be a list of block indices or 'all_layers'")
        return tuple(range(depth))
    indices = tuple(layers)
    if not indices or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise ValueError("fusion_layers must be a non-empty list of integer block indices")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("fusion_layers must be strictly increasing without duplicates")
    if indices[0] < 0 or indices[-1] >= depth:
        raise ValueError(f"fusion_layers {indices} exceed the block range [0, {depth - 1}]")
    return indices


def fuse_layer_features(features: Iterable[Tensor], method: FusionMethod) -> Tensor:
    """Fuse raw block grids [B_view, T, H, W, D] without stacking all layers.

    Each token gets a non-affine LayerNorm over D (eps=1e-5). Reductions
    accumulate in fp32, then return the ViT dtype. ``mean_plus_global`` adds
    the deepest selected layer's H/W mean separately for each time and view.
    ``last_layer`` bypasses this function to preserve the native output norm.
    """
    fusion = LayerFeatureFusion(method)
    for feature in features:
        fusion.update(feature)
    return fusion.result()


class LayerFeatureFusion:
    """Streaming reduction shared by the main and auxiliary target branches."""

    def __init__(self, method: FusionMethod):
        if method not in ("mean", "mean_plus_global"):
            raise ValueError(f"Expected a fusion method, got {method!r}")
        self.method = method
        self.total = None
        self.count = 0
        self.global_mean = None
        self.dtype = None

    def update(self, feature: Tensor) -> None:
        normalized = normalize_layer_feature(feature)
        self.total = normalized if self.total is None else self.total + normalized
        self.count += 1
        self.dtype = feature.dtype
        if self.method == "mean_plus_global":
            self.global_mean = normalized.mean(dim=(2, 3), keepdim=True)

    def result(self) -> Tensor:
        if self.count == 0:
            raise ValueError("Fusion requires at least one layer")
        fused = self.total / self.count
        if self.global_mean is not None:
            fused = fused + self.global_mean
        return fused.to(dtype=self.dtype)
