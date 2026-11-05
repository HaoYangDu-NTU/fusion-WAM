"""V-JEPA 2.1 video encoder (Mur-Labadia et al., arXiv:2603.14482).

Plugs into the :class:`VideoBackbone` external-encoder path introduced in
PR #60. ``properties.pixel_decode=False`` — the host backbone must rebuild its
DiT input projection via the ``build_dit_input_proj`` hook and skip the
strict native-VAE spec validation. Future frames are compressed by tubelet=2
and optionally averaged over time with stride=2. ``history_num_tubelets=H``
contributes H independently encoded clean latents; H=0 uses the legacy
current-observation condition (one latent). Thus H=1 preserves the original
clean/future layout while larger H extends only its clean prefix.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Literal, get_args

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from openwam.history import history_from_config, validate_history_num_tubelets
from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.bottleneck import BottleneckPatchEmbed
from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.svae import reducer
from openwam.model.video_backbone.encoder.vjepa21_fusion import (
    AdaFuse,
    DEFAULT_FUSION_LAYERS,
    FUSION_METHODS,
    FusionLayers,
    FusionMethod,
    LayerFeatureFusion,
    adafuse_feature_metadata,
    concat_feature_metadata,
    fuse_layer_features,
    normalize_layer_feature,
    resolve_fusion_layers,
)
from openwam.model.video_backbone.encoder.vjepa21_src import loader
from openwam.train.utils.step_timing import phase, timed

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# vjepa21_forward selects how the H=0 current-observation latent is obtained in
# ``batch_encode`` (see its docstring): ``"video"`` (default) routes a dup'd
# 2-frame clip through the video branch; ``"mixed"`` routes f0 through the
# image branch. The default is a deliberate departure from pre-PR-#92
# (≡ ``"mixed"``); migration notes live in :meth:`from_skeleton`.
# With history enabled both modes use the real two-frame video tubelet.
_VJEPA21Forward = Literal["video", "mixed"]
_VJEPA21_FORWARD_DEFAULT: _VJEPA21Forward = "video"
# Derived from the Literal so the runtime whitelist can't drift from it.
_VJEPA21_FORWARD_ALLOWED: tuple[_VJEPA21Forward, ...] = get_args(_VJEPA21Forward)


@register_video_encoder("vjepa21")
class VJEPA21VideoEncoder(VideoEncoder):
    """V-JEPA 2.1 video encoder.

    Constructor takes an already-built ViT module so unit tests can inject a
    mock without going through ``loader`` (which builds the vendored ViT and
    needs a local checkpoint file).
    """

    def __init__(
        self,
        vit: nn.Module,
        *,
        embed_dim: int,
        variant: str,
        vjepa21_forward: _VJEPA21Forward = _VJEPA21_FORWARD_DEFAULT,
        use_avg_t_pooling: bool = True,
        use_wan_style_patch_grid: bool = True,
        history_num_tubelets: int = 1,
        seperate_encode_view: bool = False,
        add_input_bottleneck: bool = False,
        bottleneck_dim: int = 512,
        condition_fusion_method: FusionMethod = "last_layer",
        future_fusion_method: FusionMethod = "last_layer",
        fusion_layers: FusionLayers = DEFAULT_FUSION_LAYERS,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
        svae_config: dict | None = None,
    ):
        super().__init__()
        # ``loader`` installs a RoPE monkey-patch that casts the rotated
        # Q/K back to ``x.dtype``: upstream would promote them to fp32 (fp32
        # sin/cos table), mismatching bf16 V at SDPA on this frozen ViT.
        self._m = vit
        self._variant = variant
        if vjepa21_forward not in _VJEPA21_FORWARD_ALLOWED:
            raise ValueError(f"vjepa21_forward must be one of {_VJEPA21_FORWARD_ALLOWED}, got {vjepa21_forward!r}.")
        self._vjepa21_forward: _VJEPA21Forward = vjepa21_forward
        self._use_avg_t_pooling = bool(use_avg_t_pooling)
        self._use_wan_style_patch_grid = bool(use_wan_style_patch_grid)
        self._history_num_tubelets = validate_history_num_tubelets(history_num_tubelets)
        self._seperate_encode_view = bool(seperate_encode_view)
        if not isinstance(add_input_bottleneck, bool):
            raise ValueError("add_input_bottleneck must be a boolean")
        self._add_input_bottleneck = add_input_bottleneck
        self._bottleneck_dim = bottleneck_dim
        if add_input_bottleneck and (
            isinstance(bottleneck_dim, bool)
            or not isinstance(bottleneck_dim, int)
            or bottleneck_dim <= 0
        ):
            raise ValueError("bottleneck_dim must be a positive integer when add_input_bottleneck=true")
        self._raw_embed_dim = int(embed_dim)
        for name, method in (("condition_fusion_method", condition_fusion_method),
                             ("future_fusion_method", future_fusion_method)):
            if method not in FUSION_METHODS:
                raise ValueError(f"{name} must be one of {FUSION_METHODS}, got {method!r}")
        self._condition_fusion_method = condition_fusion_method
        self._future_fusion_method = future_fusion_method
        # last_layer needs neither intermediate outputs nor a depth-compatible
        # layer list (e.g. the G defaults may be unused with a distilled L).
        self._fusion_layers = (
            resolve_fusion_layers(fusion_layers, len(vit.blocks))
            if condition_fusion_method != "last_layer" or future_fusion_method != "last_layer"
            else ()
        )
        self._concat_features = "concat_before_bottleneck" in (condition_fusion_method, future_fusion_method)
        if self.has_learned_fusion:
            if len(self._fusion_layers) < 2:
                raise ValueError("adafuse requires at least two fusion_layers")
            if any(value is not None for value in (svae_path, svae_config, svae_target_dim)):
                raise ValueError("adafuse requires raw V-JEPA features without S-VAE")
        if self._concat_features:
            if condition_fusion_method != future_fusion_method:
                raise ValueError("concat_before_bottleneck requires both condition and future to use it")
            if not add_input_bottleneck:
                raise ValueError("concat_before_bottleneck requires add_input_bottleneck=true")
            if any(value is not None for value in (svae_path, svae_config, svae_target_dim)):
                raise ValueError("concat_before_bottleneck requires raw V-JEPA features without S-VAE")
            if self._fusion_layers[-1] != len(vit.blocks) - 1:
                raise ValueError("concat_before_bottleneck fusion_layers must include the final V-JEPA block")
        logger.info("V-JEPA feature selection: %s", self._fusion_metadata())
        # Optional non-linear S-VAE reducer, applied AFTER the cond+target cat
        # (see ``batch_encode``). When enabled it advertises ``latent_dim`` as
        # ``z_dim`` so the DiT conv / unpatchify head / freeze yaml rebuild
        # against the reduced dim.
        self._svae = reducer.build(
            svae_path,
            svae_target_dim,
            svae_config,
            use_avg_t_pooling=self._use_avg_t_pooling,
        )
        feature_dim = self._raw_embed_dim * (len(self._fusion_layers) if self._concat_features else 1)
        effective_z_dim = reducer.effective_z_dim(self._svae, feature_dim)
        self._spec = VideoEncoderProperties(
            z_dim=int(effective_z_dim),
            spatial_compression=16,
            # 4 = ViT tubelet=2 × the ``_TARGET_TEMPORAL_POOL_STRIDE`` avg-pool,
            # matching Wan VAE causal grouping (1 cond + 1 latent / 4 frames) so
            # ``(T_pix - 1) // tc + 1`` lands on the same T_lat.
            temporal_compression=2 * self._TARGET_TEMPORAL_POOL_STRIDE if self._use_avg_t_pooling else 2,
            causal_temporal=True,
            pixel_decode=False,
            # Wan-style merges 2x2 spatial neighbors; otherwise each V-JEPA
            # latent location becomes its own DiT token.
            dit_patch_size=(1, 2, 2) if self._use_wan_style_patch_grid else (1, 1, 1),
        )
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        # Per-token standardization pulling V-JEPA's O(30)-scale features to
        # Wan-latent O(1). Structurally trainable but a fixed standardizer in
        # practice: the encode path runs under ``@torch.no_grad`` and the freeze
        # yaml freezes ``video_backbone.video_encoder`` wholesale. Lives OUTSIDE
        # ``self._m`` so a future PR can make it grad-enabled without touching
        # the ViT freeze granularity. (With the S-VAE on, a near-identity affine
        # over the already-whitened reduced dim.)
        # Concat keeps each source's normalization independent. In particular,
        # the native final-layer slice must remain identical to the M29 path.
        self.feature_norm = nn.LayerNorm(self._raw_embed_dim if self._concat_features else int(effective_z_dim))

    def build_dit_input_proj(self, dit_dim: int) -> nn.Module:
        if not self._add_input_bottleneck:
            return super().build_dit_input_proj(dit_dim)
        return BottleneckPatchEmbed(
            self.properties.z_dim, dit_dim, self.properties.dit_patch_size, self._bottleneck_dim,
        )

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    @property
    def latent_channel_groups(self) -> tuple[tuple[str, int], ...]:
        if not self._concat_features:
            return ()
        return tuple((f"layer_{index}", self._raw_embed_dim) for index in self._fusion_layers)

    @property
    def condition_fusion_method(self) -> FusionMethod:
        return self._condition_fusion_method

    @property
    def future_fusion_method(self) -> FusionMethod:
        return self._future_fusion_method

    @property
    def fusion_layers(self) -> tuple[int, ...]:
        """Resolved block indices; empty when both branches use last_layer."""
        return self._fusion_layers

    @property
    def has_learned_fusion(self) -> bool:
        return "adafuse" in (self._condition_fusion_method, self._future_fusion_method)

    def build_feature_fuser(self, *, future: bool) -> nn.Module | None:
        """Construct a backbone-owned module; never register it on this encoder."""
        method = self._future_fusion_method if future else self._condition_fusion_method
        if method != "adafuse":
            return None
        stride = self._TARGET_TEMPORAL_POOL_STRIDE if future and self._use_avg_t_pooling else 1
        return AdaFuse(len(self._fusion_layers), self._raw_embed_dim, temporal_pool_stride=stride)

    def _fusion_metadata(self) -> dict:
        metadata = {
            "condition_fusion_method": self._condition_fusion_method,
            "future_fusion_method": self._future_fusion_method,
            "fusion_layers": list(self._fusion_layers),
        }
        if self._concat_features:
            metadata.update(concat_feature_metadata(self._fusion_layers, self._raw_embed_dim, len(self._m.blocks)))
        if self.has_learned_fusion:
            metadata.update(adafuse_feature_metadata(self._fusion_layers, self._raw_embed_dim, len(self._m.blocks)))
        return metadata

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def vjepa21_forward(self) -> _VJEPA21Forward:
        """How the H=0 current-observation latent is computed in ``batch_encode``.

        ``"video"`` (default) — dup frame 0 to a 2-frame clip and route it
        through the V-JEPA 2.1 video branch (tubelet=2).
        ``"mixed"`` — route frame 0 through the V-JEPA 2.1 image branch
        (tubelet=1). With H>0, both modes encode all 2H history/current
        images jointly through the video branch.
        """
        return self._vjepa21_forward

    @property
    def use_avg_t_pooling(self) -> bool:
        return self._use_avg_t_pooling

    @property
    def use_wan_style_patch_grid(self) -> bool:
        return self._use_wan_style_patch_grid

    @property
    def history_num_tubelets(self) -> int:
        return self._history_num_tubelets

    @property
    def seperate_encode_view(self) -> bool:
        return self._seperate_encode_view

    def preprocess_video(self, frames: List[Image.Image]) -> torch.Tensor:
        """List[PIL] -> (1, 3, T, H, W) in ImageNet-normalized space."""
        device = next(self._m.parameters()).device
        dtype = next(self._m.parameters()).dtype
        to_tensor = T.ToTensor()
        chans = [to_tensor(img) for img in frames]
        video = torch.stack(chans, dim=1).unsqueeze(0)
        video = video.to(device=device, dtype=dtype)
        video = (video - self._mean.to(dtype)) / self._std.to(dtype)
        return video

    @timed("pixel_preprocess")
    def preprocess_video_batch(self, clips: List[List[Image.Image]]) -> torch.Tensor:
        """Batch RGB bytes once; scale and normalize on the encoder device.

        Preserve ToTensor's float32 / 255 then model-dtype rounding order.
        This avoids per-frame CPU float conversion and per-clip GPU copies.
        """
        frames_per_clip = len(clips[0])
        if any(len(clip) != frames_per_clip for clip in clips):
            raise ValueError("V-JEPA preprocessing requires equal clip lengths")
        parameter = next(self._m.parameters())
        first = np.asarray(clips[0][0])
        if first.dtype != np.uint8 or first.ndim != 3 or first.shape[-1] != 3:
            raise ValueError("V-JEPA batch preprocessing expects uint8 RGB frames")
        pixels = torch.empty(
            (len(clips), frames_per_clip, *first.shape), dtype=torch.uint8,
            pin_memory=parameter.device.type == "cuda",
        )
        destinations = pixels.numpy().reshape(-1, *first.shape)
        for dst, frame in zip(destinations, (frame for clip in clips for frame in clip)):
            array = np.asarray(frame)
            if array.dtype != np.uint8 or array.shape != first.shape:
                raise ValueError("V-JEPA batch preprocessing expects equal uint8 RGB frame shapes")
            np.copyto(dst, array)
        video = pixels.to(device=parameter.device, non_blocking=True).permute(0, 4, 1, 2, 3).contiguous()
        video = video.to(torch.float32).div_(255).to(parameter.dtype)
        return video.sub_(self._mean.to(parameter.dtype)).div_(self._std.to(parameter.dtype))

    # Extra avg-pool over time (after ViT tubelet=2) so total temporal
    # compression hits 4, matching Wan VAE causal grouping. Locked to 2:
    # tubelet=2 and Wan tc=4 are both production-fixed. Class-level so the
    # "extra 2x" assumption lives in one place.
    _TARGET_TEMPORAL_POOL_STRIDE = 2

    @timed("video_encode_total")
    def batch_encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T_pixel, H, W) -> (B, z_dim, T_lat, H/16, W/16).

        The input starts with 2H clean images for H history tubelets, or one
        current image for H=0. This produces max(H,1) clean latents followed
        by targets compressed in time according to the spec.
        The raw encode + pool lives in
        :meth:`_batch_encode_pooled_raw`; here we apply the optional S-VAE
        reducer (base hook) then the per-token feature_norm.
        """

        if self.has_learned_fusion:
            raise RuntimeError("adafuse must be encoded through the backbone's encode_video, which owns the learned fusers")
        # NOTE: 具体调用 V-JEPA 的地方，两次 forward，一次编码首帧 condition，一次编码完整 video，这里已经完成了 openwam 定义的两帧 pooling，在 temporal 上再压缩一次
        z = self._batch_encode_pooled_raw(video)
        return self._normalize_latents(z)

    @torch.no_grad()
    @timed("video_encode_total")
    def batch_encode_fusion_features(self, video: torch.Tensor) -> tuple:
        """Return frozen condition/future sources, before trainable fusion.

        Static branches already include their existing pool/feature_norm.
        AdaFuse branches retain one grid per selected layer, without pooling
        or feature_norm. A condition-only call returns future=None.
        """
        condition, future = self._batch_encode_pooled_raw(video, return_branches=True)
        condition = condition if self._condition_fusion_method == "adafuse" else self._normalize_latents(condition)
        if future is not None and self._future_fusion_method != "adafuse":
            future = self._normalize_latents(future)
        return condition, future

    @torch.no_grad()
    @timed("video_encode_total")
    def batch_encode_with_repa_target(
        self, video: torch.Tensor, *, target_fusion_method: FusionMethod,
        target_fusion_layers: FusionLayers,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unchanged main latents and clean future-only REPA latents.

        The existing future encoder pass produces both representations. Target
        selection never changes the condition or main prediction space.
        """
        layers = self.resolve_repa_target_layers(target_fusion_method, target_fusion_layers)
        main, target = self._batch_encode_pooled_raw(video, repa_target=(target_fusion_method, layers))
        return self._normalize_latents(main), self._normalize_latents(target)

    def resolve_repa_target_layers(self, method: FusionMethod, layers: FusionLayers) -> tuple[int, ...]:
        """Validate auxiliary selection without changing the main encoder config."""
        if self._concat_features or method == "concat_before_bottleneck":
            raise ValueError("REPA does not support concat_before_bottleneck; set repa_enabled=false")
        if self.has_learned_fusion or method == "adafuse":
            raise ValueError("REPA requires a static feature target; adafuse currently requires repa_enabled=false")
        if self._svae is not None:
            raise ValueError("REPA currently requires raw V-JEPA features (svae_path=null)")
        if method not in FUSION_METHODS:
            raise ValueError(f"Invalid repa_target_fusion_method: {method!r}")
        return resolve_fusion_layers(layers, len(self._m.blocks)) if method != "last_layer" else ()

    def _normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        # 调用预训练好的 svae 做降维
        z = self._apply_svae(z)
        # Per-token feature_norm AFTER the pool so the LayerNorm re-standardizes
        # the post-pool distribution: (B,D,T,h,w) -> flatten tokens -> LN -> back.
        B, D, Tp, h, w = z.shape
        # Concat sources are contiguous blocks of raw_embed_dim channels. Fold
        # the source axis into the token batch so LN never mixes source stats.
        norm_dim = self._raw_embed_dim if self._concat_features else D
        z = z.permute(0, 2, 3, 4, 1).reshape(-1, norm_dim)
        z = self.feature_norm(z)
        return z.view(B, Tp, h, w, D).permute(0, 4, 1, 2, 3).contiguous()

    def _batch_encode_pooled_raw(
        self, video: torch.Tensor, *, repa_target: tuple[FusionMethod, tuple[int, ...]] | None = None,
        return_branches: bool = False,
    ) -> torch.Tensor | tuple:
        """Encoder forward + temporal mean-pool, BEFORE S-VAE / feature_norm.

        Returns ``(B, C, T_lat, H/16, W/16)`` condition + target latents, where
        C is raw_embed_dim, or source_count * raw_embed_dim for concat. Split
        from :meth:`batch_encode` so offline S-VAE training
        (:meth:`batch_encode_pooled_for_svae_training`) sees the same post-pool
        distribution. The two passes:

        - Condition (no target leakage): encode all 2H history/current images
          jointly, yielding H latents. At H=0, ``mixed`` uses the current image
          and ``video`` duplicates it; both yield one clean latent.
        - Target: encode the FULL clean clip followed by future frames, drop
          all max(H,1) clean positions, then optionally pool only the future.
          The separately encoded condition never sees future observations.
        """
        B, C, Tp, H, W = video.shape
        if C != 3:
            raise ValueError(f"V-JEPA 2.1 expects 3-channel input; got C={C}.")
        # Align input to ViT param dtype: the RoPE monkey-patch only guarantees
        # Q/K match x.dtype at SDPA, not the input matmuls.
        m_dtype = next(self._m.parameters()).dtype
        if video.dtype != m_dtype:
            video = video.to(m_dtype)
        context_frames = max(1, 2 * self._history_num_tubelets)
        num_clean_latents = max(1, self._history_num_tubelets)
        if Tp < context_frames:
            raise ValueError(f"history_num_tubelets={self._history_num_tubelets} requires {context_frames} clean condition images.")
        f0 = video[:, :, 0:1]
        context_clip = video[:, :, :context_frames] if self._history_num_tubelets else torch.cat([f0, f0], dim=2)
        if self._vjepa21_forward == "mixed" and not self._history_num_tubelets:
            z_cond = self._vit_grid(f0, method=self._condition_fusion_method)
        else:
            z_cond = self._vit_grid(context_clip, method=self._condition_fusion_method)
        if Tp == context_frames:
            if repa_target is not None:
                raise ValueError("REPA requires at least one future tubelet")
            return (z_cond, None) if return_branches else z_cond
        # Only future frames enter the tubelet/pooling divisibility check.
        divisor = 2 * self._TARGET_TEMPORAL_POOL_STRIDE if self._use_avg_t_pooling else 2
        if (Tp - context_frames) % divisor != 0:
            raise ValueError(
                f"V-JEPA 2.1 causal emulation needs (T_pixel - {context_frames}) % {divisor} == 0, got T_pixel={Tp}."
            )
        z_target = self._vit_grid(
            torch.cat([context_clip, video[:, :, context_frames:]], dim=2),
            method=self._future_fusion_method,
            start_frame=num_clean_latents,
            repa_target=repa_target,
        )
        if repa_target is not None:
            z_target, z_repa = z_target
        if self._use_avg_t_pooling and self._future_fusion_method != "adafuse":
            z_target = self._pool_target_temporal(z_target)
            if repa_target is not None:
                z_repa = self._pool_target_temporal(z_repa)
        if return_branches:
            return z_cond, z_target
        main = torch.cat([z_cond, z_target], dim=2)
        return main if repa_target is None else (main, z_repa)

    def batch_encode_pooled_for_svae_training(self, video: torch.Tensor) -> torch.Tensor:
        """Raw post-pool features for offline S-VAE training / stats collection.

        Returns the raw ``(B, C, T_lat, H/16, W/16)`` grid (C includes all
        concat sources): max(H,1) un-pooled conditions + optionally pooled targets,
        BEFORE any reduction or ``feature_norm``. Both sub-populations (cond and
        target) are included so the reducer's prior covers what it sees at
        inference. Fails fast if an S-VAE is already attached: statistics must be
        collected on a raw encoder, never on an already-reduced one.
        """
        if self.has_learned_fusion:
            raise ValueError("adafuse requires the learned backbone fusers, not offline S-VAE encoding")
        if self._svae is not None:
            raise RuntimeError(
                "batch_encode_pooled_for_svae_training requires a raw encoder; "
                "svae_path / svae_config / svae_target_dim must be unset."
            )
        return self._batch_encode_pooled_raw(video)

    def _vit_grid(
        self, x5: torch.Tensor, *, method: FusionMethod = "last_layer", start_frame: int = 0,
        repa_target: tuple[FusionMethod, tuple[int, ...]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """(B, 3, T, H, W) -> (B, D, T_lat, H/16, W/16) via the V-JEPA 2.1 ViT.

        ``T == 1`` selects the image branch (tubelet=1, T_lat=1); ``T > 1`` the
        video branch (tubelet=2, T_lat=T/2). ``start_frame`` slices latent
        time AFTER the full clip interacts in V-JEPA, BEFORE feature fusion.
        Views are fused independently and only then packed along width.
        """
        B, C, T, H, W = x5.shape
        if self._seperate_encode_view:
            if (H, W) != (224, 448):
                raise ValueError("seperate_encode_view requires two 224x224 views packed as a 224x448 canvas")
            # Independent views share weights, never encoder attention.
            x5 = x5.reshape(B, C, T, H, 2, 224).permute(0, 4, 1, 2, 3, 5).reshape(B * 2, C, T, H, 224)
        with phase("encoder_context" if T <= 2 else "encoder_teacher"):
            if repa_target is not None:
                methods = (method, repa_target[0])
                selections = (set(self._fusion_layers), set(repa_target[1]))
                fusions = [None if m == "last_layer" else LayerFeatureFusion(m) for m in methods]
                selected = set()
                for selection, fusion in zip(selections, fusions):
                    if fusion is not None:
                        selected.update(selection)
                selected = sorted(selected)
                features = self._m.iter_intermediate_layers(
                    x5, selected, include_final_output="last_layer" in methods,
                )
                for layer_id in selected:
                    feature = next(features)[:, start_frame:]
                    for selection, fusion in zip(selections, fusions):
                        if fusion is not None and layer_id in selection:
                            fusion.update(feature)
                native = next(features)[:, start_frame:] if "last_layer" in methods else None
                # Finish the same block traversal even when only early layers
                # were selected, preserving the encoder's forward behavior.
                for _ in features:
                    pass
                grids = [native if fusion is None else fusion.result() for fusion in fusions]
                return tuple(self._pack_view_grid(grid, B) for grid in grids)
            elif method == "adafuse":
                has_final = self._fusion_layers[-1] == len(self._m.blocks) - 1
                selected = self._fusion_layers[:-1] if has_final else self._fusion_layers
                features = self._m.iter_intermediate_layers(x5, selected, include_final_output=has_final)
                # No spatial/temporal reduction: D stays the true channel axis.
                return tuple(self._pack_view_grid(x[:, start_frame:], B) for x in features)
            elif method == "concat_before_bottleneck":
                # The sorted selection includes the true final block exactly
                # once. Use its pretrained norm, not LN0(raw final block).
                features = self._m.iter_intermediate_layers(
                    x5, self._fusion_layers[:-1], include_final_output=True,
                )
                grids = []
                for _ in self._fusion_layers[:-1]:
                    feature = next(features)[:, start_frame:]
                    grids.append(normalize_layer_feature(feature).to(feature.dtype))
                grids.append(next(features)[:, start_frame:])
                grid = torch.cat(grids, dim=-1)
            elif method == "last_layer":
                # Preserve the pretrained final norm, not LN(raw last block).
                flat = self._m(x5)
                t_lat = 1 if T == 1 else T // 2
                h, w = x5.shape[-2] // 16, x5.shape[-1] // 16
                if flat.shape[1:] != (t_lat * h * w, self._raw_embed_dim):
                    raise ValueError("V-JEPA output does not match the single-view patch grid")
                grid = flat.reshape(x5.shape[0], t_lat, h, w, self._raw_embed_dim)[:, start_frame:]
            else:
                features = self._m.iter_intermediate_layers(x5, self._fusion_layers)
                grid = fuse_layer_features((x[:, start_frame:] for x in features), method)
        return self._pack_view_grid(grid, B)

    def _pack_view_grid(self, grid: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Pack independent camera grids in the existing t-h-view-w order."""
        _, t_lat, h, w, dim = grid.shape
        if self._seperate_encode_view:
            return (
                grid.reshape(batch_size, 2, t_lat, h, w, dim)
                .permute(0, 5, 2, 3, 1, 4).flatten(-2).contiguous()
            )
        return grid.permute(0, 4, 1, 2, 3).contiguous()

    def _pool_target_temporal(self, z_target: torch.Tensor) -> torch.Tensor:
        """(B, D, T_raw, h, w) -> (B, D, T_raw/2, h, w): avg-pool over time with
        stride ``_TARGET_TEMPORAL_POOL_STRIDE`` (a mean of each consecutive pair,
        NOT stride-2 indexing). Kept a named method so V5b can pin the mean
        semantics against a silent regression that would still pass shape checks.
        """
        s = self._TARGET_TEMPORAL_POOL_STRIDE
        B, D, T, h, w = z_target.shape
        return z_target.reshape(B, D, T // s, s, h, w).mean(dim=3)

    def decode(self, latents: torch.Tensor, **kw: Any) -> torch.Tensor:
        raise NotImplementedError(
            "VJEPA21VideoEncoder is irreversible (properties.pixel_decode=False); "
            "pixel decode is not defined. Pass decode_video=False to generate()."
        )

    def to_frames(self, video: torch.Tensor) -> list:
        raise NotImplementedError("VJEPA21VideoEncoder is irreversible; to_frames has no meaning.")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        vjepa21_forward: _VJEPA21Forward = _VJEPA21_FORWARD_DEFAULT,
        use_avg_t_pooling: bool = True,
        use_wan_style_patch_grid: bool = True,
        history_num_tubelets: int = 1,
        seperate_encode_view: bool = False,
        add_input_bottleneck: bool = False,
        bottleneck_dim: int = 512,
        condition_fusion_method: FusionMethod = "last_layer",
        future_fusion_method: FusionMethod = "last_layer",
        fusion_layers: FusionLayers = DEFAULT_FUSION_LAYERS,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
    ) -> "VJEPA21VideoEncoder":
        # Explicit signature (no ``**kw``) so a typo'd yaml / programmatic field
        # raises TypeError instead of being silently ignored — ``build_video_encoder``
        # forwards every non-{name, model_path} yaml field straight here.
        manifest = loader.read_and_validate_manifest(model_path)
        vit_encoder = loader.prepare_vjepa_imports_and_patch()
        vit = loader.build_vit_from_manifest(vit_encoder, manifest)
        loader.load_vit_weights(vit, model_path, manifest)
        return cls(
            vit,
            embed_dim=int(manifest["embed_dim"]),
            variant=str(manifest["variant"]),
            vjepa21_forward=vjepa21_forward,
            use_avg_t_pooling=use_avg_t_pooling,
            use_wan_style_patch_grid=use_wan_style_patch_grid,
            history_num_tubelets=history_num_tubelets,
            seperate_encode_view=seperate_encode_view,
            add_input_bottleneck=add_input_bottleneck,
            bottleneck_dim=bottleneck_dim,
            condition_fusion_method=condition_fusion_method,
            future_fusion_method=future_fusion_method,
            fusion_layers=fusion_layers,
            svae_path=svae_path,
            svae_target_dim=svae_target_dim,
        )

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "VJEPA21VideoEncoder":
        """Deploy-time zero-weight ViT shell, sized by manifest.

        Unlike :class:`WanVideoVAEEncoder` (which reconstructs from the saved
        ``components_entry``), V-JEPA's training-time component-spec
        generator did not persist ViT geometry into ``config.yaml`` for
        PR #67-era runs — ``components[vae]`` actually carries the Wan
        ``WanVideoVAE38`` class (an artifact of ``get_component_specs``
        scanning the Wan ``model_path``). We therefore ignore
        ``components_entry`` and reconstruct from
        ``<ckpt_dir>/manifest.json``, written by :meth:`save_deploy_assets`
        at checkpoint save time. Deploy is strictly self-contained: there is
        no ``encoder.model_path`` fallback, so a checkpoint saved without its
        manifest fails loudly here rather than silently reaching back to a
        training-time path that may be unmounted on the deploy host.

        ViT weights are NOT loaded here — the architecture's strict
        ``load_checkpoint`` populates ``video_encoder._m.*`` from the saved
        safetensors immediately after this call returns.
        """
        manifest_dir = loader.resolve_manifest_dir(ckpt_dir)
        manifest = loader.read_and_validate_manifest(manifest_dir)
        vit_encoder = loader.prepare_vjepa_imports_and_patch()
        with torch.device(device):
            vit = loader.build_vit_from_manifest(vit_encoder, manifest)
        # ``vjepa21_forward`` is a runtime knob plumbed through the yaml
        # ``encoder`` block so a checkpoint+yaml pair rebuilds the same encoder
        # the run trained. We warn (only) when a saved yaml omits it: that is
        # the pre-PR-#92 checkpoint signature, and those ckpts are NOT forward-
        # compatible (temporal_compression 2→4 + the 2-pass target rewrite) and
        # must be retrained. ``encoder_cfg is None`` (programmatic callers) is
        # not warned — there is no saved yaml to fix.
        vjepa21_forward = loader.read_vjepa21_forward_from_cfg(encoder_cfg, _VJEPA21_FORWARD_DEFAULT)
        use_avg_t_pooling = loader.read_use_avg_t_pooling_from_cfg(encoder_cfg, True)
        use_wan_style_patch_grid = (
            encoder_cfg.get("use_wan_style_patch_grid")
            if isinstance(encoder_cfg, dict)
            else getattr(encoder_cfg, "use_wan_style_patch_grid", None)
        )
        if use_wan_style_patch_grid is None:
            use_wan_style_patch_grid = True
        history_num_tubelets = history_from_config(encoder_cfg)
        seperate_encode_view = (
            encoder_cfg.get("seperate_encode_view", False)
            if isinstance(encoder_cfg, dict)
            else getattr(encoder_cfg, "seperate_encode_view", False)
        )
        add_input_bottleneck = (
            encoder_cfg.get("add_input_bottleneck", False)
            if isinstance(encoder_cfg, dict)
            else getattr(encoder_cfg, "add_input_bottleneck", False)
        )
        bottleneck_dim = (
            encoder_cfg.get("bottleneck_dim", 512)
            if isinstance(encoder_cfg, dict)
            else getattr(encoder_cfg, "bottleneck_dim", 512)
        )
        fusion_options = {}
        for name, default in (("condition_fusion_method", "last_layer"),
                              ("future_fusion_method", "last_layer"),
                              ("fusion_layers", DEFAULT_FUSION_LAYERS)):
            value = (encoder_cfg.get(name, default) if isinstance(encoder_cfg, dict)
                     else getattr(encoder_cfg, name, default))
            fusion_options[name] = default if value is None else value
        if encoder_cfg is not None and not loader.cfg_has_vjepa21_forward(encoder_cfg):
            logger.warning(
                "VJEPA21VideoEncoder.from_skeleton: saved encoder yaml has no "
                "``vjepa21_forward`` field; defaulting to %r. A pre-PR-#92 "
                "checkpoint is NOT forward-compatible (temporal_compression 2→4 "
                "+ 2-pass target rewrite) and must be retrained; setting "
                "``vjepa21_forward: mixed`` only restores the cond-frame branch.",
                vjepa21_forward,
            )
        # Reducer rebuild: the sidecar config (if the training ckpt carried an
        # S-VAE) sizes a zero-weight shell here; strict ``load_checkpoint``
        # fills ``_svae.*`` right after. ``svae_target_dim`` from the saved yaml
        # is an optional cross-check against the sidecar's ``latent_dim``.
        svae_config = reducer.read_sidecar(ckpt_dir)
        svae_target_dim = reducer.read_target_dim_from_cfg(encoder_cfg)
        logger.info(
            "VJEPA21VideoEncoder.from_skeleton: %s instantiated from %s "
            "(embed_dim=%d, variant=%s, vjepa21_forward=%s, svae=%s) — weights pending checkpoint load",
            manifest["arch_name"],
            manifest_dir,
            int(manifest["embed_dim"]),
            str(manifest["variant"]),
            vjepa21_forward,
            "on" if svae_config is not None else "off",
        )
        encoder = cls(
            vit,
            embed_dim=int(manifest["embed_dim"]),
            variant=str(manifest["variant"]),
            vjepa21_forward=vjepa21_forward,
            use_avg_t_pooling=use_avg_t_pooling,
            use_wan_style_patch_grid=use_wan_style_patch_grid,
            history_num_tubelets=history_num_tubelets,
            seperate_encode_view=seperate_encode_view,
            add_input_bottleneck=add_input_bottleneck,
            bottleneck_dim=bottleneck_dim,
            **fusion_options,
            svae_config=svae_config,
            svae_target_dim=svae_target_dim,
        )
        # Weights alone cannot detect changed layer indices or normalization.
        saved_fusion = manifest.get("feature_fusion")
        if (encoder._concat_features or encoder.has_learned_fusion) and saved_fusion is None:
            raise ValueError("concat_before_bottleneck / adafuse requires feature_fusion metadata in the checkpoint manifest")
        if saved_fusion is not None and saved_fusion != encoder._fusion_metadata():
            raise ValueError("V-JEPA feature fusion config differs from the checkpoint manifest")
        return encoder

    def save_deploy_assets(self, output_dir: str, cfg: Any) -> None:
        """Copy ``manifest.json`` from ``encoder.model_path`` into
        ``<output_dir>/manifest.json`` so deploy is self-contained.

        Strict self-contained — no deploy-time fallback: an unresolvable cfg,
        a missing source manifest, or a copy IO error all raise, because
        :meth:`from_skeleton` reads the manifest only from ``ckpt_dir``; a
        checkpoint saved without its manifest cannot be deployed. This runs
        once at rank-0 start-up before any weights are saved, so a raise fails
        the run fast instead of producing deploy-unloadable checkpoints.

        The source is re-read from
        ``cfg.model.video_backbone.encoder.model_path`` (the yaml field
        training read at construction) rather than cached on ``self``, so the
        call survives a future cfg-layout refactor.
        """
        import shutil

        # The S-VAE sidecar is likewise strict (``reducer.write_sidecar`` raises):
        # without it ``from_skeleton`` cannot size the reducer.
        if self._svae is not None:
            reducer.write_sidecar(self._svae, output_dir, type(self).__name__)

        try:
            enc_cfg = cfg.model.video_backbone.encoder
            if isinstance(enc_cfg, dict):
                model_path = enc_cfg.get("model_path")
            else:
                model_path = getattr(enc_cfg, "model_path", None)
        except Exception:
            # cfg shape (dict / DictConfig / mock) varies; an unreadable cfg
            # collapses to model_path=None and the hard error below.
            model_path = None

        if not model_path:
            raise FileNotFoundError(
                "VJEPA21VideoEncoder.save_deploy_assets: cannot resolve "
                "model.video_backbone.encoder.model_path from cfg; cannot copy "
                "manifest.json (deploy reads it only from ckpt_dir)."
            )
        src = os.path.join(str(model_path), "manifest.json")
        dst = os.path.join(output_dir, "manifest.json")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"VJEPA21VideoEncoder.save_deploy_assets: manifest.json not found at {src}.")
        if os.path.abspath(src) == os.path.abspath(dst):
            return
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(src, dst)
        # Keep the user's all_layers shorthand in config.yaml and record its
        # resolved meaning alongside the checkpoint-local encoder geometry.
        with open(dst) as f:
            manifest = json.load(f)
        if self._fusion_layers or "feature_fusion" in manifest:
            manifest["feature_fusion"] = self._fusion_metadata()
            with open(dst, "w") as f:
                json.dump(manifest, f, indent=2)
                f.write("\n")
        logger.info("VJEPA21VideoEncoder.save_deploy_assets: copied %s -> %s", src, dst)

    # The default output projection preserves the encoder latent dimension.


__all__ = ["VJEPA21VideoEncoder"]
