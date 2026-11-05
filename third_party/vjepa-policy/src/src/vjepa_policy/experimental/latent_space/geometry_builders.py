"""Builders and dataset masks for the two latent-geometry experiments."""

from __future__ import annotations

import copy

import torch

from vjepa_policy.experimental.latent_space.dino_native16 import (
    DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE,
    build_dinov2_native16_policy_mask,
    load_dinov2_native16_latent_encoder,
)
from vjepa_policy.experimental.latent_space.geometry_policy import (
    build_geometry_policy,
)
from vjepa_policy.experimental.latent_space.wan22_stride2_encoder import (
    build_wan22_stride2_policy_mask,
    load_wan22_stride2_t5_latent_encoder,
)
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.wm_predictor import build_world_model_predictor


class _MutableGeometryBatchCollator:
    """Keep the collated batch mutable so consumed RGB storage can be released."""

    def __init__(self, delegate):
        self._delegate = delegate

    def __call__(self, batch):
        return list(self._delegate(batch))

    def __getattr__(self, name):
        return getattr(self._delegate, name)


DINOV2_NATIVE16 = "dinov2_native16"
WAN22_STRIDE2_T5 = "wan22_stride2_t5"
GEOMETRY_EXPERIMENTS = (DINOV2_NATIVE16, WAN22_STRIDE2_T5)


def _model_image_size(args):
    from vjepa_policy.train_common import get_model_image_size

    return get_model_image_size(args)


def _camera_keys(args):
    from vjepa_policy.train_common import get_camera_keys

    return get_camera_keys(args)


def _consume_logical_encoder_initialization(args) -> None:
    """Keep the baseline RNG advance while decoupling raw Wan frame count."""
    encoder = video_vit.vit_large(
        img_size=_model_image_size(args),
        patch_size=args.patch_size,
        num_frames=10,
        tubelet_size=args.tubelet_size,
        uniform_power=False,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        interpolate_rope=args.encoder_interpolate_rope,
    )
    del encoder


def validate_geometry_config(args, experiment: str) -> None:
    """Reject accidental changes to the shared two-view baseline contract."""
    if experiment not in GEOMETRY_EXPERIMENTS:
        raise ValueError(
            f"unsupported geometry experiment {experiment!r}; "
            f"choose one of {GEOMETRY_EXPERIMENTS}"
        )
    expected = {
        "encoder_family": "vjepa2",
        "model_name": "vit_large",
        "encoder_checkpoint_key": "target_encoder",
        "crop_size": 224,
        "patch_size": 16,
        "tubelet_size": 2,
        "context_tubelets": 1,
        "view_layout": "independent",
        "pred_depth": 24,
        "pred_embed_dim": 1024,
        "pred_num_heads": 16,
        "action_hidden_size": 512,
        "action_num_layers": 24,
        "condition_num_heads": 8,
        "predictor_rope_frequency_pairing": "corrected",
        "encoder_interpolate_rope": True,
    }
    mismatches = {
        name: (getattr(args, name, None), value)
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if _model_image_size(args) != (224, 224):
        mismatches["model_image_size"] = (_model_image_size(args), (224, 224))
    camera_keys = _camera_keys(args)
    if len(camera_keys) != 2:
        mismatches["camera_count"] = (len(camera_keys), 2)

    if experiment == DINOV2_NATIVE16:
        sampling = {
            "num_frames": 33,
            "past_frames": 4,
            "video_frame_stride": 4,
        }
        for name, value in sampling.items():
            if getattr(args, name, None) != value:
                mismatches[name] = (getattr(args, name, None), value)
    else:
        sampling = {
            "num_frames": 33,
            "past_frames": 0,
            "video_frame_stride": 2,
        }
        for name, value in sampling.items():
            if getattr(args, name, None) != value:
                mismatches[name] = (getattr(args, name, None), value)
    if mismatches:
        raise ValueError(
            f"{experiment} must keep the shared V-JEPA2-L training contract: "
            f"{mismatches}"
        )


def build_geometry_encoder(args):
    experiment = args.geometry_experiment
    validate_geometry_config(args, experiment)
    _consume_logical_encoder_initialization(args)
    with torch.random.fork_rng(devices=[]):
        if experiment == DINOV2_NATIVE16:
            encoder = load_dinov2_native16_latent_encoder(
                args.latent_checkpoint,
                image_microbatch_size=args.image_microbatch_size,
            )
        elif experiment == WAN22_STRIDE2_T5:
            encoder = load_wan22_stride2_t5_latent_encoder(
                args.latent_checkpoint,
                fastwam_src_dir=args.fastwam_src_dir,
                video_microbatch_size=args.video_microbatch_size,
            )
        else:  # pragma: no cover - validate_geometry_config guards this branch
            raise ValueError(f"unsupported geometry experiment {experiment!r}")
    return encoder


def build_geometry_predictor(*, experiment: str, **kwargs):
    """Build a predictor on the *logical* five-step latent lattice."""
    if experiment == DINOV2_NATIVE16:
        if kwargs.get("embed_dim") != 1024:
            raise ValueError("DINOv2 native16 must expose 1024-D latent tokens")
        logical = dict(kwargs)
        logical.update(
            img_size=DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE,
            patch_size=16,
            num_frames=10,
            tubelet_size=2,
        )
        return build_world_model_predictor(**logical)
    if experiment == WAN22_STRIDE2_T5:
        if kwargs.get("embed_dim") != 48:
            raise ValueError("Wan2.2 stride2 must expose 48-D latent tokens")
        logical = dict(kwargs)
        logical.update(
            img_size=(224, 224),
            patch_size=16,
            num_frames=10,
            tubelet_size=2,
        )
        # Reuse the established 48-D Wan initialization scheme, including its
        # compatible baseline weight transfer and shape checks.
        from vjepa_policy.experimental.latent_space.builders import (
            build_latent_predictor,
        )

        return build_latent_predictor(latent_encoder="wan2_2", **logical)
    raise ValueError(f"unsupported geometry experiment {experiment!r}")


def build_geometry_policy_for_experiment(experiment: str, encoder, predictor, **kwargs):
    """Inject the actual logical context length into the Action Expert."""
    if experiment not in GEOMETRY_EXPERIMENTS:
        raise ValueError(f"unsupported geometry experiment {experiment!r}")
    context_len = predictor.num_views * predictor.grid_height * predictor.grid_width
    kwargs["context_len"] = context_len
    return build_geometry_policy(encoder, predictor, **kwargs)


def build_geometry_dataset_collator(args, action_chunk_size, *, experiment: str):
    """Build the normal dataset with an experiment-specific logical mask.

    The Wan raw clip has 17 frames, which is intentionally not divisible by the
    production tubelet size.  The base dataset builder does not use
    ``tubelet_size`` while decoding; a shallow argument copy with
    ``tubelet_size=1`` lets it construct the same dataset and stacker.  Its
    masks are replaced immediately by the explicit five-step logical masks.
    """
    from vjepa_policy.train_common import build_dataset_collator

    builder_args = copy.copy(args)
    if experiment == WAN22_STRIDE2_T5:
        builder_args.tubelet_size = 1
    dataset, collator = build_dataset_collator(builder_args, action_chunk_size)
    num_views = len(_camera_keys(args))
    if experiment == DINOV2_NATIVE16:
        mask = build_dinov2_native16_policy_mask(
            num_views=num_views,
            image_size=DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE,
        )
    elif experiment == WAN22_STRIDE2_T5:
        mask = build_wan22_stride2_policy_mask(
            image_size=_model_image_size(args),
            num_views=num_views,
        )
    else:
        raise ValueError(f"unsupported geometry experiment {experiment!r}")
    collator._ctx_idx = mask.ctx_idx
    collator._tgt_idx = mask.tgt_idx
    collator.n_ctx = mask.n_ctx
    collator.geometry_experiment = experiment
    return dataset, _MutableGeometryBatchCollator(collator)


__all__ = [
    "DINOV2_NATIVE16",
    "GEOMETRY_EXPERIMENTS",
    "WAN22_STRIDE2_T5",
    "build_geometry_dataset_collator",
    "build_geometry_encoder",
    "build_geometry_policy_for_experiment",
    "build_geometry_predictor",
    "build_wan22_stride2_video_offsets",
    "validate_geometry_config",
]


def build_wan22_stride2_video_offsets(*, num_frames: int = 33):
    """Compatibility re-export for launch/tests without importing the adapter."""
    from vjepa_policy.experimental.latent_space.wan22_stride2_encoder import (
        make_wan22_stride2_video_offsets,
    )

    return make_wan22_stride2_video_offsets(num_frames=num_frames)
