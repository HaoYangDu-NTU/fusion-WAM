"""LIBERO serving adapter for the isolated latent-geometry experiments."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from vjepa_policy.datasets import VideoClipTransform
from vjepa_policy.experimental.latent_space.dino_native16 import (
    DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE,
    load_dinov2_native16_latent_encoder,
)
from vjepa_policy.experimental.latent_space.geometry_builders import (
    DINOV2_NATIVE16,
    WAN22_STRIDE2_T5,
    build_geometry_policy_for_experiment,
    build_geometry_predictor,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
)
from vjepa_policy.experimental.latent_space.wan22_stride2_encoder import (
    load_wan22_stride2_t5_latent_encoder,
)
from vjepa_policy.policy_serving.libero import (
    PolicyServingConfig,
    VJEPAPolicyServing,
)


DINOV2_NATIVE16_SERVING_FAMILY = "dinov2"
DINOV2_NATIVE16_SERVING_MODEL = "dinov2_vitl_native16"
WAN22_STRIDE2_SERVING_FAMILY = "wan2_2_stride2_t5"
WAN22_STRIDE2_SERVING_MODEL = "vae38"


def geometry_experiment_from_encoder(
    encoder_family: str,
    model_name: str,
) -> str:
    identity = (encoder_family, model_name)
    if identity == (
        DINOV2_NATIVE16_SERVING_FAMILY,
        DINOV2_NATIVE16_SERVING_MODEL,
    ):
        return DINOV2_NATIVE16
    if identity == (
        WAN22_STRIDE2_SERVING_FAMILY,
        WAN22_STRIDE2_SERVING_MODEL,
    ):
        return WAN22_STRIDE2_T5
    raise ValueError(
        "unsupported latent-geometry serving encoder "
        f"family={encoder_family!r} model={model_name!r}"
    )


def geometry_serving_config(
    config: PolicyServingConfig,
    experiment: str,
) -> PolicyServingConfig:
    """Separate the physical 224px input from the logical predictor lattice."""
    if config.crop_size != 224:
        raise ValueError("latent-geometry LIBERO serving requires crop_size=224")
    if config.view_layout != "independent" or len(config.camera_keys) != 2:
        raise ValueError(
            "latent-geometry LIBERO serving requires two independent camera views"
        )
    if experiment == DINOV2_NATIVE16:
        return replace(
            config,
            crop_size=DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE[0],
            num_frames=10,
            patch_size=16,
            tubelet_size=2,
        )
    if experiment == WAN22_STRIDE2_T5:
        # The base serving adapter creates [past, current, current, ...].  With
        # 18 frames, the Wan adapter drops the leading history frame and sends
        # exactly 17 current frames to the VAE, so logical t0 is current.
        return replace(
            config,
            crop_size=224,
            num_frames=18,
            patch_size=16,
            tubelet_size=2,
        )
    raise ValueError(f"unsupported geometry experiment {experiment!r}")


def build_geometry_serving_encoder(
    config: PolicyServingConfig,
    pretrained_encoder: str,
    *,
    experiment: str,
    image_microbatch_size: int | None,
    video_microbatch_size: int | None,
    fastwam_src_dir: str,
):
    observed_experiment = geometry_experiment_from_encoder(
        config.encoder_family,
        config.model_name,
    )
    if observed_experiment != experiment:
        raise ValueError(
            f"serving encoder resolves to {observed_experiment}, expected {experiment}"
        )
    if experiment == DINOV2_NATIVE16:
        return load_dinov2_native16_latent_encoder(
            pretrained_encoder,
            image_microbatch_size=image_microbatch_size,
        )
    if experiment == WAN22_STRIDE2_T5:
        return load_wan22_stride2_t5_latent_encoder(
            pretrained_encoder,
            fastwam_src_dir=fastwam_src_dir,
            video_microbatch_size=video_microbatch_size,
        )
    raise ValueError(f"unsupported geometry experiment {experiment!r}")


class GeometryVJEPAPolicyServing(VJEPAPolicyServing):
    """Build the trained logical geometry while retaining 224px observations."""

    def __init__(
        self,
        *,
        experiment: str,
        ckpt_path: str,
        pretrained_encoder: str,
        text_cache_dir: str,
        config: PolicyServingConfig,
        device: str = "cuda",
        image_microbatch_size: int | None = 32,
        video_microbatch_size: int | None = 4,
        fastwam_src_dir: str = str(DEFAULT_FASTWAM_SRC_DIR),
    ):
        physical_crop_size = config.crop_size
        logical_config = geometry_serving_config(config, experiment)
        super().__init__(
            ckpt_path=ckpt_path,
            pretrained_encoder=pretrained_encoder,
            text_cache_dir=text_cache_dir,
            config=logical_config,
            device=device,
            encoder_builder=partial(
                build_geometry_serving_encoder,
                experiment=experiment,
                image_microbatch_size=image_microbatch_size,
                video_microbatch_size=video_microbatch_size,
                fastwam_src_dir=fastwam_src_dir,
            ),
            predictor_builder=partial(
                build_geometry_predictor,
                experiment=experiment,
            ),
            policy_builder=partial(
                build_geometry_policy_for_experiment,
                experiment,
            ),
        )
        self.geometry_experiment = experiment
        self.crop_size = physical_crop_size
        self.model_image_size = (physical_crop_size, physical_crop_size)
        self.video_transform = VideoClipTransform(
            (physical_crop_size, physical_crop_size)
        )
        self.view_combiner = "independent"


__all__ = [
    "DINOV2_NATIVE16_SERVING_FAMILY",
    "DINOV2_NATIVE16_SERVING_MODEL",
    "GeometryVJEPAPolicyServing",
    "WAN22_STRIDE2_SERVING_FAMILY",
    "WAN22_STRIDE2_SERVING_MODEL",
    "build_geometry_serving_encoder",
    "geometry_experiment_from_encoder",
    "geometry_serving_config",
]
