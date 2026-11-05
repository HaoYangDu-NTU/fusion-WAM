from __future__ import annotations

from dataclasses import replace

import pytest

from vjepa_policy.experimental.latent_space.geometry_builders import (
    DINOV2_NATIVE16,
    WAN22_STRIDE2_T5,
)
from vjepa_policy.experimental.latent_space.geometry_serving import (
    DINOV2_NATIVE16_SERVING_FAMILY,
    DINOV2_NATIVE16_SERVING_MODEL,
    WAN22_STRIDE2_SERVING_FAMILY,
    WAN22_STRIDE2_SERVING_MODEL,
    geometry_experiment_from_encoder,
    geometry_serving_config,
)
from vjepa_policy.policy_serving.libero import PolicyServingConfig


def make_config(**overrides):
    config = PolicyServingConfig(
        crop_size=224,
        view_layout="independent",
        camera_keys=("main", "wrist"),
    )
    return replace(config, **overrides)


@pytest.mark.parametrize(
    "family,model,experiment",
    [
        (
            DINOV2_NATIVE16_SERVING_FAMILY,
            DINOV2_NATIVE16_SERVING_MODEL,
            DINOV2_NATIVE16,
        ),
        (
            WAN22_STRIDE2_SERVING_FAMILY,
            WAN22_STRIDE2_SERVING_MODEL,
            WAN22_STRIDE2_T5,
        ),
    ],
)
def test_geometry_serving_encoder_identity_is_explicit(family, model, experiment):
    assert geometry_experiment_from_encoder(family, model) == experiment


def test_dinov2_serving_separates_224_input_from_256_predictor_grid():
    physical = make_config(num_frames=10)
    logical = geometry_serving_config(physical, DINOV2_NATIVE16)

    assert physical.crop_size == 224
    assert logical.crop_size == 256
    assert logical.num_frames == 10
    assert logical.patch_size == 16


def test_wan_serving_adds_one_discarded_history_frame_before_current_window():
    physical = make_config(num_frames=17)
    logical = geometry_serving_config(physical, WAN22_STRIDE2_T5)

    assert physical.num_frames == 17
    assert logical.crop_size == 224
    assert logical.num_frames == 18


@pytest.mark.parametrize(
    "config",
    [
        make_config(crop_size=256),
        make_config(view_layout="quadrant"),
        make_config(camera_keys=("main",)),
    ],
)
def test_geometry_serving_rejects_changed_physical_protocol(config):
    with pytest.raises(ValueError, match="latent-geometry LIBERO serving requires"):
        geometry_serving_config(config, DINOV2_NATIVE16)
