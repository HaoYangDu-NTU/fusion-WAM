import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vjepa_policy.policy_serving.gr1 import (
    GR1_ACTION_HORIZON,
    GR1_ACTION_STATE_INDICES,
    GR1_DATASET_VIDEO_KEY,
    GR1_GROUP_DIMS,
    GR1_LANGUAGE_KEY,
    GR1_VIDEO_KEY,
    VJEPAGR1Serving,
    build_gr1_model_clip,
    decode_gr1_action_chunk,
    flat_gr1_state_to_29d,
    gr1_language_batch,
    prepare_gr1_model_inputs,
    validate_gr1_wire_observation,
)


def _wire_observation(batch_size=2):
    video = np.zeros((batch_size, 2, 256, 256, 3), dtype=np.uint8)
    video[:, 1] = 255
    observation = {GR1_VIDEO_KEY: video}
    offset = 1.0
    for group_name, group_dim in GR1_GROUP_DIMS:
        values = np.arange(group_dim, dtype=np.float32) + offset
        observation[f"state.{group_name}"] = np.broadcast_to(
            values, (batch_size, 1, group_dim)
        ).copy()
        offset += 10.0
    observation[GR1_LANGUAGE_KEY] = tuple(
        "unlocked_waist: pick up the cup" for _ in range(batch_size)
    )
    return observation


def test_prepare_gr1_model_inputs_is_batched_and_repeats_only_current_frame():
    observation = _wire_observation(batch_size=2)

    inputs = prepare_gr1_model_inputs(observation)

    assert inputs.clip.shape == (2, 1, 3, 6, 224, 224)
    assert inputs.clip.dtype == torch.float32
    torch.testing.assert_close(
        inputs.clip[:, :, :, 0], torch.full((2, 1, 3, 224, 224), -1.0)
    )
    torch.testing.assert_close(
        inputs.clip[:, :, :, 1:], torch.ones(2, 1, 3, 5, 224, 224)
    )
    assert inputs.raw_state.shape == (2, 1, 29)
    assert inputs.instructions == ["pick up the cup", "pick up the cup"]


def test_clip_builder_rejects_noncanonical_wire_shape():
    with pytest.raises(ValueError, match="GR-1 video must be"):
        build_gr1_model_clip(np.zeros((1, 1, 256, 256, 3), dtype=np.uint8))


def test_clip_builder_supports_ck32_ten_frame_configuration():
    clip = build_gr1_model_clip(
        _wire_observation(batch_size=1)[GR1_VIDEO_KEY], model_frames=10
    )
    assert clip.shape == (1, 1, 3, 10, 224, 224)


def test_flat_state_and_language_use_official_order_and_prefix():
    observation = _wire_observation(batch_size=1)
    state = flat_gr1_state_to_29d(observation)

    assert state.shape == (1, 1, 29)
    np.testing.assert_array_equal(state[0, 0, :7], np.arange(7, dtype=np.float32) + 1)
    np.testing.assert_array_equal(
        state[0, 0, 7:14], np.arange(7, dtype=np.float32) + 11
    )
    assert gr1_language_batch(observation[GR1_LANGUAGE_KEY], 1) == ["pick up the cup"]

    with pytest.raises(ValueError, match="must start"):
        gr1_language_batch(["pick up the cup"], 1)
    with pytest.raises(TypeError, match="object dtype"):
        gr1_language_batch(np.array(["unlocked_waist: task"], dtype=object), 1)


def test_wire_validation_is_strict_about_state_dtype_and_batch_shape():
    observation = _wire_observation(batch_size=2)
    assert validate_gr1_wire_observation(observation) == 2

    observation["state.waist"] = observation["state.waist"].astype(np.float64)
    with pytest.raises(TypeError, match="state.waist must use float32"):
        validate_gr1_wire_observation(observation)


def test_decode_ck16_returns_absolute_action_without_using_raw_state():
    absolute = torch.full((2, GR1_ACTION_HORIZON, 29), 2.0)
    absolute[..., 26:] = torch.tensor([101.0, 102.0, 103.0])

    action = decode_gr1_action_chunk(absolute)

    assert action["action.left_arm"].shape == (2, 16, 7)
    torch.testing.assert_close(action["action.left_arm"], torch.full((2, 16, 7), 2.0))
    torch.testing.assert_close(
        action["action.waist"],
        torch.tensor([101.0, 102.0, 103.0]).expand(2, 16, 3),
    )


class _ClippingNormalizer:
    features = {
        "action": SimpleNamespace(shape=(29,)),
        "observation.state": SimpleNamespace(shape=(29,)),
    }
    metadata = {
        "action_codec": {
            "action_indices": list(GR1_ACTION_STATE_INDICES),
            "state_indices": list(GR1_ACTION_STATE_INDICES),
            "relative_action_indices": [],
            "action_chunk_size": GR1_ACTION_HORIZON,
        },
        "proprio_encoding": {
            "type": "sincos",
            "raw_dim": 29,
            "encoded_dim": 58,
        },
    }
    action_mode = "MIN_MAX"
    state_mode = "IDENTITY"
    clip_values = True

    def __call__(self, batch):
        return batch

    def unnormalize_action(self, action):
        return action.clamp(-1.0, 1.0) * 10.0


class _FakeEncoder:
    patch_size = 16
    tubelet_size = 2

    def __init__(self):
        self.mask_shape = None

    def __call__(self, inputs, selection):
        self.mask_shape = selection.shape
        return torch.zeros(inputs.shape[0], selection.shape[1], 8)


class _FakePredictor:
    def __init__(self):
        self.mask_shapes = None

    def __call__(self, context, masks_enc, masks_pred, **kwargs):
        del kwargs
        self.mask_shapes = (masks_enc.shape, masks_pred.shape)
        prediction = torch.zeros(
            context.shape[0], masks_pred.shape[1], context.shape[-1]
        )
        return prediction, [(torch.zeros(1), torch.zeros(1))]


class _FakeActionExpert:
    action_dim = 29
    action_chunk_size = GR1_ACTION_HORIZON

    def __init__(self):
        self.state_shape = None

    def predict_action(self, predictor_kv, language, language_mask, state, context_len):
        del predictor_kv, language_mask, context_len
        self.state_shape = state.shape
        return torch.full(
            (language.shape[0], self.action_chunk_size, self.action_dim), 2.0
        )


class _FakeCore:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.precision = "fp32"
        self.view_layout = "independent"
        self.camera_keys = (GR1_DATASET_VIDEO_KEY,)
        self.crop_size = 224
        self.video_transform = None
        self.encoder = _FakeEncoder()
        self.predictor = _FakePredictor()
        self.action_expert = _FakeActionExpert()
        self.action_normalizer = _ClippingNormalizer()
        self.num_frames = 6
        self.proprio_encoding = "sincos"
        self.m_enc = torch.arange(196).unsqueeze(0)
        self.m_pred = torch.arange(196, 3 * 196).unsqueeze(0)
        self.context_len = 196

    def _load_language(self, text):
        del text
        return torch.zeros(1, 32, 4), torch.ones(1, 32, dtype=torch.bool)


def test_batched_runtime_expands_predictor_masks_to_every_environment():
    runtime = VJEPAGR1Serving.__new__(VJEPAGR1Serving)
    runtime.core = _FakeCore()
    runtime._validate_core()

    action = runtime.infer_flat_observation(_wire_observation(batch_size=2))

    assert runtime.core.encoder.mask_shape == (2, 196)
    assert runtime.core.predictor.mask_shapes == ((2, 196), (2, 392))
    assert runtime.core.action_expert.state_shape == (2, 58)
    assert action["action.left_arm"].shape == (2, 16, 7)
    np.testing.assert_array_equal(action["action.left_arm"], 10.0)


class _FakeRuntime:
    action_horizon = GR1_ACTION_HORIZON

    def infer_flat_observation(self, observation):
        batch_size = validate_gr1_wire_observation(observation)
        return {
            f"action.{name}": np.zeros(
                (batch_size, GR1_ACTION_HORIZON, width), dtype=np.float32
            )
            for name, width in GR1_GROUP_DIMS
        }


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_official_policy_server_roundtrip_with_batched_fake_runtime():
    pytest.importorskip("gr00t")
    from examples.gr1.serve_policy import VJEPAGR1Policy
    from gr00t.data.types import ModalityConfig
    from gr00t.policy import BasePolicy
    from gr00t.policy.server_client import PolicyClient, PolicyServer

    policy = VJEPAGR1Policy(_FakeRuntime())
    assert isinstance(policy, BasePolicy)
    port = _free_port()
    with PolicyServer(policy, host="127.0.0.1", port=port) as server:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        with PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000) as client:
            modality = client.get_modality_config()
            assert isinstance(modality["video"], ModalityConfig)
            assert modality["video"].delta_indices == [-4, 0]
            assert modality["video"].modality_keys == [
                "ego_view_bg_crop_pad_res256_freq20"
            ]
            assert modality["action"].delta_indices == list(range(16))

            action, info = client.get_action(_wire_observation(batch_size=2))
            assert info == {}
            assert action["action.left_arm"].shape == (2, 16, 7)
            assert action["action.waist"].shape == (2, 16, 3)
            client.kill_server()
        thread.join(timeout=2)
        assert not thread.is_alive()
