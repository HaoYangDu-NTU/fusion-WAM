from __future__ import annotations

from types import SimpleNamespace

import torch

import vjepa_policy.policy_serving.libero as serving_module
from vjepa_policy.datasets import VideoClipTransform
from vjepa_policy.policy_serving.libero import (
    PolicyServingConfig,
    VJEPAPolicyServing,
)


class IdentityNormalizer:
    def __call__(self, batch):
        return batch

    def unnormalize_action(self, actions):
        return actions


class FakePredictor:
    def __init__(self):
        self.mask_shapes = None
        self.conditioning = None

    def __call__(self, context, m_enc, m_pred, **kwargs):
        self.mask_shapes = (tuple(m_enc.shape), tuple(m_pred.shape))
        self.conditioning = kwargs
        batch_size = context.shape[0]
        key = torch.zeros(batch_size, 1, 2, 2)
        return None, ((key, key),)


class FakeActionExpert:
    def __init__(self):
        self.state_shape = None

    def predict_action(self, predictor_kv, language, language_mask, state, context_len):
        self.state_shape = tuple(state.shape)
        return torch.zeros(state.shape[0], 4, 7)


def test_checkpoint_serving_config_prefers_saved_values_then_overrides(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "train_precision": "bf16",
            "model_topology": {
                "encoder_family": "dinov2",
                "predictor_image_height": 224,
                "view_layout": "independent",
                "camera_keys": ["main", "wrist"],
                "action_chunk_size": 32,
                "predictor_rope_frequency_pairing": "corrected",
            },
            "policy_serving": {
                "config": {
                    "num_frames": 10,
                    "t5_len": 48,
                    "action_num_inference_steps": 20,
                }
            },
        },
        path,
    )

    config = PolicyServingConfig.from_checkpoint(path, overrides={"t5_len": 64})

    assert config.precision == "bf16"
    assert config.encoder_family == "dinov2"
    assert config.camera_keys == ("main", "wrist")
    assert config.action_chunk_size == 32
    assert config.corrected_predictor_rope is True
    assert config.num_frames == 10
    assert config.action_num_inference_steps == 20
    assert config.t5_len == 64


def test_horizontal_serving_preparation_keeps_full_view_resolution():
    policy = VJEPAPolicyServing.__new__(VJEPAPolicyServing)
    policy.camera_keys = ("main", "wrist")
    policy.num_frames = 4
    policy.video_transform = VideoClipTransform((8, 8))
    policy.view_combiner = "horizontal"
    policy._load_language = lambda task: (
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
    )
    image = torch.zeros(1, 3, 8, 8)
    obs = {
        "main": image,
        "main_past": image,
        "wrist": image,
        "wrist_past": image,
        "observation.state": torch.zeros(1, 8),
        "task": ["task"],
    }

    clip, _, _, state = policy._prepare_observation(obs)

    assert clip.shape == (3, 4, 8, 16)
    assert state.shape == (1, 8)


def test_infer_batch_expands_masks_for_token_cross_attention(monkeypatch):
    policy = VJEPAPolicyServing.__new__(VJEPAPolicyServing)
    policy.device = torch.device("cpu")
    policy.precision = "fp32"
    policy.dtype = torch.float32
    policy.proprio_encoding = "identity"
    policy.context_len = 2
    policy.m_enc = torch.tensor([[0, 1]])
    policy.m_pred = torch.tensor([[2, 3]])
    policy.encoder = object()
    policy.predictor = FakePredictor()
    policy.action_expert = FakeActionExpert()
    policy.action_normalizer = IdentityNormalizer()
    policy.binarize_gripper = True
    policy._prepare_observation = lambda obs: (
        torch.zeros(3, 4, 2, 2),
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 8),
    )
    monkeypatch.setattr(
        serving_module,
        "encode_video_views",
        lambda encoder, clips, mask: torch.zeros(clips.shape[0], mask.shape[1], 4),
    )

    results = policy.infer_batch([SimpleNamespace(), SimpleNamespace()])

    assert policy.predictor.mask_shapes == ((2, 2), (2, 2))
    assert set(policy.predictor.conditioning) == {
        "language",
        "language_mask",
        "proprio",
    }
    assert policy.predictor.conditioning["proprio"].shape == (2, 8)
    assert len(results) == 2
    assert results[0]["actions"].shape == (4, 7)
    assert policy.action_expert.state_shape == (2, 8)
