import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.gr1.compute_dataset_stats import (
    GR1_ACTION_STATE_INDICES,
    compute_stats,
)
from vjepa_policy.datasets.lerobot_video import (
    LeRobotClipDataset,
    _make_relative_action,
    _validate_feature_indices,
    read_instructions,
)
from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.text_embeddings import read_prompts


GR1_ARMS_HANDS_WAIST = (
    tuple(range(7))
    + tuple(range(22, 29))
    + tuple(range(7, 13))
    + tuple(range(29, 35))
    + tuple(range(41, 44))
)


def _metadata(offset, q01, q99, count=10):
    stats = {}
    for key in ("action", "observation.state"):
        stats[key] = {
            "min": np.arange(44, dtype=np.float32) + offset - 1,
            "max": np.arange(44, dtype=np.float32) + offset + 1,
            "mean": np.arange(44, dtype=np.float32) + offset,
            "std": np.ones(44, dtype=np.float32),
            "q01": np.full(44, q01, dtype=np.float32),
            "q99": np.full(44, q99, dtype=np.float32),
        }
    return SimpleNamespace(
        features={
            "action": {"dtype": "object", "shape": [44]},
            "observation.state": {"dtype": "object", "shape": [44]},
        },
        stats=stats,
        total_frames=count,
    )


def test_gr1_feature_selection_order_and_validation():
    selected = _validate_feature_indices(GR1_ARMS_HANDS_WAIST, 44, "action")
    assert len(selected) == 29
    assert torch.arange(44).index_select(-1, torch.tensor(selected)).tolist() == [
        *range(7),
        *range(22, 29),
        *range(7, 13),
        *range(29, 35),
        *range(41, 44),
    ]
    with pytest.raises(ValueError, match="duplicates"):
        _validate_feature_indices([0, 0], 44, "action")
    with pytest.raises(ValueError, match="exceed"):
        _validate_feature_indices([44], 44, "action")


def test_projected_quantile_stats_and_round_trip():
    normalizer = ActionStateNormalizer.from_metadata(
        [_metadata(0, q01=-2, q99=2), _metadata(1, q01=-3, q99=4)],
        action_key="action",
        state_key="observation.state",
        action_mode="QUANTILE",
        state_mode="QUANTILE",
        action_indices=GR1_ARMS_HANDS_WAIST,
        state_indices=GR1_ARMS_HANDS_WAIST,
    )
    assert normalizer.features["action"].shape == (29,)
    assert normalizer.stats["action"]["q01"].shape == (29,)
    np.testing.assert_array_equal(normalizer.stats["action"]["q01"], -3)
    np.testing.assert_array_equal(normalizer.stats["action"]["q99"], 4)

    action = torch.linspace(-3, 4, 29).repeat(2, 1)
    normalized = normalizer({"action": action})["action"]
    restored = normalizer.unnormalize_action(normalized)
    torch.testing.assert_close(restored, action)


def test_constant_quantile_dimension_maps_to_zero_and_round_trips():
    features = ActionStateNormalizer.from_metadata(
        [_metadata(0, q01=2, q99=2)],
        action_key="action",
        state_key="observation.state",
        action_mode="QUANTILE",
        state_mode="QUANTILE",
        action_indices=(0,),
        state_indices=(0,),
    )
    raw = torch.tensor([[2.0]])
    normalized = features({"action": raw})["action"]
    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
    torch.testing.assert_close(features.unnormalize_action(normalized), raw)
    predictions = torch.tensor([[-1.0], [0.75], [1.0]])
    torch.testing.assert_close(
        features.unnormalize_action(predictions),
        torch.full_like(predictions, 2.0),
    )


def test_gr1_relative_action_uses_current_state_for_arms_and_hands_only():
    state = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    action = state.expand(32, -1).clone() + 2
    relative = _make_relative_action(action, state, torch.arange(26))
    torch.testing.assert_close(relative[:, :26], torch.full((32, 26), 2.0))
    torch.testing.assert_close(relative[:, 26:], action[:, 26:])


def test_clipped_quantile_normalization_saturates_and_decodes_to_bounds():
    normalizer = ActionStateNormalizer.from_metadata(
        [_metadata(0, q01=-2, q99=2)],
        action_key="action",
        state_key="observation.state",
        action_mode="QUANTILE",
        state_mode="QUANTILE",
        action_indices=(0,),
        state_indices=(0,),
        clip_values=True,
    )
    normalized = normalizer({"action": torch.tensor([[-4.0], [4.0]])})["action"]
    torch.testing.assert_close(normalized, torch.tensor([[-1.0], [1.0]]))
    restored = normalizer.unnormalize_action(torch.tensor([[-3.0], [3.0]]))
    torch.testing.assert_close(restored, torch.tensor([[-2.0], [2.0]]))


def test_normalizer_preserves_codec_metadata_and_clipping(tmp_path):
    normalizer = ActionStateNormalizer.from_metadata(
        [_metadata(0, q01=-2, q99=2)],
        action_key="action",
        state_key="observation.state",
        action_mode="QUANTILE",
        state_mode="QUANTILE",
        action_indices=(0,),
        state_indices=(0,),
        clip_values=True,
    )
    normalizer.metadata = {"action_codec": {"action_chunk_size": 32}}
    path = tmp_path / "stats.json"
    normalizer.save(path)
    restored = ActionStateNormalizer.load(path)
    assert restored.clip_values is True
    assert restored.metadata == normalizer.metadata


def test_episode_remarks_are_used_for_prompt_cache(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    records = [
        {"episode_index": 0, "remarks": "pick up the cup"},
        {"episode_index": 1, "remarks": "pick up the cup"},
        {"episode_index": 2, "remarks": "close the drawer"},
    ]
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    assert read_instructions(tmp_path, "remarks") == {
        0: "pick up the cup",
        1: "pick up the cup",
        2: "close the drawer",
    }
    prompts = read_prompts([tmp_path], "remarks")
    assert len(prompts) == 2
    assert any("pick up the cup" in prompt for prompt in prompts)


def test_missing_episode_remarks_fail_explicitly(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "remarks": ""}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No non-empty"):
        read_instructions(tmp_path, "remarks")


def test_gr1_stats_use_global_absolute_min_max_and_identity_state(tmp_path, monkeypatch):
    task_bounds = ((-2.0, 3.0), (-4.0, 2.0))
    for index, (minimum, maximum) in enumerate(task_bounds):
        meta = tmp_path / f"task_{index}" / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{}", encoding="utf-8")
        stats = {
            "action": {
                "min": (np.arange(44, dtype=np.float32) + minimum).tolist(),
                "max": (np.arange(44, dtype=np.float32) + maximum).tolist(),
            }
        }
        (meta / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    manifest = {"sha256": "test-dataset"}
    monkeypatch.setattr(
        "scripts.gr1.compute_dataset_stats.build_dataset_manifest",
        lambda *args, **kwargs: manifest,
    )
    output = tmp_path / "dataset_stats.json"
    compute_stats(
        tmp_path,
        output,
        action_chunk_size=16,
        expected_tasks=2,
        expected_episodes=1,
    )

    normalizer = ActionStateNormalizer.load(output)
    selected = np.asarray(GR1_ACTION_STATE_INDICES)
    np.testing.assert_array_equal(normalizer.stats["action"]["min"], selected - 4)
    np.testing.assert_array_equal(normalizer.stats["action"]["max"], selected + 3)
    assert normalizer.action_mode == "MIN_MAX"
    assert normalizer.state_mode == "IDENTITY"
    assert normalizer.metadata == {
        "dataset": manifest,
        "action_codec": {
            "action_indices": list(GR1_ACTION_STATE_INDICES),
            "state_indices": list(GR1_ACTION_STATE_INDICES),
            "relative_action_indices": [],
            "action_chunk_size": 16,
        },
        "proprio_encoding": {"type": "sincos", "raw_dim": 29, "encoded_dim": 58},
    }


def test_full_future_window_maps_compact_indices_without_dropping_episode_starts():
    dataset = object.__new__(LeRobotClipDataset)
    dataset._video_offsets = (-4, 0, 4, 8, 12, 16, 20, 24, 28, 32)
    dataset._action_offsets = tuple(range(32))
    dataset._state_offsets = (0,)
    dataset.episode_data_index = {
        "from": torch.tensor([0, 40, 104]),
        "to": torch.tensor([40, 104, 124]),
    }
    dataset._build_full_window_index()

    assert len(dataset) == 40
    assert dataset._full_window_cumulative_sizes == [8, 40, 40]
    assert dataset._frame_index(0) == 0
    assert dataset._frame_index(7) == 7
    assert dataset._frame_index(8) == 40
    assert dataset._frame_index(39) == 71
    assert dataset._frame_index(-1) == 71
    with pytest.raises(IndexError):
        dataset._frame_index(40)


def test_full_future_window_is_disabled_by_default():
    dataset = object.__new__(LeRobotClipDataset)
    dataset.hf_dataset = list(range(5))
    dataset._full_window_cumulative_sizes = None

    assert len(dataset) == 5
    assert dataset._frame_index(4) == 4
