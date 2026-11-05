import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.gr1.evaluate_policy import (
    GR1_EXECUTION_HORIZON,
    GR1_TASKS,
    GR1_WIRE_VIDEO_KEY,
    _truncate_episode_results,
    env_id,
    main,
    parse_args,
)


def test_gr1_evaluation_lists_all_official_tasks_and_executes_sixteen_steps():
    assert len(GR1_TASKS) == 24
    assert len(set(GR1_TASKS)) == 24
    assert GR1_EXECUTION_HORIZON == 16
    assert env_id("PnPBottleToCabinetClose") == (
        "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    )


def test_rollout_overshoot_is_truncated_with_aligned_episode_info():
    successes, episode_info = _truncate_episode_results(
        [True, False, True, False],
        {
            "episode_lengths": [10, 20, 30, 40],
            "episode_rewards": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        n_episodes=3,
    )

    assert successes == [True, False, True]
    assert episode_info["episode_lengths"] == [10, 20, 30]
    np.testing.assert_array_equal(episode_info["episode_rewards"], [1.0, 2.0, 3.0])


def test_rollout_result_alignment_is_validated():
    with pytest.raises(ValueError, match="expected at least 3"):
        _truncate_episode_results([True, False], {}, n_episodes=3)
    with pytest.raises(ValueError, match="episode_lengths.*1 entries"):
        _truncate_episode_results(
            [True, False], {"episode_lengths": [10]}, n_episodes=2
        )


def test_video_recording_is_opt_in(tmp_path):
    assert (
        parse_args(
            [
                "--output",
                str(tmp_path / "default.json"),
            ]
        ).video_dir
        is None
    )
    assert (
        parse_args(
            [
                "--video-dir",
                str(tmp_path / "videos"),
                "--output",
                str(tmp_path / "recorded.json"),
            ]
        ).video_dir
        == tmp_path / "videos"
    )


def test_execution_horizon_is_configurable(tmp_path):
    args = parse_args(
        [
            "--execution-horizon",
            "8",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    assert args.execution_horizon == 8


def test_main_truncates_overshoot_and_records_only_wire_view(monkeypatch, tmp_path):
    calls = []
    rollout_policy = SimpleNamespace(ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX={})

    def fake_rollout(**kwargs):
        calls.append(kwargs)
        return (
            kwargs["env_name"],
            [True, False],
            {"episode_lengths": [8, 16], "episode_rewards": [1.0, 0.0]},
        )

    rollout_policy.run_gr00t_sim_policy = fake_rollout
    gr00t = ModuleType("gr00t")
    gr00t_eval = ModuleType("gr00t.eval")
    gr00t_eval.rollout_policy = rollout_policy
    monkeypatch.setitem(sys.modules, "gr00t", gr00t)
    monkeypatch.setitem(sys.modules, "gr00t.eval", gr00t_eval)

    output = tmp_path / "result.json"
    video_dir = tmp_path / "videos"
    main(
        [
            "--task",
            "PnPBottleToCabinetClose",
            "--n-episodes",
            "1",
            "--n-envs",
            "1",
            "--output",
            str(output),
            "--video-dir",
            str(video_dir),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["tasks"][0]["successes"] == [True]
    assert report["tasks"][0]["episode_info"] == {
        "episode_lengths": [8],
        "episode_rewards": [1.0],
    }
    assert calls[0]["video_dir"] == str(video_dir / "PnPBottleToCabinetClose")
    assert calls[0]["n_action_steps"] == GR1_EXECUTION_HORIZON
    assert rollout_policy.ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX["gr1_unified"] == (
        GR1_WIRE_VIDEO_KEY,
    )
