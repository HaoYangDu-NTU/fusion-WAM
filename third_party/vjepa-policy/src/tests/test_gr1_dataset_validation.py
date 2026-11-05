import json

import pytest

from scripts.gr1.validate_dataset import build_dataset_manifest


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_task(root, episodes=2):
    meta = root / "meta"
    meta.mkdir(parents=True)
    _write_json(
        meta / "info.json",
        {
            "total_episodes": episodes,
            "fps": 20.0,
            "features": {
                "observation.images.ego_view": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "video_info": {"video.fps": 20.0},
                },
                "observation.state": {"dtype": "object", "shape": [44]},
                "action": {"dtype": "object", "shape": [44]},
            },
        },
    )
    modality_ranges = {
        "left_arm": (0, 7),
        "right_arm": (22, 29),
        "left_hand": (7, 13),
        "right_hand": (29, 35),
        "waist": (41, 44),
    }
    _write_json(
        meta / "modality.json",
        {
            section: {
                group: {
                    "original_key": original_key,
                    "start": start,
                    "end": end,
                }
                for group, (start, end) in modality_ranges.items()
            }
            for section, original_key in (
                ("state", "observation.state"),
                ("action", "action"),
            )
        }
        | {"video": {"ego_view": {"original_key": "observation.images.ego_view"}}},
    )
    for name in ("embodiment.json", "metadata.json", "stats.json"):
        _write_json(meta / name, {})
    (meta / "initial_actions.npz").write_bytes(b"npz")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0}) + "\n")
    records = [
        {"episode_index": index, "remarks": f"instruction {index}"}
        for index in range(episodes)
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    for index in range(episodes):
        parquet = root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        video = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{index:06d}.mp4"
        )
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(f"parquet-{index}".encode())
        video.write_bytes(f"video-{index}".encode())


def test_gr1_manifest_validates_complete_task_and_tracks_file_changes(tmp_path):
    root = tmp_path / "task"
    _make_task(root)

    first = build_dataset_manifest(tmp_path, expected_tasks=1, expected_episodes=2)
    assert first["task_roots"] == ["task"]
    assert first["episodes_per_task"] == 2

    video = root / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4"
    video.write_bytes(b"changed-video-size")
    second = build_dataset_manifest(tmp_path, expected_tasks=1, expected_episodes=2)
    assert first["sha256"] != second["sha256"]


def test_gr1_manifest_rejects_incomplete_video_download(tmp_path):
    root = tmp_path / "task"
    _make_task(root)
    (root / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4").unlink()

    with pytest.raises(ValueError, match="Expected 2 video files"):
        build_dataset_manifest(tmp_path, expected_tasks=1, expected_episodes=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda info: info.update(fps=30.0), "must use fps=20"),
        (
            lambda info: info["features"]["action"].update(shape=[29]),
            "feature 'action' must have shape \\[44\\]",
        ),
        (
            lambda info: info["features"]["observation.images.ego_view"].update(
                shape=[224, 224, 3]
            ),
            "observation.images.ego_view.*shape \\[256, 256, 3\\]",
        ),
    ],
)
def test_gr1_manifest_rejects_incompatible_info_schema(tmp_path, mutation, message):
    root = tmp_path / "task"
    _make_task(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    mutation(info)
    _write_json(info_path, info)

    with pytest.raises(ValueError, match=message):
        build_dataset_manifest(tmp_path, expected_tasks=1, expected_episodes=2)


def test_gr1_manifest_rejects_incompatible_codec_range(tmp_path):
    root = tmp_path / "task"
    _make_task(root)
    modality_path = root / "meta/modality.json"
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    modality["action"]["right_arm"]["start"] = 21
    _write_json(modality_path, modality)

    with pytest.raises(ValueError, match="modality action.right_arm"):
        build_dataset_manifest(tmp_path, expected_tasks=1, expected_episodes=2)
