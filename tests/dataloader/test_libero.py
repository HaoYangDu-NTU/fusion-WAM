from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.libero import OUTPUT_REPRESENTATION, LiberoDataset
from openwam.dataloader.registry import DATASET_REGISTRY


def test_canonical_config_and_registry_use_native_action_reader() -> None:
    config = yaml.safe_load(Path("configs/dataloader/libero.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "libero"
    assert config["action_mode"] == "eef"
    assert DATASET_REGISTRY["libero"] is LiberoDataset


def test_reader_hard_preserves_rot6d_under_custom_stats() -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    # Deliberately non-identity rotation stats: the reader must still leave
    # rot6d untouched rather than relying only on the generated stats artifact.
    reader._normalization_stats = {
        "min": np.zeros(10, dtype=np.float32),
        "max": np.ones(10, dtype=np.float32),
    }
    raw = np.array(
        [[0.25, 0.5, 0.75, 0.998, 0.02, -0.04, 0.01, 0.999, 0.02, -0.5]],
        dtype=np.float32,
    )
    normalized = reader._normalize_array(raw)
    np.testing.assert_array_equal(normalized[:, 3:9], raw[:, 3:9])
    expected_non_rot = np.clip(raw[:, [0, 1, 2, 9]] * 2.0 - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(normalized[:, [0, 1, 2, 9]], expected_non_rot)


def test_reader_rejects_incomplete_compatibility_stats(tmp_path) -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    reader._source_stats_path = str(tmp_path / "normalization_stats.npy")
    reader._dataset_dir = tmp_path
    reader._raw_action_dim = 10
    np.save(reader._source_stats_path, {"eef": {}}, allow_pickle=True)
    with pytest.raises(KeyError, match="eef"):
        reader._load_stats({})


@pytest.fixture
def native_libero_bucket(tmp_path: Path) -> Path:
    (tmp_path / "meta" / "episodes").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    features = {"action": {"shape": [10]}, "observation.state": {"shape": [10]}}
    episode = {"episode_index": 0, "length": 40, "dataset_from_index": 0, "data/chunk_index": 0, "data/file_index": 0}
    for camera in LiberoDataset.DEFAULT_CAMERA_LAYOUT[:2]:
        features[camera] = {"dtype": "video"}
        episode[f"videos/{camera}/chunk_index"] = 0
        episode[f"videos/{camera}/file_index"] = 0
        video = tmp_path / "videos" / camera / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        video.touch()  # The decoder is replaced below; metadata/indexing use real files.
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 20, "features": features}))
    (tmp_path / "meta" / "conversion.json").write_text(json.dumps({"output_representation": OUTPUT_REPRESENTATION}))
    pd.DataFrame([episode]).to_parquet(tmp_path / "meta" / "episodes" / "chunk-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=["pick cup"]).to_parquet(tmp_path / "meta" / "tasks.parquet")
    eef = np.zeros((40, 10), dtype=np.float32)
    eef[:, 3:9] = [1, 0, 0, 0, 1, 0]
    pd.DataFrame({"action": list(eef), "observation.state": list(eef), "task_index": 0}).to_parquet(
        tmp_path / "data" / "chunk-000" / "file-000.parquet"
    )
    return tmp_path


@pytest.mark.parametrize(
    "overrides, expected_cameras",
    [
        ({}, ("observation.images.image", "observation.images.image2")),
        ({"target_camera": None}, ("observation.images.image", "observation.images.image2")),
        ({"camera_layout": None}, ("observation.images.image", "observation.images.image2")),
        (
            {"camera_layout": ["observation.images.image2", "observation.images.image", None]},
            ("observation.images.image2", "observation.images.image"),
        ),
        (
            {"camera_layout": ["observation.images.image", None, None], "target_camera": "observation.images.image2"},
            ("observation.images.image",),
        ),
        ({"multiview": False, "height": 256}, ("observation.images.image",)),
        (
            {"multiview": False, "height": 256, "target_camera": "observation.images.image2"},
            ("observation.images.image2",),
        ),
        ({"multiview": False, "height": 256, "target_camera": None}, ("observation.images.image",)),
    ],
    ids=[
        "official-default",
        "null-target",
        "default-layout",
        "custom-layout",
        "empty-wrist-slot",
        "single-head",
        "single-wrist",
        "single-fallback",
    ],
)
def test_camera_config_controls_decoded_views(native_libero_bucket, monkeypatch, overrides, expected_cameras):
    config = yaml.safe_load(Path("configs/dataloader/libero.yaml").read_text(encoding="utf-8"))
    config.update(dataset_dir=str(native_libero_bucket), normalize_mode=None)
    config.update(overrides)
    colors = {"observation.images.image": (255, 0, 0), "observation.images.image2": (0, 255, 0)}
    decoded = []

    def decode(path, frame_indices, h, w):
        camera = Path(path).parent.parent.name
        decoded.append((camera, tuple(frame_indices)))
        return [Image.new("RGB", (w, h), colors[camera]) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    reader = LiberoDataset.from_config(OmegaConf.create(config), split="train")
    sample = reader[0]

    assert tuple(camera for camera, _ in decoded) == expected_cameras
    assert all(indices == tuple(range(0, 33, 4)) for _, indices in decoded)
    assert reader._fail_count == reader._wrist_fail_count == 0
    assert len(sample["video"]) == 9
    assert sample["prompt"] == "pick cup"
    for frame in sample["video"]:
        assert frame.size == (320, config["height"])
        assert frame.getpixel((10, 10)) == colors[expected_cameras[0]]
        if config["multiview"]:
            wrist_color = colors[expected_cameras[1]] if len(expected_cameras) > 1 else (0, 0, 0)
            assert frame.getpixel((10, 300)) == wrist_color
            assert frame.getpixel((250, 300)) == (0, 0, 0)
    np.testing.assert_array_equal(np.asarray(sample["clean_condition_images"][0]), np.asarray(sample["video"][0]))
