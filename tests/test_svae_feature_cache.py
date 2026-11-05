"""Real file-cache lifetime checks, without GPU work or changing feature values."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from scripts.svae_train.feature_cache import release_feature_cache
from scripts.svae_train.train_svae import _ShardedFeatureDataset


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("fincore"), reason="Linux file cache inspection required"
)
ROOT = Path(__file__).resolve().parents[1]


def resident(path):
    return int(subprocess.check_output(
        ["fincore", "--bytes", "--noheadings", "--output", "RES", str(path)], text=True
    ).strip())


def payload():
    return {"features": torch.arange(128).reshape(4, 8, 1, 2, 2).half(), "raw_dim": 8,
            "encoder_name": "test", "padding": torch.zeros(1 << 20)}


def test_release_preserves_feature_values_and_metadata(tmp_path):
    data = payload()
    path = tmp_path / "features_rank0_part00000.pt"
    torch.save(data, path)
    release_feature_cache(tmp_path)
    assert resident(path) == 0
    loaded = torch.load(path, weights_only=True)
    assert loaded.keys() == data.keys()
    assert loaded["raw_dim"] == 8 and loaded["encoder_name"] == "test"
    assert torch.equal(loaded["features"], data["features"])
    assert torch.equal(loaded["padding"], data["padding"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sample_read_releases_mmap_and_cache(tmp_path, dtype):
    data = payload()
    data["features"] = data["features"].to(dtype)
    path = tmp_path / "features_rank0_part00000.pt"
    torch.save(data, path)
    before_size = path.stat().st_size
    # Production shards are completed writes. Flush the freshly-created test
    # file before checking DONTNEED: Linux intentionally retains dirty pages.
    release_feature_cache(tmp_path)
    dataset = _ShardedFeatureDataset(tmp_path)
    sample = dataset[2]
    assert sample.dtype == torch.float32
    assert torch.equal(sample, data["features"][2].float())
    assert str(path) not in Path("/proc/self/maps").read_text()
    assert resident(path) == 0
    assert all(isinstance(part, str) for part in dataset._parts)
    dataset.close()
    assert path.stat().st_size == before_size
    # A copied fp32 sample must remain valid after closing the dataset.
    assert torch.equal(sample, data["features"][2].float())


def test_external_live_reader_still_skipped(tmp_path):
    path = tmp_path / "features_rank0_part00000.pt"
    torch.save(payload(), path)
    mapped = torch.load(path, mmap=True, weights_only=True)
    assert mapped["features"].sum() > 0
    result = release_feature_cache(tmp_path)
    assert result["skipped_in_use"] == 1 and result["released"] == 0
    del mapped
    assert release_feature_cache(tmp_path)["released"] == 1
    assert resident(path) == 0


def test_random_reads_match_original_shards(tmp_path):
    expected = []
    for shard, count in enumerate((3, 2, 4)):
        features = torch.randn(count, 8, 5, 2, 2).half()
        torch.save({"features": features, "raw_dim": 8, "use_avg_t_pooling": False},
                   tmp_path / f"features_rank0_part{shard:05d}.pt")
        expected.extend(features.float())
    dataset = _ShardedFeatureDataset(tmp_path)
    assert len(dataset) == len(expected)
    assert dataset.use_avg_t_pooling is False
    for i in [8, 2, 3, 0, 4, 7, 1, 6, 5, 8, 0]:
        assert torch.equal(dataset[i], expected[i])
    dataset.close()


def test_unrelated_files_are_not_evicted(tmp_path):
    path = tmp_path / "features_rank0_part00000.pt"
    other = tmp_path / "checkpoint.pt"
    torch.save(payload(), path)
    torch.save(payload(), other)
    before = resident(other)
    assert before > 0
    result = release_feature_cache(tmp_path)
    assert result["released"] == 1 and resident(path) == 0
    assert resident(other) == before and other.exists()


def test_release_error_is_reported_without_deleting_data(tmp_path, monkeypatch):
    path = tmp_path / "features_rank0_part00000.pt"
    torch.save(payload(), path)
    def fail(*args):
        raise OSError("test advice failure")
    monkeypatch.setattr(os, "posix_fadvise", fail)
    result = release_feature_cache(tmp_path)
    assert result["released"] == 0 and len(result["errors"]) == 1
    assert path.exists()


def test_failed_torchrun_cleanup_preserves_exit_status(tmp_path):
    path = tmp_path / "features_rank0_part00000.pt"
    torch.save(payload(), path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    launch = fakebin / "torchrun"
    launch.write_text("#!/bin/sh\nexit 7\n")
    launch.chmod(0o755)
    (fakebin / "python").symlink_to(sys.executable)
    env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"], "NPROC_PER_NODE": "1"}
    result = subprocess.run(["bash", str(ROOT / "scripts/svae_train/train_svae.sh"),
                             f"train.features_dir={tmp_path}"], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stdout + result.stderr
    assert resident(path) == 0
    assert path.exists()
