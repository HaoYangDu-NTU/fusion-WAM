"""Opt-in test using a complete record from the beginning of an official DROID shard."""
import json
import os
from pathlib import Path
import shutil
import struct

import pytest

from .droid import batches, samples, statistics, training_batch


@pytest.mark.skipif(not os.environ.get("MOLMO_DROID_PROBE"), reason="Official DROID probe unavailable")
def test_decode_official_record(tmp_path):
    import tensorflow_datasets as tfds

    probe = Path(os.environ["MOLMO_DROID_PROBE"])
    for name in ("features.json", "dataset_info.json"):
        shutil.copyfile(probe / f"droid-{name}", tmp_path / name)
    builder = tfds.builder_from_directory(str(tmp_path))
    assert (builder.info.name, str(builder.info.version)) == ("droid_101", "0.0.1")
    with (probe / "official-shard-prefix.tfrecord").open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        stream.read(4)
        payload = stream.read(length)
    assert len(payload) == length
    trajectory = tfds.as_numpy(builder.info.features.deserialize_example(payload))
    meta = trajectory["episode_metadata"]
    path = meta["file_path"].decode()
    episode = meta["recording_folderpath"].decode() + "--" + path
    ranges = json.loads((probe.parent / "droid_sample_ranges_v1_0_1.json").read_text())
    assert "/success/" in path
    rows = [{**x, "episode_id": episode} for x in samples(list(trajectory["steps"]), ranges[episode], 8)]
    assert len(rows) >= 2
    batch = training_batch(next(batches(rows, 2)), statistics(rows))
    assert batch["state"].shape == (2, 8)
    assert batch["actions"].shape == (2, 8, 8)
    assert all(image.shape[-1] == 3 and image.ndim == 4 for image in batch["images"].values())
    print(f"DROID official record: {len(rows)} retained anchors, state {batch['state'].shape}, "
          f"actions {batch['actions'].shape}, views {[x.shape for x in batch['images'].values()]}")
