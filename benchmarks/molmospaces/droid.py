"""DROID 1.0.1 RLDS inputs. This module has no simulator dependency."""
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from .provenance import hash_file

CAMERAS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
SOURCE = "gs://gresearch/robotics/droid/1.0.1"
SCHEMA = "droid.absolute-joint-position.v1"



def vector(fields):
    joints = np.asarray(fields["joint_position"], dtype=np.float32)
    grip = np.asarray(fields["gripper_position"], dtype=np.float32).reshape(-1)
    if joints.shape != (7,) or grip.shape != (1,):
        raise ValueError("Expected seven Franka joints and one DROID gripper value")
    value = np.concatenate((joints, grip))
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite DROID state or action")
    return value


def samples(steps, ranges, chunk):
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if not steps:
        return
    targets = np.stack([vector(step["action_dict"]) for step in steps])
    keep = np.zeros(len(steps), dtype=bool)
    for start, end in ranges:
        if not 0 <= start < end <= len(steps):
            raise ValueError(f"Invalid DROID range {(start, end)} for {len(steps)} frames")
        keep[start:end] = True
    for index in np.flatnonzero(keep):
        step = steps[index]
        prompt = next((step[key] for key in ("language_instruction", "language_instruction_2",
                                             "language_instruction_3") if step.get(key)), b"")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if not prompt.strip():
            continue
        offsets = index + np.arange(chunk)
        yield {
            "images": {key: step["observation"][key] for key in CAMERAS},
            "state": vector(step["observation"]),
            "language": prompt,
            "actions": targets[np.minimum(offsets, len(steps) - 1)],
            "action_valid": offsets < len(steps),
            "frame_index": int(index),
        }


def rlds_samples(root, filter_path, chunk):
    """Apply the OpenPI keep ranges to anchor frames; keep their original temporal targets."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    path = Path(root).resolve() / "droid" / "1.0.1"
    builder = tfds.builder_from_directory(str(path))
    if builder.info.name != "droid_101" or str(builder.info.version) != "0.0.1":
        raise ValueError("Only DROID 1.0.1 RLDS is supported")
    ranges = json.loads(Path(filter_path).read_text())
    for trajectory in tfds.as_numpy(builder.as_dataset(split="train", shuffle_files=False)):
        metadata = trajectory["episode_metadata"]
        file_path = metadata["file_path"].decode()
        folder = metadata["recording_folderpath"].decode()
        episode_id = f"{folder}--{file_path}"
        if "/success/" not in f"/{file_path.strip('/')}/" or episode_id not in ranges:
            continue
        for sample in samples(list(trajectory["steps"]), ranges[episode_id], chunk):
            yield {**sample, "episode_id": episode_id}


def batches(stream, size):
    if size < 1:
        raise ValueError("batch size must be positive")
    batch = []
    for sample in stream:
        batch.append(sample)
        if len(batch) == size:
            yield collate(batch)
            batch = []
    if batch:
        yield collate(batch)


def collate(batch):
    return {
        "images": {key: np.stack([sample["images"][key] for sample in batch]) for key in CAMERAS},
        **{key: np.stack([sample[key] for sample in batch])
           for key in ("state", "language", "actions", "action_valid", "frame_index", "episode_id")},
    }


def statistics(stream):
    """Streaming mean/std over retained anchors and their frame-aligned actions."""
    count = 0
    mean = np.zeros((2, 8))
    m2 = np.zeros_like(mean)
    for sample in stream:
        count += 1
        value = np.stack([sample["state"], sample["actions"][0]])
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    if not count:
        raise ValueError("No eligible DROID frames")
    scale = np.maximum(np.sqrt(m2 / count), 1e-6)
    return {"schema": SCHEMA, "count": count, **{
        key: {"mean": mean[i].tolist(), "std": scale[i].tolist()}
        for i, key in enumerate(("state", "actions"))
    }}


def normalize(values, stats, field, inverse=False):
    if stats["schema"] != SCHEMA:
        raise ValueError("DROID normalization schema mismatch")
    mean = np.asarray(stats[field]["mean"], dtype=np.float32)
    std = np.asarray(stats[field]["std"], dtype=np.float32)
    if mean.shape != (8,) or std.shape != (8,) or not np.isfinite([mean, std]).all() or np.any(std <= 0):
        raise ValueError("Invalid DROID statistics")
    return values * std + mean if inverse else (values - mean) / std


class TrainablePolicy(Protocol):
    def compute_loss(self, batch: dict): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


def training_batch(batch, stats):
    return {**batch, "state": normalize(batch["state"], stats, "state"),
            "actions": normalize(batch["actions"], stats, "actions")}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--keep-ranges", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve() / "droid" / "1.0.1"
    shards = sorted(root.glob("droid_101-train.tfrecord-*-of-02048"))
    expected = [f"droid_101-train.tfrecord-{i:05d}-of-02048" for i in range(2048)]
    if [p.name for p in shards] != expected or any(p.stat().st_size == 0 for p in shards):
        raise ValueError("Expected all 2,048 official DROID 1.0.1 train shards")
    stats = statistics(rlds_samples(args.root, args.keep_ranges, chunk=1))
    stats.update({
        "source": SOURCE,
        "split": "train",
        "keep_ranges_sha256": hash_file(args.keep_ranges),
        "metadata_sha256": {name: hash_file(root / name)
                            for name in ("dataset_info.json", "features.json")},
        "shards": [{"name": p.name, "bytes": p.stat().st_size} for p in shards],
    })
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.stats.open("x") as output:
        json.dump(stats, output, indent=2)


if __name__ == "__main__":
    main()
