"""Verify pinned GR1 Teleop inventory, paired episodes and parquet frame counts."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    tasks = inventory["datasets"]
    if {p.name for p in args.dataset.glob("gr1_unified.*")} != set(tasks):
        raise ValueError("Local task names differ from pinned inventory")
    for task, expected in tasks.items():
        root = args.dataset / task
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != 20 or info["total_frames"] != expected["frames"]:
            raise ValueError(f"Metadata mismatch for {task}")
        names = {f"episode_{i:06d}" for i in range(expected["episodes"])}
        frames = 0
        for directory, suffix in [("data", "parquet"), ("videos", "mp4")]:
            paths = sorted((root / directory).rglob(f"*.{suffix}"))
            if len(paths) != len(names) or {p.stem for p in paths} != names:
                raise ValueError(f"Incomplete {directory} for {task}")
            if any(p.stat().st_size == 0 for p in paths):
                raise ValueError(f"Empty file in {task}/{directory}")
            if suffix == "parquet":
                frames = sum(pq.read_metadata(p).num_rows for p in paths)
        if frames != expected["frames"]:
            raise ValueError(f"Frame count mismatch for {task}")
        print(task, expected["episodes"], frames, flush=True)


if __name__ == "__main__":
    main()
