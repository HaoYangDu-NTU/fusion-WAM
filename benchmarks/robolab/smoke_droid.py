"""Inventory the local DROID LeRobot conversion and read an aligned training window."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def check(root):
    info = json.loads((root / "meta/info.json").read_text())
    cameras = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    assert len(cameras) == 3 and info["codebase_version"] == "v3.0"
    tables = sorted(root.glob("data/chunk-*/*.parquet"))
    frames = sum(pq.ParquetFile(path).metadata.num_rows for path in tables)
    episodes = sorted(root.glob("meta/episodes/chunk-*/*.parquet"))
    referenced_videos = set()
    video_indices = [f"videos/{c}/{field}" for c in cameras for field in ("chunk_index", "file_index")]
    length = count = 0
    for path in episodes:
        for batch in pq.ParquetFile(path).iter_batches(columns=["episode_index", "length", "dataset_from_index", "dataset_to_index", *video_indices]):
            ids, lengths, starts, ends = (x.to_numpy() for x in batch.columns[:4])
            for camera in cameras:
                prefix = f"videos/{camera}/"
                pairs = set(zip(batch[prefix+"chunk_index"].to_pylist(), batch[prefix+"file_index"].to_pylist()))
                referenced_videos.update(root / info["video_path"].format(video_key=camera, chunk_index=c, file_index=f) for c, f in pairs)
            assert np.array_equal(ids, np.arange(count, count + len(ids)))
            assert np.all(lengths > 0) and np.array_equal(ends - starts, lengths)
            assert starts[0] == length and np.array_equal(ends[:-1], starts[1:])
            count += len(ids)
            length = int(ends[-1])
    videos = list(root.glob("videos/*/chunk-*/*.mp4"))
    assert count == info["total_episodes"] and length == frames == info["total_frames"]
    assert set(videos) == referenced_videos
    assert len(videos) == info["total_videos"] and all(p.stat().st_size for p in videos)
    ep = next(pq.ParquetFile(episodes[0]).iter_batches(batch_size=1)).to_pylist()[0]
    table = root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action", "language_instruction"]
    rows = next(pq.ParquetFile(table).iter_batches(batch_size=16, columns=columns)).to_pylist()
    assert [r["frame_index"] for r in rows] == list(range(16))
    assert all(r["episode_index"] == ep["episode_index"] for r in rows)
    assert np.allclose([r["timestamp"] for r in rows], np.arange(16) / info["fps"], atol=1e-5)
    for key in ["observation.state", "action"]:
        x = np.asarray([r[key] for r in rows])
        assert x.shape == (16, 8) and np.isfinite(x).all()
    prompt = rows[0]["language_instruction"]
    assert isinstance(prompt, str) and prompt.strip()
    rgb = {}
    for camera in cameras:
        prefix = f"videos/{camera}/"
        assert ep[prefix + "from_timestamp"] == 0
        path = root / info["video_path"].format(video_key=camera, chunk_index=ep[prefix+"chunk_index"], file_index=ep[prefix+"file_index"])
        with av.open(str(path)) as video:
            decoder = video.decode(video=0)
            images = np.stack([next(decoder).to_ndarray(format="rgb24") for _ in rows])
        assert tuple(images.shape[1:]) == tuple(info["features"][camera]["shape"]) and images.std() > 1
        rgb[camera] = list(images.shape)
    return dict(root=str(root.resolve()), episodes=count, frames=frames, parquet_files=len(tables),
                video_files=len(videos), fps=info["fps"], episode=ep["episode_index"],
                language=prompt, rgb=rgb, action_shape=[16, 8], state_shape=[16, 8])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(check(parser.parse_args().root), indent=2))
