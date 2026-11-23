"""Read one LIBERO stereo frame at a time and write cache progress atomically."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from openwam.dataloader.transforms.multiview import assemble_two_view_layout


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class CurrentStereoFrames(Dataset):
    """Use the unchanged reader's camera decode/layout, without decoding future windows."""

    def __init__(self, reader, layout=None):
        self.reader = reader
        self.layout = getattr(reader, "assemble_stereo", assemble_two_view_layout) if layout is None else layout
        assert reader._window_stride == 1 and reader._seperate_encode_view
        assert reader._color_jitter is None and reader._head_camera_choices is None
        assert np.all(reader._ep_valid_start == 0), "Trimmed episodes need explicit historical-frame coverage"
        assert len(reader) == int(reader._eps_df["length"].sum())

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        ds = self.reader
        episode = int(np.searchsorted(ds._cum_n_starts, index, side="right") - 1)
        offset = int(ds._ep_valid_start[episode]) + index - int(ds._cum_n_starts[episode])
        row = ds._eps_df.iloc[episode]
        local = np.array([0], dtype=np.int64)
        head = ds._decode_one_camera(ds._head_camera, row, episode, offset, local, kind="head")
        wrist = ds._decode_one_camera(ds._left_wrist_camera, row, episode, offset, local, kind="wrist")
        if not head:
            raise RuntimeError(f"Empty head frame at {index}")
        image = self.layout(head[0], wrist[0] if wrist else None)
        return dict(
            index=index,
            episode=int(row["episode_index"]),
            frame=offset,
            pixels=torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1),
        )
