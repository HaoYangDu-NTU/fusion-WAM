from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from scripts.train_vjepa_policy import _load_predictor_initialization
from scripts.teledata.train import parse_teledata_args
from vjepa_policy.datasets.lerobot_video_v3 import (
    LeRobotVideoV3Dataset,
    read_v3_tasks,
)


def test_read_v3_tasks_uses_task_index(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pq.write_table(
        pa.table({"task_index": [2, 0], "instruction": ["third", "first"]}),
        meta / "tasks.parquet",
    )

    assert read_v3_tasks(tmp_path) == {2: "third", 0: "first"}


def test_teledata_training_always_uses_letterbox_resize():
    args = parse_teledata_args(
        [
            "--dataset-dirs",
            "/tmp/data",
            "--text-cache-dir",
            "/tmp/text",
            "--checkpoint",
            "/tmp/encoder",
            "--image-resize-mode",
            "stretch",
        ]
    )

    assert args.image_resize_mode == "letterbox"


def test_training_config_records_image_resize_mode():
    script = Path(__file__).parents[1] / "scripts" / "teledata" / "run.sh"
    assert "--image-resize-mode letterbox" in script.read_text(encoding="utf-8")

    gr1_script = Path(__file__).parents[1] / "scripts" / "gr1" / "train.sh"
    assert "--image-resize-mode letterbox" in gr1_script.read_text(encoding="utf-8")


def test_clamped_indices_match_lerobot_padding():
    indices, is_pad = LeRobotVideoV3Dataset._clamped_indices(
        anchor=10,
        offsets=(-4, 0, 4),
        start=8,
        end=13,
    )

    assert indices == [8, 10, 12]
    assert is_pad.tolist() == [True, False, True]


class _Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.view_embedding = nn.Embedding(4, 3)
        self.projection = nn.Linear(3, 3, bias=False)


def test_predictor_view_map_preserves_strict_loading(tmp_path: Path):
    predictor = _Predictor()
    source = predictor.state_dict()
    source["view_embedding.weight"] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    checkpoint = tmp_path / "predictor.pt"
    torch.save(
        {"predictor": source, "model_topology": {"predictor_num_views": 2}},
        checkpoint,
    )

    metadata = _load_predictor_initialization(
        predictor, checkpoint, view_map=[0, 0, 0, 0]
    )

    assert predictor.view_embedding.weight.detach().tolist() == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
    assert metadata["view_embedding_initialization"] == {
        "source_rows": 2,
        "target_rows": 4,
        "target_to_source_map": [0, 0, 0, 0],
    }


def test_predictor_view_map_rejects_invalid_source_row(tmp_path: Path):
    predictor = _Predictor()
    source = predictor.state_dict()
    source["view_embedding.weight"] = torch.zeros(2, 3)
    checkpoint = tmp_path / "predictor.pt"
    torch.save({"predictor": source}, checkpoint)

    with pytest.raises(ValueError, match="exceed source rows=2"):
        _load_predictor_initialization(predictor, checkpoint, view_map=[0, 0, 1, 2])
