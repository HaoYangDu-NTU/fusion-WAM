import numpy as np
import pytest

from .droid import CAMERAS, SCHEMA, batches, normalize, samples, statistics, training_batch
from .provenance import checkpoint_manifest, hash_file
from .serve import Adapter


def step(index):
    state = {"joint_position": np.full(7, index), "gripper_position": [0.5]}
    target = {"joint_position": np.full(7, index + 10), "gripper_position": [0.25]}
    return {"observation": {**state, **{k: np.full((2, 3, 3), index, np.uint8) for k in CAMERAS}},
            "action_dict": target, "language_instruction": b"", "language_instruction_2": b"Pick up the cup"}


def test_filtered_anchors_and_masked_terminal_targets():
    result = list(samples([step(i) for i in range(4)], [(1, 2), (3, 4)], chunk=3))
    assert [x["frame_index"] for x in result] == [1, 3]
    np.testing.assert_array_equal(result[0]["actions"][:, 0], [11, 12, 13])
    np.testing.assert_array_equal(result[-1]["actions"][:, 0], [13, 13, 13])
    assert result[-1]["action_valid"].tolist() == [True, False, False]
    assert result[0]["language"] == "Pick up the cup"
    assert set(result[0]["images"]) == set(CAMERAS)


def test_batch_and_action_normalization_round_trip():
    stream = [{**x, "episode_id": "test"} for x in samples([step(0), step(1)], [(0, 2)], 2)]
    stats = statistics(stream)
    batch = next(batches(stream, 2))
    converted = training_batch(batch, stats)
    assert batch["actions"].shape == (2, 2, 8)
    np.testing.assert_allclose(normalize(converted["actions"], stats, "actions", inverse=True),
                               batch["actions"])


def test_deployment_denormalizes_once_and_preserves_arm_order():
    stats = {"schema": SCHEMA,
             "state": {"mean": [1.] * 8, "std": [2.] * 8},
             "actions": {"mean": list(range(8)), "std": [2.] * 8}}
    image = np.zeros((2, 2, 3), np.uint8)

    class Probe:
        def predict(self, obs):
            np.testing.assert_array_equal(obs["state"], np.ones(8))
            assert obs["images"]["wrist_image_left"] is image
            return np.ones((8, 8))

    obs = {"observation/joint_position": np.full(7, 3), "observation/gripper_position": [3],
           "observation/exterior_image_1_left": image, "observation/wrist_image_left": image,
           "prompt": "test"}
    output = Adapter(Probe(), stats, 8).infer(obs)["actions"]
    np.testing.assert_array_equal(output[0], np.arange(8) + 2)


def test_checkpoint_manifest_records_real_bytes_and_refuses_partial(tmp_path):
    file = tmp_path / "weight.bin"
    file.write_bytes(b"sample")
    assert checkpoint_manifest(tmp_path)["files"][0]["sha256"] == hash_file(file)
    (tmp_path / "weight.part").touch()
    with pytest.raises(ValueError, match="partial"):
        checkpoint_manifest(tmp_path)
