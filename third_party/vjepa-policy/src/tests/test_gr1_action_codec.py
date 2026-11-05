import numpy as np
import pytest
import torch

from vjepa_policy.policy_serving.gr1 import (
    action_29d_to_robocasa,
    decode_gr1_action_chunk,
    grouped_state_to_29d,
    relative_action_to_absolute,
)


def _grouped_state(array_module, leading_shape=()):
    specs = (
        ("left_arm", 7, 10),
        ("right_arm", 7, 20),
        ("left_hand", 6, 30),
        ("right_hand", 6, 40),
        ("waist", 3, 50),
    )
    result = {}
    for name, width, offset in specs:
        values = np.arange(width, dtype=np.float32) + offset
        values = np.broadcast_to(values, (*leading_shape, width)).copy()
        result[name] = torch.from_numpy(values) if array_module is torch else values
    return result


@pytest.mark.parametrize("array_module", [np, torch])
def test_grouped_state_uses_official_29d_order(array_module):
    grouped = _grouped_state(array_module, leading_shape=(2, 3))
    packed = grouped_state_to_29d(grouped)

    assert packed.shape == (2, 3, 29)
    expected = [
        *range(10, 17),
        *range(20, 27),
        *range(30, 36),
        *range(40, 46),
        *range(50, 53),
    ]
    np.testing.assert_array_equal(
        packed[0, 0].cpu().numpy()
        if isinstance(packed, torch.Tensor)
        else packed[0, 0],
        expected,
    )


@pytest.mark.parametrize("array_module", [np, torch])
def test_relative_action_adds_latest_state_to_first_26_dimensions_only(array_module):
    grouped = _grouped_state(array_module, leading_shape=(2, 3))
    state = grouped_state_to_29d(grouped)
    relative = np.full((2, 4, 29), 2.0, dtype=np.float32)
    relative[..., 26:] = 100.0
    relative = torch.from_numpy(relative) if array_module is torch else relative
    original = (
        relative.clone() if isinstance(relative, torch.Tensor) else relative.copy()
    )

    absolute = relative_action_to_absolute(relative, state)

    expected_relative = state[:, -1, :26]
    if isinstance(absolute, torch.Tensor):
        expected = (expected_relative[:, None] + 2.0).expand(2, 4, 26)
        torch.testing.assert_close(absolute[..., :26], expected)
        torch.testing.assert_close(absolute[..., 26:], torch.full((2, 4, 3), 100.0))
        torch.testing.assert_close(relative, original)
    else:
        expected = np.broadcast_to(expected_relative[:, None] + 2.0, (2, 4, 26))
        np.testing.assert_allclose(absolute[..., :26], expected)
        np.testing.assert_allclose(absolute[..., 26:], 100.0)
        np.testing.assert_array_equal(relative, original)


def test_unbatched_action_uses_latest_state_and_splits_robocasa_groups():
    state = grouped_state_to_29d(_grouped_state(np, leading_shape=(2,)))
    relative = np.zeros((4, 29), dtype=np.float32)
    relative[:, 26:] = np.array([1.0, 2.0, 3.0])

    absolute = relative_action_to_absolute(relative, state)
    grouped_action = action_29d_to_robocasa(absolute)

    assert list(grouped_action) == [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    assert grouped_action["action.left_arm"].shape == (4, 7)
    assert grouped_action["action.right_arm"].shape == (4, 7)
    assert grouped_action["action.left_hand"].shape == (4, 6)
    assert grouped_action["action.right_hand"].shape == (4, 6)
    assert grouped_action["action.waist"].shape == (4, 3)
    np.testing.assert_array_equal(
        grouped_action["action.left_arm"],
        np.broadcast_to(state[-1, :7], (4, 7)),
    )
    np.testing.assert_array_equal(
        grouped_action["action.waist"],
        np.broadcast_to([1.0, 2.0, 3.0], (4, 3)),
    )


@pytest.mark.parametrize("array_module", [np, torch])
@pytest.mark.parametrize("action_horizon", [16, 32])
def test_absolute_action_chunk_is_split_without_state_relative_decoding(
    array_module, action_horizon
):
    absolute = np.arange(
        2 * action_horizon * 29, dtype=np.float32
    ).reshape(2, action_horizon, 29)
    absolute = torch.from_numpy(absolute) if array_module is torch else absolute

    grouped = decode_gr1_action_chunk(absolute, action_horizon=action_horizon)

    assert grouped["action.left_arm"].shape == (2, action_horizon, 7)
    assert grouped["action.waist"].shape == (2, action_horizon, 3)
    if array_module is torch:
        torch.testing.assert_close(grouped["action.left_arm"], absolute[..., :7])
        torch.testing.assert_close(grouped["action.waist"], absolute[..., -3:])
    else:
        np.testing.assert_array_equal(grouped["action.left_arm"], absolute[..., :7])
        np.testing.assert_array_equal(grouped["action.waist"], absolute[..., -3:])


def test_absolute_action_chunk_rejects_a_mismatched_declared_horizon():
    with pytest.raises(ValueError, match=r"absolute_action must be \[B,32,29\]"):
        decode_gr1_action_chunk(
            np.zeros((2, 16, 29), dtype=np.float32), action_horizon=32
        )


def test_codec_rejects_missing_or_misaligned_grouped_state():
    grouped = _grouped_state(np)
    del grouped["right_hand"]
    with pytest.raises(KeyError, match="right_hand"):
        grouped_state_to_29d(grouped)

    grouped = _grouped_state(torch, leading_shape=(2,))
    grouped["waist"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="leading shape"):
        grouped_state_to_29d(grouped)


def test_codec_rejects_invalid_action_or_state_shapes():
    with pytest.raises(ValueError, match="action must have shape"):
        relative_action_to_absolute(
            np.zeros((2, 4, 28), dtype=np.float32),
            np.zeros((2, 29), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="does not match action batch"):
        relative_action_to_absolute(
            torch.zeros(2, 4, 29),
            torch.zeros(3, 29),
        )
    with pytest.raises(TypeError, match="same array type"):
        relative_action_to_absolute(
            torch.zeros(2, 4, 29),
            np.zeros((2, 29), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="dimension 29"):
        action_29d_to_robocasa(np.zeros((2, 4, 28), dtype=np.float32))
