import pytest
import torch

from vjepa_policy.state import (
    pack_normalized_state,
    packed_state_dim,
    prepare_proprio_state,
)


def test_pack_normalized_state_uses_zero_padding_and_valid_mask():
    state = torch.tensor([[1.0, -2.0, 0.5]])
    packed = pack_normalized_state(state, max_state_dim=48)

    assert packed.shape == (1, 96)
    torch.testing.assert_close(packed[0, :3], state[0])
    torch.testing.assert_close(packed[0, 3:48], torch.zeros(45))
    torch.testing.assert_close(packed[0, 48:51], torch.ones(3))
    torch.testing.assert_close(packed[0, 51:], torch.zeros(45))


def test_fixed_width_state_rejects_sincos_and_oversized_input():
    state = torch.randn(2, 4)
    with pytest.raises(ValueError, match="fixed-width state packing"):
        prepare_proprio_state(
            state, raw_state_dim=4, encoding="sincos", max_state_dim=48
        )
    with pytest.raises(ValueError, match="max_state_dim"):
        pack_normalized_state(torch.zeros(2, 49), max_state_dim=48)
    assert packed_state_dim(48) == 96
