"""Fixed-width proprioceptive state packing for cross-robot checkpoints.

The DROID predictor is conditioned on a native state vector padded to a common
width and accompanied by a validity mask.  The historical LIBERO path remains
available by passing ``max_state_dim=None`` (the CLI uses ``0`` as its legacy
sentinel).
"""

from __future__ import annotations

import torch


DEFAULT_MAX_STATE_DIM = 48


def packed_state_dim(max_state_dim: int) -> int:
    """Return the width of ``[padded_state, valid_mask]``."""

    max_state_dim = int(max_state_dim)
    if max_state_dim <= 0:
        raise ValueError(f"max_state_dim must be positive, got {max_state_dim}")
    return 2 * max_state_dim


def pack_normalized_state(
    state: torch.Tensor,
    max_state_dim: int = DEFAULT_MAX_STATE_DIM,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad a normalized state and append a binary validity mask.

    Leading batch/time dimensions are preserved.  Padding is always zero and
    supplied dimensions are marked valid unless an explicit mask is provided.
    """

    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    if state.ndim == 0:
        raise ValueError("state must have at least one dimension")
    max_state_dim = int(max_state_dim)
    if max_state_dim <= 0:
        raise ValueError(f"max_state_dim must be positive, got {max_state_dim}")
    native_dim = int(state.shape[-1])
    if native_dim <= 0 or native_dim > max_state_dim:
        raise ValueError(
            "state last dimension must be in "
            f"[1, max_state_dim={max_state_dim}], got {native_dim}"
        )

    padded = state.new_zeros((*state.shape[:-1], max_state_dim))
    padded[..., :native_dim] = state

    mask = state.new_zeros((*state.shape[:-1], max_state_dim))
    if valid_mask is None:
        mask[..., :native_dim] = 1
    else:
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.as_tensor(valid_mask, device=state.device)
        if valid_mask.shape[:-1] != state.shape[:-1]:
            raise ValueError(
                "valid_mask leading dimensions must match state: "
                f"{tuple(valid_mask.shape)} vs {tuple(state.shape)}"
            )
        mask_dim = int(valid_mask.shape[-1])
        if mask_dim not in (native_dim, max_state_dim):
            raise ValueError(
                "valid_mask last dimension must match native state or max_state_dim, "
                f"got {mask_dim}; state={native_dim}, max={max_state_dim}"
            )
        mask[..., :mask_dim] = valid_mask.to(device=state.device, dtype=state.dtype)
        if mask_dim == max_state_dim:
            # Dimensions not represented by the native vector cannot be valid.
            mask[..., native_dim:] = 0

    return torch.cat((padded, mask), dim=-1)


def prepare_proprio_state(
    state: torch.Tensor,
    *,
    raw_state_dim: int,
    encoding: str = "identity",
    max_state_dim: int | None = None,
) -> torch.Tensor:
    """Validate and encode state for policy conditioning.

    ``max_state_dim=None`` preserves the original variable-width identity or
    sin/cos representation.  Fixed-width packing intentionally supports only
    identity encoding because the second half of the vector is a validity mask.
    """

    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    raw_state_dim = int(raw_state_dim)
    if state.ndim < 1 or state.shape[-1] != raw_state_dim:
        raise ValueError(
            f"state must end in dimension {raw_state_dim}, got {tuple(state.shape)}"
        )
    if max_state_dim is None or int(max_state_dim) == 0:
        if encoding == "identity":
            return state
        if encoding == "sincos":
            return torch.cat((torch.sin(state), torch.cos(state)), dim=-1)
        raise ValueError(f"unsupported proprio encoding {encoding!r}")
    max_state_dim = int(max_state_dim)
    if encoding != "identity":
        raise ValueError(
            "fixed-width state packing requires proprio_encoding='identity'; "
            "use max_state_dim=None for the legacy sincos representation"
        )
    return pack_normalized_state(state, max_state_dim)


__all__ = [
    "DEFAULT_MAX_STATE_DIM",
    "packed_state_dim",
    "pack_normalized_state",
    "prepare_proprio_state",
]
