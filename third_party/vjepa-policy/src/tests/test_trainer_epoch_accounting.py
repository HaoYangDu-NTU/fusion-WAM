import pytest

from vjepa_policy.trainer import _full_accumulation_microbatches


@pytest.mark.parametrize(
    ("microbatches", "accumulation", "usable", "optimizer_steps", "dropped"),
    [
        (2136, 1, 2136, 2136, 0),
        (8545, 4, 8544, 2136, 1),
        (17091, 8, 17088, 2136, 3),
        (34183, 16, 34176, 2136, 7),
    ],
)
def test_full_accumulation_windows(
    microbatches, accumulation, usable, optimizer_steps, dropped
):
    result = _full_accumulation_microbatches(microbatches, accumulation)
    assert result == usable
    assert result // accumulation == optimizer_steps
    assert microbatches - result == dropped


@pytest.mark.parametrize("microbatches,accumulation", [(0, 1), (1, 0)])
def test_full_accumulation_windows_require_positive_counts(microbatches, accumulation):
    with pytest.raises(ValueError):
        _full_accumulation_microbatches(microbatches, accumulation)
