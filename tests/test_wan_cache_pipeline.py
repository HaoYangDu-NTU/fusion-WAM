from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors import safe_open

from scripts.precompute_wan_latents import write_shard


def test_async_writers_merge_without_losing_existing_windows(tmp_path):
    path = tmp_path / "shard_000000.safetensors"
    values = {str(i): torch.full((16, 2, 2), i, dtype=torch.bfloat16) for i in range(8)}
    with ThreadPoolExecutor(4) as writer:
        futures = [writer.submit(write_shard, path, {key: value}) for key, value in values.items()]
        for future in futures:
            future.result()
    with safe_open(path, framework="pt") as handle:
        assert set(handle.keys()) == set(values)
        for key, expected in values.items():
            torch.testing.assert_close(handle.get_tensor(key), expected, rtol=0, atol=0)


def test_async_failure_preserves_published_shard(tmp_path):
    path = tmp_path / "shard_000000.safetensors"
    original = torch.zeros(4, dtype=torch.bfloat16)
    write_shard(path, {"0": original})
    with ThreadPoolExecutor(1) as writer:
        future = writer.submit(write_shard, path, {"0": original + 1})
        with pytest.raises(AssertionError):
            future.result()
    with safe_open(path, framework="pt") as handle:
        torch.testing.assert_close(handle.get_tensor("0"), original, rtol=0, atol=0)
    assert not list(tmp_path.glob("*.tmp.*"))
