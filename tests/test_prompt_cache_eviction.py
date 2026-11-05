"""Regression for OrderedDict.popitem invoking the LRU __getitem__ hook."""
from openwam.deploy.engine import _BoundedPromptEmbedCache


def test_eviction_preserves_recency_and_values():
    cache = _BoundedPromptEmbedCache(2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1
    cache["c"] = 3
    assert list(cache) == ["a", "c"]
    cache["a"] = 4
    cache["d"] = 5
    assert list(cache) == ["a", "d"]
    assert cache["a"] == 4
    for i in range(100):
        cache[str(i)] = i
    assert list(cache) == ["98", "99"]
    assert cache["98"] == 98 and cache["99"] == 99
