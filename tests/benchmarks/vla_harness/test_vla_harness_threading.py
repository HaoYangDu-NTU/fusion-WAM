"""Model loading and calls from rotating harness threads share one CUDA owner."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from benchmarks.vla_harness import model_server


def test_model_thread_affinity_and_serialization(monkeypatch):
    loaded, inference = [], []
    active = False

    def load(*_):
        loaded.append(threading.get_ident())
        return SimpleNamespace(
            engine=None,
            _ckpt_contract=lambda: {"representation": "eef"},
            _obs_preprocessor=SimpleNamespace(preprocess=lambda x: x),
        )

    class Engine:
        def __init__(self, _):
            assert threading.get_ident() == loaded[0]

        def generate_batch(self, observations):
            nonlocal active
            assert not active
            active = True
            inference.append(threading.get_ident())
            time.sleep(0.01)
            active = False
            return [np.tile([0.2, -0.3, 0.4, 1, 0, 0, 1, 0, 0, 0.37], (10, 1)) for _ in observations]

    monkeypatch.setattr(model_server, "load_native_server", load)
    monkeypatch.setattr(model_server, "BatchedOpenWAMEngine", Engine)
    server = model_server.OpenWAMModelServer("unused")
    server._episodes["s"] = "e"
    obs = {"images": {}, "state": np.zeros(10)}
    ctx = SimpleNamespace(session_id="s", episode_id="e")
    barrier = threading.Barrier(4)

    def call(_):
        barrier.wait(timeout=5)
        return threading.get_ident(), server.predict_batch([obs], [ctx])[0]["actions"]

    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            results = list(callers.map(call, range(4)))
        assert len({thread for thread, _ in results}) == 4
        assert len(inference) == 4
        assert set(inference) == set(loaded)
        assert loaded[0] not in {thread for thread, _ in results}
        for _, actions in results:
            np.testing.assert_array_equal(actions, results[0][1])
    finally:
        server._inference_thread.shutdown()
