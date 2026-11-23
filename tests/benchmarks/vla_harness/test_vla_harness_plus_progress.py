"""A long healthy Plus episode must not look like a stalled worker."""
from types import SimpleNamespace
import numpy as np
import pytest
from vla_eval import watchdog
from vla_eval.recording import NullEpisodeRecorder
from benchmarks.vla_harness.libero_plus import OpenWAMLIBEROPlusBenchmark

def test_watchdog_tracks_completed_steps_in_long_episode(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: now[0])
    dog = watchdog.ProgressWatchdog(1200)
    monkeypatch.setattr(watchdog, "_watchdog", dog)
    bench = OpenWAMLIBEROPlusBenchmark()
    bench._recorder = NullEpisodeRecorder()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    def step(action):
        now[0] += 200
        return {"agentview_image": image}, 0.0, False, {}
    bench._env = SimpleNamespace(step=step)
    for _ in range(8):
        bench.step({"actions": np.zeros(7)})
        assert dog.idle_s() < 1200, "Healthy environment steps were invisible to watchdog"
    assert bench._steps == 8
    now[0] += 1201
    assert dog.idle_s() > 1200  # no timer thread may hide a genuine stall

def test_failed_step_does_not_report_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(watchdog, "pet", calls.append)
    bench = OpenWAMLIBEROPlusBenchmark()
    bench._recorder = NullEpisodeRecorder()
    def step(action):
        raise TimeoutError("simulator stalled")
    bench._env = SimpleNamespace(step=step)
    with pytest.raises(TimeoutError):
        bench.step({"actions": np.zeros(7)})
    assert calls == []
