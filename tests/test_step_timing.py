import json
from types import SimpleNamespace

from openwam.train.utils import step_timing as timing


def test_disabled_tracker_never_calls_cuda(monkeypatch):
    monkeypatch.delenv('OPENWAM_TIMING_DIR', raising=False)
    def reject(*args, **kwargs):
        raise AssertionError('Disabled tracker called CUDA')
    monkeypatch.setattr(timing.torch.cuda, 'Event', reject)
    monkeypatch.setattr(timing.torch.cuda, 'synchronize', reject)
    callbacks = list(timing.gc.callbacks)
    timer = timing.StepTimer(0)
    assert timing.gc.callbacks == callbacks
    fn = timing.timed('compute')(lambda x: x + 1)
    assert [fn(x) for x in timer.batches([1, 2])] == [2, 3]
    timer.finish_step(1, {})
    timer.close()


def test_nested_timings_write_per_rank_and_restore_methods(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENWAM_TIMING_DIR', str(tmp_path))
    class Event:
        clock = 0
        def __init__(self, **kwargs):
            pass
        def record(self):
            Event.clock += 1
            self.value = Event.clock
        def elapsed_time(self, other):
            return other.value - self.value
    syncs = []
    monkeypatch.setattr(timing.torch.cuda, 'Event', Event)
    monkeypatch.setattr(timing.torch.cuda, 'synchronize', lambda: syncs.append(1))
    monkeypatch.setattr(timing.torch.cuda, 'memory_allocated', lambda: 0)
    monkeypatch.setattr(timing.torch.cuda, 'max_memory_allocated', lambda: 0)
    original = lambda x: x * 2
    obj = SimpleNamespace(step=original)
    timer = timing.StepTimer(2)
    timer.wrap(obj, 'step', 'inner')
    for step, batch in enumerate(timer.batches([3, 4]), 1):
        with timing.phase('outer'):
            assert obj.step(batch) == batch * 2
        timer._gc_callback("start", {"generation": 2})
        timer._gc_callback("stop", {"generation": 2})
        timer.finish_step(step, {'loss': 1.25})
    timer.close()
    assert obj.step is original and timing._active is None
    assert timer._gc_callback not in timing.gc.callbacks
    assert len(syncs) == 2
    rows = [json.loads(line) for line in (tmp_path / 'rank2.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2]
    for row in rows:
        assert row['metrics'] == {'loss': 1.25}
        assert row['process_cpu_ms'] >= 0
        assert row['profiled'] is False
        assert row['gc_ms'] >= 0
        assert row['gc_calls'] >= 1
        assert set(row['phases']) == {'data_wait', 'inner', 'outer'}
        assert all(entry['calls'] == 1 for entry in row['phases'].values())
        assert row['phases']['outer']['cuda_elapsed_ms'] > row['phases']['inner']['cuda_elapsed_ms']
