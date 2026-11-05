"""Opt-in per-rank training timings: OPENWAM_TIMING_DIR=/path/to/run.

CUDA events measure elapsed stream time, including idle gaps, not summed kernel
time. Nested phases overlap and must not be summed. process_cpu_ms covers the
training process (not its DataLoader workers); profiled steps should be excluded
from steady-state throughput summaries. Synchronize only once when
finishing a measured step. Without the environment variable no events are used.
"""
import gc
import json
import os
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch

_active = None


@contextmanager
def phase(name):
    timer = _active
    if timer is None:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    host_start = time.perf_counter()
    try:
        with torch.profiler.record_function('openwam.' + name):
            yield
    finally:
        host_ms = (time.perf_counter() - host_start) * 1000
        end.record()
        timer.events.append((name, start, end, host_ms))


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if _active is None:
                return fn(*args, **kwargs)
            with phase(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


class StepTimer:
    def __init__(self, rank):
        directory = os.environ.get('OPENWAM_TIMING_DIR')
        self.path = Path(directory) / f'rank{rank}.jsonl' if directory else None
        self.events = []
        self.wrapped = []
        self.trace_step = int(os.environ.get('OPENWAM_TRACE_STEP', '0')) if rank == 0 else 0
        self.step = 0
        self.profiler = None
        self.gc_start = None
        self.gc_ms = 0.0
        self.gc_calls = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            gc.callbacks.append(self._gc_callback)

    def _gc_callback(self, event, info):
        if event == "start" and _active is self:
            self.gc_start = time.perf_counter()
        elif event == "stop" and self.gc_start is not None:
            self.gc_ms += (time.perf_counter() - self.gc_start) * 1000
            self.gc_calls += 1
            self.gc_start = None

    def wrap(self, obj, method, name):
        if self.path:
            original = getattr(obj, method)
            self.wrapped.append((obj, method, original))
            @wraps(original)
            def observed(*args, **kwargs):
                if _active is None:
                    return original(*args, **kwargs)
                with phase(name):
                    result = original(*args, **kwargs)
                def shapes(value):
                    if isinstance(value, torch.Tensor):
                        return list(value.shape)
                    if isinstance(value, (tuple, list)):
                        return [shapes(x) for x in value if isinstance(x, (torch.Tensor, tuple))]
                    return None
                self.shapes[name] = dict(inputs=shapes(args), output=shapes(result))
                return result
            setattr(obj, method, observed)

    def batches(self, iterable):
        global _active
        if not self.path:
            yield from iterable
            return
        iterator = iter(iterable)
        try:
            while True:
                self.events = []
                self.shapes = {}
                self.gc_ms = 0.0
                self.gc_calls = 0
                self.step += 1
                if self.step == self.trace_step:
                    self.profiler = torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
                    ], record_shapes=True)
                    self.profiler.__enter__()
                self.wall_start = time.perf_counter()
                self.cpu_start = time.process_time()
                _active = self
                try:
                    with phase('data_wait'):
                        batch = next(iterator)
                except StopIteration:
                    return
                yield batch
        finally:
            _active = None

    def finish_step(self, step, metrics):
        global _active
        if not self.path:
            return
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self.wall_start) * 1000
        phases = {}
        for name, start, end, host_ms in self.events:
            entry = phases.setdefault(name, {'cuda_elapsed_ms': 0.0, 'host_ms': 0.0, 'calls': 0})
            entry['cuda_elapsed_ms'] += start.elapsed_time(end)
            entry['host_ms'] += host_ms
            entry['calls'] += 1
        row = dict(step=step, wall_ms=wall_ms, phases=phases,
                   shapes=self.shapes,
                   process_cpu_ms=(time.process_time() - self.cpu_start) * 1000,
                   profiled=self.profiler is not None,
                   gc_ms=self.gc_ms, gc_calls=self.gc_calls,
                   allocated_gb=torch.cuda.memory_allocated() / 2**30,
                   peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                   metrics={key: float(value) for key, value in metrics.items()})
        with self.path.open('a') as out:
            out.write(json.dumps(row) + '\n')
        _active = None
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            events = self.profiler.key_averages(group_by_input_shape=True)
            table = events.table(sort_by="self_cuda_time_total", row_limit=50)
            (self.path.parent / 'operators.txt').write_text(table)
            self.profiler.export_chrome_trace(str(self.path.parent / 'trace.json'))
            self.profiler = None

    def close(self):
        global _active
        _active = None
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        for obj, method, original in reversed(self.wrapped):
            setattr(obj, method, original)
        self.wrapped.clear()
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
