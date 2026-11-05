"""Shared model capabilities and low-overhead evaluation phase counters."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import time

from openwam.history import validate_condition_frame_offsets


@dataclass(frozen=True)
class EvaluationCapabilities:
    execution_horizon: int
    history_offsets: tuple[int, ...] = (0,)
    history_key: str | None = None
    native_batch: bool = True
    requires_step_observe: bool = False
    stateful: bool = False
    protocol_version: int = 1
    prediction_horizon: int | None = None

    def __post_init__(self):
        if type(self.execution_horizon) is not int or self.execution_horizon < 1:
            raise ValueError("execution_horizon must be a positive integer")
        if self.prediction_horizon is not None and (
                type(self.prediction_horizon) is not int or self.prediction_horizon < self.execution_horizon):
            raise ValueError("prediction_horizon must cover execution_horizon")
        validate_condition_frame_offsets(self.history_offsets)
        if self.protocol_version != 1:
            raise ValueError("Unsupported evaluation capability version")
        if self.history_key not in (None, "history_images", "clean_condition_images"):
            raise ValueError("Unsupported history field")
        if self.history_key is None and tuple(self.history_offsets) != (0,):
            raise ValueError("Historical frames require a history field")
        if self.history_key == "history_images" and len(self.history_offsets) != 2:
            raise ValueError("history_images requires past/current offsets")
        for name in ("native_batch", "requires_step_observe", "stateful"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "history_offsets": tuple(value.get("history_offsets", (0,)))})


class PhaseStats:
    """CPU wall times, not CUDA kernel times; one object per owning thread."""

    def __init__(self, role):
        self.role = role
        self.seconds = {}
        self.counts = {}
        self.started = time.monotonic()

    def add(self, phase, seconds=0.0, count=1):
        self.seconds[phase] = self.seconds.get(phase, 0.0) + seconds
        self.counts[phase] = self.counts.get(phase, 0) + count

    @contextmanager
    def measure(self, phase, count=1):
        started = time.monotonic()
        try:
            yield
        finally:
            self.add(phase, time.monotonic() - started, count)

    def snapshot(self):
        return {"role": self.role, "pid": os.getpid(), "wall_seconds": time.monotonic() - self.started,
                "seconds": dict(self.seconds), "counts": dict(self.counts)}

    def emit(self, **context):
        row = {**self.snapshot(), **context}
        logging.getLogger(__name__).info("evaluation_performance %s", json.dumps(row, sort_keys=True))
        directory = os.environ.get("OPENWAM_PERFORMANCE_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"{self.role}-{os.getpid()}.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return row
