"""Sample evaluation memory and preserve child exit reasons before cleanup."""

import json
import signal
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path


def process_snapshot(process):
    result = {"pid": process.pid, "command": process.args, "returncode": process.poll()}
    if result["returncode"] is not None and result["returncode"] < 0:
        result["signal"] = signal.Signals(-result["returncode"]).name
    try:
        status = dict(line.split(":", 1) for line in Path(f"/proc/{process.pid}/status").read_text().splitlines())
        for key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
            result[key + "_bytes"] = int(status.get(key, "0").split()[0]) * 1024
        rollup = Path(f"/proc/{process.pid}/smaps_rollup").read_text().splitlines()[1:]
        fields = dict(line.split(":", 1) for line in rollup)
        result["pss_bytes"] = int(fields["Pss"].split()[0]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return result


class RuntimeMonitor:
    def __init__(self, output_dir, recording_db=None):
        self.output_dir = output_dir
        self.recording_db = recording_db
        self.last_sample = 0.0
        self.last_recording_check = 0.0

    def sample(self, processes, *, force=False):
        if not force and time.monotonic() - self.last_sample < 15:
            return
        self.last_sample = time.monotonic()
        cgroup = Path("/sys/fs/cgroup")
        stat = {k: int(v) for k, v in (line.split() for line in (cgroup / "memory.stat").read_text().splitlines())}
        events = {k: int(v) for k, v in (line.split() for line in (cgroup / "memory.events").read_text().splitlines())}
        row = {
            "time": datetime.now().isoformat(),
            "memory_current": int((cgroup / "memory.current").read_text()),
            "memory_max": (cgroup / "memory.max").read_text().strip(),
            "memory_stat": stat,
            "memory_events": events,
            "processes": [process_snapshot(p) for p in processes],
        }
        for name in ("cpu.max", "cpu.stat", "cpu.pressure", "memory.pressure", "io.pressure"):
            source = cgroup / name
            if source.is_file():
                row[name.replace(".", "_")] = source.read_text().strip()
        with (self.output_dir / "runtime_memory.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return row

    def check(self, servers, workers, *, force=False):
        failed = [p for p in servers if p.poll() is not None]
        failed += [p for p in workers if p.poll() not in (None, 0)]
        if failed:
            row = self.sample([*servers, *workers], force=True)
            row["failed_processes"] = [process_snapshot(p) for p in failed]
            (self.output_dir / "process_failure.json").write_text(json.dumps(row, indent=2) + "\n")
            raise RuntimeError(f"Evaluation child failed: {row['failed_processes']}")
        # Harness records an episode exception and keeps its process alive.
        # Stop the run before such an exception can be mistaken for a rollout failure.
        if self.recording_db is None or (not force and time.monotonic() - self.last_recording_check < 15):
            return
        self.last_recording_check = time.monotonic()
        with closing(sqlite3.connect(f"file:{Path(self.recording_db).resolve()}?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            errors = [
                dict(row)
                for row in connection.execute(
                    "SELECT task_name, episode_id, context, failure_reason, failure_detail "
                    "FROM episode_results WHERE status = 'error' OR coalesce(failure_reason, '') != ''"
                )
            ]
        if errors:
            (self.output_dir / "episode_failure.json").write_text(json.dumps(errors, indent=2) + "\n")
            raise RuntimeError(f"Evaluation episode failed; see {self.output_dir / 'episode_failure.json'}")
