"""Wait for a clear four-GPU window, then launch the matched 600-trial pilot."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/E00_plus/frozen_router_20261111/evaluation_queue"
STATUS = REPORT / "status.json"
LOG = REPORT / "console.log"
FREE_MEMORY_MIB = 3000
REQUIRED_CLEAR_POLLS = 3
POLL_SECONDS = 60


def write_status(state, **extra):
    REPORT.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps({"state": state, "updated": datetime.now().isoformat(), **extra}, indent=2) + "\n")
    temp.replace(STATUS)


def gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    clear_polls = 0
    with LOG.open("a", buffering=1) as log:
        log.write(f"[{datetime.now().isoformat()}] waiting for four GPUs below {FREE_MEMORY_MIB} MiB each\n")
        while clear_polls < REQUIRED_CLEAR_POLLS:
            try:
                used = gpu_memory()
            except Exception as exc:
                used = None
                log.write(f"[{datetime.now().isoformat()}] nvidia-smi unavailable: {exc}\n")
            if used is not None and len(used) == 4 and all(value < FREE_MEMORY_MIB for value in used):
                clear_polls += 1
            else:
                clear_polls = 0
            write_status("waiting", gpu_memory_mib=used, clear_polls=clear_polls,
                         required_clear_polls=REQUIRED_CLEAR_POLLS)
            if clear_polls < REQUIRED_CLEAR_POLLS:
                time.sleep(POLL_SECONDS)

        log.write(f"[{datetime.now().isoformat()}] four-GPU window clear; launching matched pilot\n")
        write_status("launching", gpu_memory_mib=gpu_memory())
        command = [sys.executable, str(ROOT / "scripts/e00_plus_frozen_router.py"), "evaluate"]
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write_status("completed" if result.returncode == 0 else "failed",
                     returncode=result.returncode, finished=datetime.now().isoformat())
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
