#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/ablation_runtime.sh"

TARGET_PID=${1:?usage: watch_training_memory.sh TARGET_PID LOG_DIR [LIMIT_GIB]}
LOG_DIR=${2:?usage: watch_training_memory.sh TARGET_PID LOG_DIR [LIMIT_GIB]}
LIMIT_GIB=${3:-600}
MEMORY_LIMIT_BYTES=$((LIMIT_GIB * 1024 * 1024 * 1024))
LOG_FILE="$LOG_DIR/memory_watchdog.log"
mkdir -p "$LOG_DIR"
TARGET_START=$(process_start_time "$TARGET_PID")
if [[ -z "$TARGET_START" ]]; then
  echo "Target process $TARGET_PID does not exist" >&2
  exit 1
fi
if ! PGID=$(wait_for_process_group_leader "$TARGET_PID" "$TARGET_START"); then
  echo "Target process $TARGET_PID did not become its process-group leader" >&2
  exit 1
fi

while process_matches_start_time "$TARGET_PID" "$TARGET_START"; do
  now=$(date --iso-8601=seconds)
  current=$(< /sys/fs/cgroup/memory.current)
  limit=$(< /sys/fs/cgroup/memory.max)
  oom_kill=$(awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events)
  gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
  printf '%s cgroup=%s/%s gpu_used_mib=%s oom_kill=%s\n' \
    "$now" "$current" "$limit" "$gpu_used" "$oom_kill" >>"$LOG_FILE"
  if (( current > MEMORY_LIMIT_BYTES )); then
    printf '%s stopping process group %s: cgroup memory exceeded %s GiB\n' \
      "$now" "$PGID" "$LIMIT_GIB" >>"$LOG_FILE"
    if process_matches_start_time "$TARGET_PID" "$TARGET_START"; then
      kill -TERM -- "-$PGID"
    fi
    exit 70
  fi
  sleep 30
done

printf '%s target process %s exited\n' "$(date --iso-8601=seconds)" "$TARGET_PID" >>"$LOG_FILE"
