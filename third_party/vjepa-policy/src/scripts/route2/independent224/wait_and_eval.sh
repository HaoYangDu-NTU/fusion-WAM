#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
RUN_DIR=${1:?usage: wait_and_eval.sh RUN_DIR [TRAIN_PID]}
TRAIN_PID=${2:-}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoint_step021360.pt}
echo $$ >"$RUN_DIR/eval_queue.pid"

while [[ ! -s "$CHECKPOINT" ]]; do
  if [[ -n "$TRAIN_PID" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Training process $TRAIN_PID exited without final checkpoint: $CHECKPOINT" >&2
    exit 1
  fi
  sleep 60
done

if [[ -n "$TRAIN_PID" ]]; then
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 10
  done
fi

exec bash scripts/route2/independent224/eval.sh "$RUN_DIR" "$CHECKPOINT"
