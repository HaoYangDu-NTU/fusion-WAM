#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/ablation_runtime.sh"
verify_source_snapshot "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
ENCODER=${1:?usage: wait_and_eval.sh ENCODER RUN_DIR TRAIN_PID}
RUN_DIR=${2:?usage: wait_and_eval.sh ENCODER RUN_DIR TRAIN_PID}
TRAIN_PID=${3:?usage: wait_and_eval.sh ENCODER RUN_DIR TRAIN_PID}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoint_step021360.pt}
echo $$ >"$RUN_DIR/eval_queue.pid"
TRAIN_START=$(process_start_time "$TRAIN_PID")

while [[ ! -s "$CHECKPOINT" ]]; do
  if ! process_matches_start_time "$TRAIN_PID" "$TRAIN_START"; then
    echo "Training process $TRAIN_PID exited without final checkpoint: $CHECKPOINT" >&2
    exit 1
  fi
  sleep 60
done

while process_matches_start_time "$TRAIN_PID" "$TRAIN_START"; do
  sleep 10
done

verify_source_snapshot "$ROOT"
exec bash scripts/latent_space_ablation/eval.sh "$ENCODER" "$RUN_DIR" "$CHECKPOINT"
