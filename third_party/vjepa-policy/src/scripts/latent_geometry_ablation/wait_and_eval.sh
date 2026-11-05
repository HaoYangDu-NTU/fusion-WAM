#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/ablation_runtime.sh"

EXPERIMENT=${1:?usage: wait_and_eval.sh EXPERIMENT RUN_DIR TRAIN_PID}
RUN_DIR=${2:?usage: wait_and_eval.sh EXPERIMENT RUN_DIR TRAIN_PID}
TRAIN_PID=${3:?usage: wait_and_eval.sh EXPERIMENT RUN_DIR TRAIN_PID}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoint_step021360.pt}
EVAL_GPUS=${EVAL_GPUS:?Set EVAL_GPUS to four comma-separated GPU IDs}
BASE_PORT=${BASE_PORT:?Set BASE_PORT for four parallel LIBERO servers}
EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_egl_r16_50trials_seed7}
FINAL_STEP=21360

if [[ ! "$TRAIN_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid training PID: $TRAIN_PID" >&2
  exit 64
fi
if [[ ! "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT > 65512 )); then
  echo "Invalid evaluation base port: $BASE_PORT" >&2
  exit 64
fi
IFS=',' read -r -a gpus <<<"$EVAL_GPUS"
if (( ${#gpus[@]} != 4 )) || \
   (( $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l) != 4 )); then
  echo "EVAL_GPUS must contain four distinct GPU IDs: $EVAL_GPUS" >&2
  exit 64
fi
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
    echo "Invalid GPU ID: $gpu" >&2
    exit 64
  fi
done

echo $$ >"$RUN_DIR/eval_queue.pid"
process_start_time $$ >"$RUN_DIR/eval_queue.start"
TRAIN_START=$(process_start_time "$TRAIN_PID" || true)
if [[ -z "$TRAIN_START" && ! -s "$CHECKPOINT" ]]; then
  echo "Training process $TRAIN_PID is not running and checkpoint is missing" >&2
  exit 1
fi

echo "[$(date -Is)] waiting experiment=$EXPERIMENT train_pid=$TRAIN_PID checkpoint=$CHECKPOINT"
while [[ ! -s "$CHECKPOINT" ]]; do
  if ! process_matches_start_time "$TRAIN_PID" "$TRAIN_START"; then
    echo "Training exited without final checkpoint: $CHECKPOINT" >&2
    exit 1
  fi
  sleep 60
done
while process_matches_start_time "$TRAIN_PID" "$TRAIN_START"; do
  sleep 10
done

if ! rg -q "step=$FINAL_STEP/$FINAL_STEP loss=[+-]?[0-9]" "$RUN_DIR/train.log"; then
  echo "Training log has no finite final-step record: $RUN_DIR/train.log" >&2
  exit 1
fi
if rg -qi \
  '(loss|world|action|cos_sim|grad_norm)=([+-]?(nan|inf(inity)?))' \
  "$RUN_DIR/train.log"; then
  echo "Training log contains a non-finite metric: $RUN_DIR/train.log" >&2
  exit 1
fi

while :; do
  busy=()
  for gpu in "${gpus[@]}"; do
    used=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
      --format=csv,noheader | sed '/^[[:space:]]*$/d')
    if [[ -n "$used" ]]; then
      busy+=("$gpu:$used")
    fi
  done
  if (( ${#busy[@]} == 0 )); then
    break
  fi
  echo "[$(date -Is)] waiting for evaluation GPUs: ${busy[*]}"
  sleep 30
done

for attempt in 1 2 3; do
  attempt_port=$((BASE_PORT + 10 * (attempt - 1)))
  echo "[$(date -Is)] LIBERO attempt=$attempt experiment=$EXPERIMENT gpus=$EVAL_GPUS ports=$attempt_port-$((attempt_port + 3))"
  if env \
    EVAL_GPUS="$EVAL_GPUS" \
    BASE_PORT="$attempt_port" \
    EVAL_DIR="$EVAL_DIR" \
    bash scripts/latent_geometry_ablation/eval.sh \
      "$EXPERIMENT" "$RUN_DIR" "$CHECKPOINT" && \
    grep -Fxq 'successful_suites: 4/4' "$EVAL_DIR/summary.txt"; then
    printf 'completed=%s\nsummary=%s\n' "$(date -Is)" "$EVAL_DIR/summary.txt" \
      >"$RUN_DIR/eval_complete.txt"
    exit 0
  fi
  if (( attempt < 3 )); then
    sleep 30
  fi
done

echo "LIBERO evaluation failed after three attempts: $EXPERIMENT" >&2
exit 1
