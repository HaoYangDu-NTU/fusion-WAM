#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WAIT_PID=${1:-}
ENCODERS=(vjepa2_vitl vjepa2_1_vitl vjepa2_1_vitg)
mkdir -p "$ROOT/runs/encoder_ablation_queue"

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] waiting for process $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

run_watched() {
  local encoder=$1
  local output_dir=$2
  local max_steps=$3
  local checkpointing_blocks=$4
  mkdir -p "$output_dir"
  setsid env \
    OUTDIR="$output_dir" \
    MAX_STEPS="$max_steps" \
    ACTIVATION_CHECKPOINTING_BLOCKS="$checkpointing_blocks" \
    bash "$ROOT/scripts/encoder_ablation/run.sh" "$encoder" \
    >"$output_dir/launcher.log" 2>&1 &
  local train_pid=$!
  bash "$ROOT/scripts/watch_training_memory.sh" "$train_pid" "$output_dir" 600 &
  local watchdog_pid=$!
  set +e
  wait "$train_pid"
  local status=$?
  wait "$watchdog_pid"
  local watchdog_status=$?
  set -e
  if (( status != 0 )); then
    return "$status"
  fi
  return "$watchdog_status"
}

select_checkpointing() {
  local encoder=$1
  local candidate
  for candidate in 4 8 12; do
    local output_dir="$ROOT/runs/encoder_ablation_queue/preflight_${encoder}_acb${candidate}"
    echo "[$(date -Is)] preflight $encoder ACB=$candidate" >&2
    if run_watched "$encoder" "$output_dir" 1 "$candidate"; then
      rm -rf "$output_dir"
      printf '%s\n' "$candidate"
      return 0
    fi
    echo "[$(date -Is)] preflight failed $encoder ACB=$candidate" >&2
    rm -rf "$output_dir"
  done
  return 1
}

for encoder in "${ENCODERS[@]}"; do
  run_dir="$ROOT/runs/full_encoder_ablation_${encoder}_independent224_gbs128_step21360_seed7"
  final_checkpoint="$run_dir/checkpoint_step021360.pt"
  if [[ ! -s "$final_checkpoint" ]]; then
    if ! checkpointing_blocks=$(select_checkpointing "$encoder"); then
      echo "[$(date -Is)] no passing preflight for $encoder" >&2
      continue
    fi
    echo "[$(date -Is)] training $encoder ACB=$checkpointing_blocks"
    if ! run_watched "$encoder" "$run_dir" 21360 "$checkpointing_blocks"; then
      echo "[$(date -Is)] training failed for $encoder" >&2
      continue
    fi
  fi

  echo "[$(date -Is)] evaluating $encoder"
  if ! bash "$ROOT/scripts/encoder_ablation/eval.sh" \
      "$encoder" "$run_dir" "$final_checkpoint"; then
    echo "[$(date -Is)] evaluation failed for $encoder" >&2
    continue
  fi
  echo "[$(date -Is)] completed $encoder"
  if [[ "$encoder" == vjepa2_1_vitl ]]; then
    echo "[$(date -Is)] evaluating original Route2 seeds 17 and 29 in parallel"
    if ! bash "$ROOT/scripts/route2/action_cross/eval_seed_stability_parallel.sh"; then
      echo "[$(date -Is)] Route2 seed stability evaluation failed" >&2
    fi
  fi
done

echo "[$(date -Is)] encoder ablation queue finished"
