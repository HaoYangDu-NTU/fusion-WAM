#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
SOURCE_ROOT=$(realpath -e "${SOURCE_ROOT:-$ROOT}")
source "$SOURCE_ROOT/scripts/ablation_runtime.sh"
verify_source_snapshot "$SOURCE_ROOT"

PREREQ_PID=${1:?usage: queue_vjepa21_healthy.sh PREREQ_EVAL_PID PREREQ_RUN_DIR [RUN_DIR]}
PREREQ_RUN_DIR=$(realpath -m "${2:?usage: queue_vjepa21_healthy.sh PREREQ_EVAL_PID PREREQ_RUN_DIR [RUN_DIR]}")
RUN_DIR=$(realpath -m "${3:-$ROOT/runs/action_only_vjepa2_1_vitl_independent224_gbs128_step21360_seed7_restart2}")
PREREQ_SUMMARY=$PREREQ_RUN_DIR/libero_eval_r16_50trials_seed7/summary.txt
GPU_IDS=0,1,2,3
GPUS=(0 1 2 3)

if [[ ! "$PREREQ_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid prerequisite PID: $PREREQ_PID" >&2
  exit 64
fi
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to append to non-empty run directory: $RUN_DIR" >&2
  exit 1
fi
echo $$ >"$PREREQ_RUN_DIR/vjepa21_restart2_queue.pid"
PREREQ_START=$(process_start_time "$PREREQ_PID")

echo "[$(date -Is)] waiting for prerequisite training and evaluation PID $PREREQ_PID"
while process_matches_start_time "$PREREQ_PID" "$PREREQ_START"; do
  sleep 60
done
if [[ ! -s "$PREREQ_SUMMARY" ]] || \
   ! grep -Fxq 'successful_suites: 4/4' "$PREREQ_SUMMARY"; then
  echo "Prerequisite LIBERO evaluation did not complete successfully: $PREREQ_SUMMARY" >&2
  exit 1
fi

check_gpu_health() {
  for gpu in "${GPUS[@]}"; do
    uncorrected=$(nvidia-smi --id="$gpu" \
      --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader,nounits | tr -d ' ')
    recovery=$(nvidia-smi -q -i "$gpu" | \
      awk -F ': ' '/GPU Recovery Action/ && !found {print $2; found=1}' | xargs)
    if [[ "$uncorrected" != 0 || "$recovery" != None ]]; then
      echo "GPU $gpu is unhealthy: uncorrected_ecc=$uncorrected recovery=$recovery" >&2
      return 1
    fi
  done
}

check_gpu_health

while :; do
  busy=()
  for gpu in "${GPUS[@]}"; do
    used=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
      --format=csv,noheader | sed '/^[[:space:]]*$/d')
    if [[ -n "$used" ]]; then
      busy+=("$gpu:$used")
    fi
  done
  if (( ${#busy[@]} == 0 )); then
    break
  fi
  echo "[$(date -Is)] waiting for GPUs $GPU_IDS: ${busy[*]}"
  sleep 30
done
check_gpu_health
verify_source_snapshot "$SOURCE_ROOT"

if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to append to non-empty run directory: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

setsid env \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  MAIN_PROCESS_PORT=29543 \
  ACTIVATION_CHECKPOINTING_BLOCKS=7 \
  OUTDIR="$RUN_DIR" \
  bash "$SOURCE_ROOT/scripts/action_only_ablation/run.sh" vjepa2_1_vitl \
  >"$RUN_DIR/launcher.log" 2>&1 &
train_pid=$!
echo "$train_pid" >"$RUN_DIR/train.pid"
process_start_time "$train_pid" >"$RUN_DIR/train.start"

setsid bash "$SOURCE_ROOT/scripts/watch_training_memory.sh" \
  "$train_pid" "$RUN_DIR" 760 \
  >"$RUN_DIR/watchdog_launcher.log" 2>&1 &
echo $! >"$RUN_DIR/watchdog.pid"

setsid env \
  PYTHONDONTWRITEBYTECODE=1 \
  EVAL_GPUS="$GPU_IDS" \
  BASE_PORT=15420 \
  bash "$SOURCE_ROOT/scripts/action_only_ablation/wait_and_eval.sh" \
    vjepa2_1_vitl "$RUN_DIR" "$train_pid" \
  >"$RUN_DIR/eval_launcher.log" 2>&1 &
eval_pid=$!
echo "$eval_pid" >"$RUN_DIR/eval_queue_launcher.pid"
process_start_time "$eval_pid" >"$RUN_DIR/eval_queue_launcher.start"

printf '[%s] vjepa2_1_vitl train_pid=%s gpus=%s acb=7 run_dir=%s\n' \
  "$(date -Is)" "$train_pid" "$GPU_IDS" "$RUN_DIR"
