#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/ablation_runtime.sh"
verify_source_snapshot "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

ENCODER=${1:?usage: launch_one.sh ENCODER GPU_IDS [RUN_DIR]}
GPU_IDS=${2:?usage: launch_one.sh ENCODER GPU_IDS [RUN_DIR]}
RUN_DIR=${3:-$ROOT/runs/latent_${ENCODER}_independent224_gbs128_step21360_seed7}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29551}
EVAL_BASE_PORT=${EVAL_BASE_PORT:-15510}

bash scripts/latent_space_ablation/run.sh "$ENCODER" --check
IFS=',' read -r -a gpus <<<"$GPU_IDS"
if (( ${#gpus[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
  exit 64
fi
if (( $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l) != 4 )); then
  echo "GPU_IDS must not contain duplicates: $GPU_IDS" >&2
  exit 64
fi

for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
    echo "Invalid GPU ID: $gpu" >&2
    exit 64
  fi
  uncorrected=$(nvidia-smi --id="$gpu" \
    --query-gpu=ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits | tr -d ' ')
  recovery=$(nvidia-smi -q -i "$gpu" | \
    awk -F ': ' '/GPU Recovery Action/ && !found {print $2; found=1}' | xargs)
  if [[ "$uncorrected" != 0 || "$recovery" != None ]]; then
    echo "GPU $gpu is unhealthy: uncorrected_ecc=$uncorrected recovery=$recovery" >&2
    exit 1
  fi
  used=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
    --format=csv,noheader | sed '/^[[:space:]]*$/d')
  if [[ -n "$used" ]]; then
    echo "GPU $gpu already has compute processes: $used" >&2
    exit 1
  fi
done

final_checkpoint=$RUN_DIR/checkpoint_step021360.pt
if [[ -e "$final_checkpoint" ]]; then
  echo "Refusing to overwrite existing final checkpoint: $final_checkpoint" >&2
  exit 1
fi
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to append to non-empty run directory: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
verify_source_snapshot "$ROOT"

setsid env \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
  OUTDIR="$RUN_DIR" \
  bash scripts/latent_space_ablation/run.sh "$ENCODER" \
  >"$RUN_DIR/launcher.log" 2>&1 &
train_pid=$!
echo "$train_pid" >"$RUN_DIR/train.pid"
process_start_time "$train_pid" >"$RUN_DIR/train.start"

setsid bash scripts/watch_training_memory.sh "$train_pid" "$RUN_DIR" 760 \
  >"$RUN_DIR/watchdog_launcher.log" 2>&1 &
echo $! >"$RUN_DIR/watchdog.pid"

setsid env \
  PYTHONDONTWRITEBYTECODE=1 \
  EVAL_GPUS="$GPU_IDS" \
  BASE_PORT="$EVAL_BASE_PORT" \
  bash scripts/latent_space_ablation/wait_and_eval.sh \
    "$ENCODER" "$RUN_DIR" "$train_pid" \
  >"$RUN_DIR/eval_launcher.log" 2>&1 &
eval_pid=$!
echo "$eval_pid" >"$RUN_DIR/eval_queue_launcher.pid"
process_start_time "$eval_pid" >"$RUN_DIR/eval_queue_launcher.start"

printf '%s train_pid=%s gpus=%s run_dir=%s\n' \
  "$ENCODER" "$train_pid" "$GPU_IDS" "$RUN_DIR"
