#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

VJEPA2_DIR=${VJEPA2_DIR:-$ROOT/runs/action_only_vjepa2_vitl_independent224_gbs128_step21360_seed7}
VJEPA21_DIR=${VJEPA21_DIR:-$ROOT/runs/action_only_vjepa2_1_vitl_independent224_gbs128_step21360_seed7}
VJEPA2_ACB=${VJEPA2_ACB:-7}
VJEPA21_ACB=${VJEPA21_ACB:-7}

for gpu in {0..7}; do
  uncorrected=$(nvidia-smi --id="$gpu" \
    --query-gpu=ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits | tr -d ' ')
  recovery=$(nvidia-smi -q -i "$gpu" | \
    awk -F ': ' '/GPU Recovery Action/ && !found {print $2; found=1}' | xargs)
  if [[ "$uncorrected" != 0 || "$recovery" != None ]]; then
    echo "GPU $gpu is unhealthy: uncorrected_ecc=$uncorrected recovery=$recovery" >&2
    exit 1
  fi
  used=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')
  if [[ -n "$used" ]]; then
    echo "GPU $gpu already has compute processes: $used" >&2
    exit 1
  fi
done

launch_one() {
  local encoder=$1
  local gpu_ids=$2
  local port=$3
  local eval_gpus=$4
  local eval_port=$5
  local checkpointing_blocks=$6
  local run_dir=$7
  local final_checkpoint="$run_dir/checkpoint_step021360.pt"

  if [[ -e "$final_checkpoint" ]]; then
    echo "Refusing to overwrite existing final checkpoint: $final_checkpoint" >&2
    return 1
  fi
  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty run directory: $run_dir" >&2
    return 1
  fi
  mkdir -p "$run_dir"

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    MAIN_PROCESS_PORT="$port" \
    ACTIVATION_CHECKPOINTING_BLOCKS="$checkpointing_blocks" \
    OUTDIR="$run_dir" \
    bash scripts/action_only_ablation/run.sh "$encoder" \
    >"$run_dir/launcher.log" 2>&1 &
  local train_pid=$!
  echo "$train_pid" >"$run_dir/train.pid"

  setsid bash scripts/watch_training_memory.sh "$train_pid" "$run_dir" 760 \
    >"$run_dir/watchdog_launcher.log" 2>&1 &
  echo $! >"$run_dir/watchdog.pid"

  setsid env \
    EVAL_GPUS="$eval_gpus" \
    BASE_PORT="$eval_port" \
    bash scripts/action_only_ablation/wait_and_eval.sh \
      "$encoder" "$run_dir" "$train_pid" \
    >"$run_dir/eval_launcher.log" 2>&1 &
  echo $! >"$run_dir/eval_queue_launcher.pid"

  printf '%s train_pid=%s gpus=%s acb=%s run_dir=%s\n' \
    "$encoder" "$train_pid" "$gpu_ids" "$checkpointing_blocks" "$run_dir"
}

launch_one vjepa2_vitl 0,1,2,3 29541 0,1,2,3 15410 "$VJEPA2_ACB" "$VJEPA2_DIR"
launch_one vjepa2_1_vitl 4,5,6,7 29542 4,5,6,7 15420 "$VJEPA21_ACB" "$VJEPA21_DIR"
