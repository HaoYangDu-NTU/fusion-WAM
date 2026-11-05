#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
FIRST=${1:?usage: launch_pair.sh FIRST_ENCODER SECOND_ENCODER}
SECOND=${2:?usage: launch_pair.sh FIRST_ENCODER SECOND_ENCODER}

bash scripts/latent_space_ablation/run.sh "$FIRST" --check
bash scripts/latent_space_ablation/run.sh "$SECOND" --check

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
  local run_dir=$6
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
    OUTDIR="$run_dir" \
    bash scripts/latent_space_ablation/run.sh "$encoder" \
    >"$run_dir/launcher.log" 2>&1 &
  local train_pid=$!
  echo "$train_pid" >"$run_dir/train.pid"

  setsid bash scripts/watch_training_memory.sh "$train_pid" "$run_dir" 760 \
    >"$run_dir/watchdog_launcher.log" 2>&1 &
  echo $! >"$run_dir/watchdog.pid"

  setsid env \
    EVAL_GPUS="$eval_gpus" \
    BASE_PORT="$eval_port" \
    bash scripts/latent_space_ablation/wait_and_eval.sh \
      "$encoder" "$run_dir" "$train_pid" \
    >"$run_dir/eval_launcher.log" 2>&1 &
  echo $! >"$run_dir/eval_queue_launcher.pid"

  printf '%s train_pid=%s gpus=%s run_dir=%s\n' \
    "$encoder" "$train_pid" "$gpu_ids" "$run_dir"
}

FIRST_DIR=${FIRST_DIR:-$ROOT/runs/latent_${FIRST}_independent224_gbs128_step21360_seed7}
SECOND_DIR=${SECOND_DIR:-$ROOT/runs/latent_${SECOND}_independent224_gbs128_step21360_seed7}
launch_one "$FIRST" 0,1,2,3 29551 0,1,2,3 15510 "$FIRST_DIR"
launch_one "$SECOND" 4,5,6,7 29552 4,5,6,7 15520 "$SECOND_DIR"
