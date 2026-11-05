#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

bash scripts/latent_geometry_ablation/run.sh dinov2_native16 --check
bash scripts/latent_geometry_ablation/run.sh wan22_stride2_t5 --check

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
  local experiment=$1
  local gpu_ids=$2
  local port=$3
  local run_dir=$4
  if [[ -e "$run_dir/checkpoint_step021360.pt" ]]; then
    echo "Refusing to overwrite existing final checkpoint: $run_dir/checkpoint_step021360.pt" >&2
    return 1
  fi
  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty run directory: $run_dir" >&2
    return 1
  fi
  mkdir -p "$run_dir"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    NUM_GPUS=4 \
    MAIN_PROCESS_PORT="$port" \
    OUTDIR="$run_dir" \
    ALLOW_NONEMPTY_OUTDIR=1 \
    bash scripts/latent_geometry_ablation/run.sh "$experiment" \
    >"$run_dir/launcher.log" 2>&1 &
  local train_pid=$!
  echo "$train_pid" >"$run_dir/train.pid"
  printf '%s train_pid=%s gpus=%s run_dir=%s\n' \
    "$experiment" "$train_pid" "$gpu_ids" "$run_dir"
}

FIRST_DIR=${FIRST_DIR:-$ROOT/runs/latent_dinov2_native16_independent224_gbs128_step21360_seed7}
SECOND_DIR=${SECOND_DIR:-$ROOT/runs/latent_wan22_stride2_t5_independent224_gbs128_step21360_seed7}
FIRST_PORT=${FIRST_PORT:-29561}
SECOND_PORT=${SECOND_PORT:-29562}

for run_dir in "$FIRST_DIR" "$SECOND_DIR"; do
  if [[ -e "$run_dir/checkpoint_step021360.pt" ]]; then
    echo "Refusing to overwrite existing final checkpoint: $run_dir/checkpoint_step021360.pt" >&2
    exit 1
  fi
  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty run directory: $run_dir" >&2
    exit 1
  fi
done

# The two four-GPU groups are intentionally fixed by the experiment protocol.
launch_one dinov2_native16 0,1,2,3 "$FIRST_PORT" "$FIRST_DIR"
launch_one wan22_stride2_t5 4,5,6,7 "$SECOND_PORT" "$SECOND_DIR"
