#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

RUN_DIR=${RUN_DIR:-$ROOT/runs/full_route2_action_cross_correctedrope_pred24x1024_gbs128_step21360_bs32ga1_acb4_nw8_seed7}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoint_step021360.pt}
WEIGHTS=${WEIGHTS:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints}
export VJEPA2_ENCODER_CHECKPOINT=${VJEPA2_ENCODER_CHECKPOINT:-$WEIGHTS/vitg.pt}
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
export POLICY_CROP_SIZE=256
export POLICY_VIEW_LAYOUT=quadrant
export POLICY_ENCODER_INTERPOLATE_ROPE=0
export POLICY_CORRECTED_PREDICTOR_ROPE=1
export REPLAN_STEPS=16
export NUM_TRIALS_PER_TASK=50
export EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}

[[ -s "$CHECKPOINT" ]] || { echo "Missing $CHECKPOINT" >&2; exit 1; }

run_seed() {
  local seed=$1
  local base_port=$2
  EVAL_SEED="$seed" \
  BASE_PORT="$base_port" \
  EVAL_DIR="$RUN_DIR/libero_eval_r16_50trials_seed${seed}" \
    bash examples/libero/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
}

declare -A pids=()
run_seed 17 17001 >"$RUN_DIR/libero_seed17_queue.log" 2>&1 &
pids[17]=$!
run_seed 29 18001 >"$RUN_DIR/libero_seed29_queue.log" 2>&1 &
pids[29]=$!

failed=()
for seed in 17 29; do
  if ! wait "${pids[$seed]}"; then
    failed+=("$seed")
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "Route2 parallel evaluation failed for seeds: ${failed[*]}" >&2
  exit 1
fi
echo "[done] Route2 Seed 17 and Seed 29 evaluations completed"
