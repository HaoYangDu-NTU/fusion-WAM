#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
RUN_DIR=${1:?usage: eval.sh RUN_DIR [CHECKPOINT]}
CHECKPOINT=${2:-$RUN_DIR/checkpoint_step021360.pt}

export VJEPA2_ENCODER_CHECKPOINT=${VJEPA2_ENCODER_CHECKPOINT:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints/vitg.pt}
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
export EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_r16_50trials}
export POLICY_CROP_SIZE=224
export POLICY_VIEW_LAYOUT=independent
export POLICY_ENCODER_INTERPOLATE_ROPE=1
export POLICY_CORRECTED_PREDICTOR_ROPE=1
export REPLAN_STEPS=16
export NUM_TRIALS_PER_TASK=50

exec bash examples/libero/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
