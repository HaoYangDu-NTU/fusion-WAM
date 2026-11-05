#!/usr/bin/env bash
set -euo pipefail

# Evaluate one exact encoder-ablation checkpoint with the same LIBERO
# protocol as the 96.9 baseline, while selecting the untruncated 128-token
# cache explicitly.  EVAL_GPUS/EVAL_SIM_GPUS are supplied by the caller so
# two policies can occupy disjoint four-GPU groups.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
VARIANT=${1:?usage: eval_exact_t5len128.sh VARIANT RUN_DIR [CHECKPOINT]}
RUN_DIR=${2:?usage: eval_exact_t5len128.sh VARIANT RUN_DIR [CHECKPOINT]}
CHECKPOINT=${3:-$RUN_DIR/checkpoint_step021360.pt}
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints

case "$VARIANT" in
  vjepa2_vitg)
    export ENCODER_FAMILY=vjepa2 ENCODER_MODEL_NAME=vit_giant_xformers
    export ENCODER_CHECKPOINT_KEY=target_encoder
    export VJEPA2_ENCODER_CHECKPOINT="$WEIGHTS/vitg.pt"
    ;;
  vjepa2_1_vitl)
    export ENCODER_FAMILY=vjepa2_1 ENCODER_MODEL_NAME=vit_large
    export ENCODER_CHECKPOINT_KEY=ema_encoder
    export VJEPA2_ENCODER_CHECKPOINT="$WEIGHTS/vjepa2_1_vitl_dist_vitG_384.pt"
    ;;
  vjepa2_1_vitg)
    export ENCODER_FAMILY=vjepa2_1 ENCODER_MODEL_NAME=vit_giant_xformers
    export ENCODER_CHECKPOINT_KEY=target_encoder
    export VJEPA2_ENCODER_CHECKPOINT="$WEIGHTS/vjepa2_1_vitg_384.pt"
    ;;
  *) echo "Unknown encoder variant: $VARIANT" >&2; exit 64 ;;
esac

export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128}
export DATASET_STATS=${DATASET_STATS:-$RUN_DIR/dataset_stats.json}
export POLICY_T5_LEN=128
export POLICY_CROP_SIZE=224
export POLICY_VIEW_LAYOUT=independent
export POLICY_ENCODER_INTERPOLATE_ROPE=1
export POLICY_CORRECTED_PREDICTOR_ROPE=1
export REPLAN_STEPS=16
export NUM_TRIALS_PER_TASK=50
export EVAL_SEED=7
export LIBERO_MUJOCO_GL=egl
export EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_r16_50trials_seed7_egl}

exec bash "$ROOT/examples/libero/evaluate_policy.sh" "$RUN_DIR" "$CHECKPOINT"
