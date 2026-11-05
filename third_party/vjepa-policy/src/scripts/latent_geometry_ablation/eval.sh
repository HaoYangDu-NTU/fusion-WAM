#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

EXPERIMENT=${1:?usage: eval.sh EXPERIMENT RUN_DIR CHECKPOINT}
RUN_DIR=${2:?usage: eval.sh EXPERIMENT RUN_DIR CHECKPOINT}
CHECKPOINT=${3:?usage: eval.sh EXPERIMENT RUN_DIR CHECKPOINT}
WEIGHTS=${WEIGHTS:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub}

bash scripts/latent_geometry_ablation/run.sh "$EXPERIMENT" --check

case "$EXPERIMENT" in
  dinov2_native16)
    VJEPA2_ENCODER_CHECKPOINT=${DINOV2_CHECKPOINT:-$WEIGHTS/dinov2-large-47b73eefe95e}
    ENCODER_FAMILY=dinov2
    ENCODER_MODEL_NAME=dinov2_vitl_native16
    ENCODER_CHECKPOINT_KEY=local_pretrained
    ;;
  wan22_stride2_t5)
    VJEPA2_ENCODER_CHECKPOINT=${WAN22_CHECKPOINT:-$WEIGHTS/wan2.2-ti2v-5b-diffusers-b8fff7315c76/vae/diffusion_pytorch_model.safetensors}
    ENCODER_FAMILY=wan2_2_stride2_t5
    ENCODER_MODEL_NAME=vae38
    ENCODER_CHECKPOINT_KEY=encoder_quant_conv
    ;;
  *)
    echo "Unknown latent geometry experiment: $EXPERIMENT" >&2
    exit 64
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export POLICY_PYTHON=${POLICY_PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
export LIBERO_PYTHON=${LIBERO_PYTHON:-/data/zhangyang/miniconda3/envs/libero/bin/python}
export POLICY_SERVER_SCRIPT=scripts/latent_geometry_ablation/serve.py
export VJEPA2_ENCODER_CHECKPOINT ENCODER_FAMILY ENCODER_MODEL_NAME ENCODER_CHECKPOINT_KEY
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
export DATASET_STATS=${DATASET_STATS:-$RUN_DIR/dataset_stats.json}
export POLICY_ENCODER_INTERPOLATE_ROPE=0
export POLICY_CORRECTED_PREDICTOR_ROPE=1
export POLICY_CROP_SIZE=224
export POLICY_VIEW_LAYOUT=independent
export PRED_DEPTH=24
export PRED_EMBED_DIM=1024
export PRED_NUM_HEADS=16
export ACTION_HIDDEN_SIZE=512
export ACTION_NUM_LAYERS=24
export REPLAN_STEPS=${REPLAN_STEPS:-16}
export NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
export EVAL_SEED=${EVAL_SEED:-7}
export LIBERO_MUJOCO_GL=${LIBERO_MUJOCO_GL:-egl}
export EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_egl_r16_50trials_seed7}

exec bash examples/libero/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
