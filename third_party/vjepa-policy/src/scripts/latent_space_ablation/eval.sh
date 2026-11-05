#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ENCODER=${1:?usage: eval.sh ENCODER RUN_DIR CHECKPOINT}
RUN_DIR=${2:?usage: eval.sh ENCODER RUN_DIR CHECKPOINT}
CHECKPOINT=${3:?usage: eval.sh ENCODER RUN_DIR CHECKPOINT}
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub

bash "$ROOT/scripts/latent_space_ablation/run.sh" "$ENCODER" --check

case "$ENCODER" in
  dinov2_vitl)
    VJEPA2_ENCODER_CHECKPOINT=${DINOV2_CHECKPOINT:-$WEIGHTS/dinov2-large-47b73eefe95e}
    ENCODER_FAMILY=dinov2
    ENCODER_MODEL_NAME=dinov2_vitl
    ENCODER_CHECKPOINT_KEY=local_pretrained
    ;;
  dinov3_vitl)
    VJEPA2_ENCODER_CHECKPOINT=${DINOV3_CHECKPOINT:-$WEIGHTS/dinov3-vitl16-ea8dc2863c51}
    ENCODER_FAMILY=dinov3
    ENCODER_MODEL_NAME=dinov3_vitl
    ENCODER_CHECKPOINT_KEY=local_pretrained
    ;;
  wan22_vae38)
    VJEPA2_ENCODER_CHECKPOINT=${WAN22_CHECKPOINT:-$WEIGHTS/wan2.2-ti2v-5b-diffusers-b8fff7315c76/vae/diffusion_pytorch_model.safetensors}
    ENCODER_FAMILY=wan2_2
    ENCODER_MODEL_NAME=vae38
    ENCODER_CHECKPOINT_KEY=encoder_quant_conv
    ;;
  internvideo3_8b)
    VJEPA2_ENCODER_CHECKPOINT=${INTERNVIDEO3_CHECKPOINT:-$WEIGHTS/internvideo3-8b-instruct-c4602918}
    ENCODER_FAMILY=internvideo3
    ENCODER_MODEL_NAME=internvideo3_8b_vision_premerger
    ENCODER_CHECKPOINT_KEY=model.visual
    DEFAULT_TEXT_EMBEDDING_CACHE=/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128
    DEFAULT_T5_LEN=128
    ;;
  *)
    echo "Unknown latent encoder: $ENCODER" >&2
    exit 64
    ;;
esac

DEFAULT_TEXT_EMBEDDING_CACHE=${DEFAULT_TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
DEFAULT_T5_LEN=${DEFAULT_T5_LEN:-32}

export POLICY_PYTHON=${POLICY_PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
export LIBERO_PYTHON=${LIBERO_PYTHON:-/data/zhangyang/miniconda3/envs/libero/bin/python}
export POLICY_SERVER_SCRIPT=examples/libero/serve_latent_policy.py
export VJEPA2_ENCODER_CHECKPOINT ENCODER_FAMILY ENCODER_MODEL_NAME ENCODER_CHECKPOINT_KEY
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-$DEFAULT_TEXT_EMBEDDING_CACHE}
export POLICY_T5_LEN=${POLICY_T5_LEN:-$DEFAULT_T5_LEN}
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
export EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_r16_50trials_seed7}

exec bash "$ROOT/examples/libero/evaluate_policy.sh" "$RUN_DIR" "$CHECKPOINT"
