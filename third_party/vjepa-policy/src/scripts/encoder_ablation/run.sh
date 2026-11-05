#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ENCODER=${1:?usage: run.sh ENCODER}
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints

case "$ENCODER" in
  vjepa2_vitl)
    ENCODER_FAMILY=vjepa2
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=target_encoder
    VJEPA2_ENCODER_CHECKPOINT=$WEIGHTS/vitl.pt
    ;;
  vjepa2_1_vitl)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=ema_encoder
    VJEPA2_ENCODER_CHECKPOINT=$WEIGHTS/vjepa2_1_vitl_dist_vitG_384.pt
    ;;
  vjepa2_1_vitg)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=target_encoder
    VJEPA2_ENCODER_CHECKPOINT=$WEIGHTS/vjepa2_1_vitg_384.pt
    ;;
  *)
    echo "Unknown encoder: $ENCODER" >&2
    exit 64
    ;;
esac

export PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA}
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
export DATASET_STATS=${DATASET_STATS:-$ROOT/runs/full_route2_action_cross_correctedrope_pred24x1024_gbs128_step21360_bs32ga1_acb4_nw8_seed7/dataset_stats.json}
export TRAIN_SCRIPT=scripts/train_vjepa_policy.py
export VJEPA2_ENCODER_CHECKPOINT ENCODER_FAMILY ENCODER_MODEL_NAME ENCODER_CHECKPOINT_KEY
export CROP_SIZE=224
export VIEW_LAYOUT=independent
export ENCODER_INTERPOLATE_ROPE=1
export RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-500}
export NUM_GPUS=${NUM_GPUS:-4}
export BATCH_SIZE=${BATCH_SIZE:-32}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_STEPS=${MAX_STEPS:-21360}
export SEED=${SEED:-7}
export SAVE_EVERY=${SAVE_EVERY:-4000}
export OUTDIR=${OUTDIR:-$ROOT/runs/full_encoder_ablation_${ENCODER}_independent224_gbs128_step21360_seed7}

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != 128 || SEED != 7 )); then
  echo "Encoder comparison requires global batch 128 and training seed 7" >&2
  exit 64
fi

exec bash "$ROOT/scripts/train_vjepa_policy.sh"
