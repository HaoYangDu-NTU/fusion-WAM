#!/usr/bin/env bash
set -euo pipefail

# Exact 96.9-LIBERO training contract, with only the frozen encoder replaced.
# The caller supplies one variant; this script intentionally runs one process
# group at a time so checkpoints remain independent and comparable.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints
VARIANT=${1:?usage: run_exact_t5len128.sh vjepa2_vitg|vjepa2_1_vitl|vjepa2_1_vitg}

case "$VARIANT" in
  vjepa2_vitg)
    ENCODER_FAMILY=vjepa2
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=target_encoder
    ENCODER_CHECKPOINT="$WEIGHTS/vitg.pt"
    LABEL=vjepa2_vitg
    ;;
  vjepa2_1_vitl)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=ema_encoder
    ENCODER_CHECKPOINT="$WEIGHTS/vjepa2_1_vitl_dist_vitG_384.pt"
    LABEL=vjepa2_1_vitl
    ;;
  vjepa2_1_vitg)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=target_encoder
    ENCODER_CHECKPOINT="$WEIGHTS/vjepa2_1_vitg_384.pt"
    LABEL=vjepa2_1_vitg
    ;;
  *)
    echo "Unknown encoder variant: $VARIANT" >&2
    exit 64
    ;;
esac

export PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA}
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128}
export DATASET_STATS=${DATASET_STATS:-$ROOT/runs/full_encoder_ablation_vjepa2_vitl_independent224_t5len128_gbs128_step21360_seed7_prtsft/dataset_stats.json}
export VJEPA2_ENCODER_CHECKPOINT="$ENCODER_CHECKPOINT"
export ENCODER_FAMILY ENCODER_MODEL_NAME ENCODER_CHECKPOINT_KEY
export CROP_SIZE=${CROP_SIZE:-224}
export VIEW_LAYOUT=${VIEW_LAYOUT:-independent}
export CONTEXT_LEN=${CONTEXT_LEN:-128}
export ENCODER_INTERPOLATE_ROPE=${ENCODER_INTERPOLATE_ROPE:-1}
export ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-8}
export RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-500}
export NUM_GPUS=${NUM_GPUS:-8}
export BATCH_SIZE=${BATCH_SIZE:-16}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_STEPS=${MAX_STEPS:-21360}
export SAVE_EVERY=${SAVE_EVERY:-4000}
export SEED=${SEED:-7}
export NUM_WORKERS=${NUM_WORKERS:-8}
export PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29700}
export OUTDIR=${OUTDIR:-$ROOT/runs/full_encoder_ablation_${LABEL}_independent224_t5len128_gbs128_step21360_seed7_8gpu}

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != 128 )); then
  echo "Expected global batch size 128, got $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))" >&2
  exit 64
fi
if (( CONTEXT_LEN != 128 || CROP_SIZE != 224 || SEED != 7 || MAX_STEPS != 21360 )); then
  echo "This launcher requires context_len=128, crop=224, seed=7, max_steps=21360" >&2
  exit 64
fi
for path in "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot/meta/info.json" \
  "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot/meta/info.json" \
  "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot/meta/info.json" \
  "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot/meta/info.json" \
  "$TEXT_EMBEDDING_CACHE" "$DATASET_STATS" "$VJEPA2_ENCODER_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required training input: $path" >&2
    exit 66
  fi
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VJEPA_DDP_FAST=${VJEPA_DDP_FAST:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p "$OUTDIR"

echo "variant=$VARIANT"
echo "encoder=$ENCODER_FAMILY/$ENCODER_MODEL_NAME key=$ENCODER_CHECKPOINT_KEY checkpoint=$VJEPA2_ENCODER_CHECKPOINT"
echo "global_batch_size=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM)) context_len=$CONTEXT_LEN crop=$CROP_SIZE steps=$MAX_STEPS"

exec bash "$ROOT/scripts/train_vjepa_policy.sh"
