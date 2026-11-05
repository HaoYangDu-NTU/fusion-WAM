#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA}
export VJEPA2_ENCODER_CHECKPOINT=${VJEPA2_ENCODER_CHECKPOINT:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints/vitg.pt}
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
export DATASET_STATS=${DATASET_STATS:-$ROOT/runs/full_route2_action_cross_correctedrope_pred24x1024_gbs128_step21360_bs32ga1_acb4_nw8_seed7/dataset_stats.json}
export TRAIN_SCRIPT=scripts/route2/clean_context/train.py
export CROP_SIZE=256
export VIEW_LAYOUT=quadrant
export ENCODER_INTERPOLATE_ROPE=0
export OUTDIR=${OUTDIR:-$ROOT/runs/route2_clean_context_reproduction_seed${SEED:-7}}
export NUM_GPUS=${NUM_GPUS:-4}
export BATCH_SIZE=${BATCH_SIZE:-32}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_STEPS=${MAX_STEPS:-21360}
export SEED=${SEED:-7}

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != 128 || MAX_STEPS != 21360 || SEED != 7 )); then
  echo "Route 2 reproduction requires global batch 128, 21,360 steps, and seed 7" >&2
  exit 64
fi

exec bash "$ROOT/scripts/train_vjepa_policy.sh"
