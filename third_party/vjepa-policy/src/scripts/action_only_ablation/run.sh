#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/scripts/ablation_runtime.sh"
verify_source_snapshot "$ROOT"
ENCODER=${1:?usage: run.sh ENCODER [--check]}
MODE=${2:-run}
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints

case "$ENCODER" in
  vjepa2_vitl)
    ENCODER_FAMILY=vjepa2
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=target_encoder
    VJEPA2_ENCODER_CHECKPOINT=$WEIGHTS/vitl.pt
    EXPECTED_SIZE=5127726842
    EXPECTED_SHA256=5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012
    ;;
  vjepa2_1_vitl)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=ema_encoder
    VJEPA2_ENCODER_CHECKPOINT=$WEIGHTS/vjepa2_1_vitl_dist_vitG_384.pt
    EXPECTED_SIZE=5151198524
    EXPECTED_SHA256=7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6
    ;;
  *)
    echo "Unknown action-only encoder: $ENCODER" >&2
    exit 64
    ;;
esac

if [[ ! -s "$VJEPA2_ENCODER_CHECKPOINT" ]]; then
  echo "Missing encoder checkpoint: $VJEPA2_ENCODER_CHECKPOINT" >&2
  exit 1
fi
observed_size=$(stat -c %s "$VJEPA2_ENCODER_CHECKPOINT")
if [[ "$observed_size" != "$EXPECTED_SIZE" ]]; then
  echo "Checkpoint size mismatch: $VJEPA2_ENCODER_CHECKPOINT" >&2
  exit 1
fi
observed_sha256=$(sha256sum "$VJEPA2_ENCODER_CHECKPOINT" | awk '{print $1}')
if [[ "$observed_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "Checkpoint SHA256 mismatch: $VJEPA2_ENCODER_CHECKPOINT" >&2
  exit 1
fi
if [[ "$MODE" == --check ]]; then
  printf '%s checkpoint=%s sha256=%s\n' \
    "$ENCODER" "$VJEPA2_ENCODER_CHECKPOINT" "$observed_sha256"
  exit 0
fi
if [[ "$MODE" != run ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 64
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHON=/data/zhangyang/miniconda3/envs/prts-ft/bin/python
export LIBERO_DATA_ROOT=/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA
export TEXT_EMBEDDING_CACHE=/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl
export DATASET_STATS=$ROOT/runs/full_route2_action_cross_correctedrope_pred24x1024_gbs128_step21360_bs32ga1_acb4_nw8_seed7/dataset_stats.json
export TRAIN_SCRIPT=scripts/action_only_ablation/train.py
export VJEPA2_ENCODER_CHECKPOINT ENCODER_FAMILY ENCODER_MODEL_NAME ENCODER_CHECKPOINT_KEY
export CROP_SIZE=224
export VIEW_LAYOUT=independent
export ENCODER_INTERPOLATE_ROPE=1
export PREDICTOR_ROPE_FREQUENCY_PAIRING=corrected
export PRED_DEPTH=24
export PRED_EMBED_DIM=1024
export PRED_NUM_HEADS=16
export ACTION_HIDDEN_SIZE=512
export CONDITION_NUM_HEADS=8
export MIXED_PRECISION=bf16
export NUM_WORKERS=8
export PREFETCH_FACTOR=2
export RECYCLE_WORKERS_EVERY=500
export NUM_GPUS=4
export BATCH_SIZE=32
export GRAD_ACCUM=1
export MAX_STEPS=21360
export SEED=7
export SAVE_EVERY=4000
export ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-7}
export OUTDIR=${OUTDIR:-$ROOT/runs/action_only_${ENCODER}_independent224_gbs128_step21360_seed7}

printf 'action_runtime encoder=%s acb=%s checkpoint_sha256=%s\n' \
  "$ENCODER" "$ACTIVATION_CHECKPOINTING_BLOCKS" "$observed_sha256"

exec bash "$ROOT/scripts/train_vjepa_policy.sh"
