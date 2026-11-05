#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/scripts/ablation_runtime.sh"
verify_source_snapshot "$ROOT"
ENCODER=${1:?usage: run.sh ENCODER [--check|--smoke]}
MODE=${2:-run}
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub

case "$ENCODER" in
  dinov2_vitl)
    LATENT_ENCODER=dinov2
    LATENT_CHECKPOINT=${DINOV2_CHECKPOINT:-$WEIGHTS/dinov2-large-47b73eefe95e}
    DEFAULT_ACB=7
    DEFAULT_IMAGE_MICROBATCH_SIZE=320
    DEFAULT_VIDEO_MICROBATCH_SIZE=32
    EXPECTED_SIZE=1217522888
    EXPECTED_SHA256=399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23
    ;;
  dinov3_vitl)
    LATENT_ENCODER=dinov3
    LATENT_CHECKPOINT=${DINOV3_CHECKPOINT:-$WEIGHTS/dinov3-vitl16-ea8dc2863c51}
    DEFAULT_ACB=7
    DEFAULT_IMAGE_MICROBATCH_SIZE=320
    DEFAULT_VIDEO_MICROBATCH_SIZE=32
    EXPECTED_SIZE=1212559808
    EXPECTED_SHA256=dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179
    ;;
  wan22_vae38)
    LATENT_ENCODER=wan2_2
    LATENT_CHECKPOINT=${WAN22_CHECKPOINT:-$WEIGHTS/wan2.2-ti2v-5b-diffusers-b8fff7315c76/vae/diffusion_pytorch_model.safetensors}
    DEFAULT_ACB=0
    DEFAULT_IMAGE_MICROBATCH_SIZE=32
    DEFAULT_VIDEO_MICROBATCH_SIZE=32
    EXPECTED_SIZE=2818777808
    EXPECTED_SHA256=62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b
    ;;
  internvideo3_8b)
    LATENT_ENCODER=internvideo3
    LATENT_CHECKPOINT=${INTERNVIDEO3_CHECKPOINT:-$WEIGHTS/internvideo3-8b-instruct-c4602918}
    DEFAULT_ACB=8
    DEFAULT_IMAGE_MICROBATCH_SIZE=32
    DEFAULT_VIDEO_MICROBATCH_SIZE=8
    DEFAULT_NUM_GPUS=8
    DEFAULT_BATCH_SIZE=16
    DEFAULT_TEXT_EMBEDDING_CACHE=/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128
    DEFAULT_CONTEXT_LEN=128
    DEFAULT_DATASET_STATS=$ROOT/runs/full_encoder_ablation_vjepa2_vitl_independent224_t5len128_gbs128_step21360_seed7_prtsft/dataset_stats.json
    ;;
  *)
    echo "Unknown latent encoder: $ENCODER" >&2
    exit 64
    ;;
esac

DEFAULT_NUM_GPUS=${DEFAULT_NUM_GPUS:-4}
DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE:-32}
DEFAULT_TEXT_EMBEDDING_CACHE=${DEFAULT_TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
DEFAULT_CONTEXT_LEN=${DEFAULT_CONTEXT_LEN:-32}
DEFAULT_DATASET_STATS=${DEFAULT_DATASET_STATS:-$ROOT/runs/full_encoder_ablation_vjepa2_vitl_independent224_gbs128_step21360_seed7/dataset_stats.json}

if [[ "$LATENT_ENCODER" == dinov2 || "$LATENT_ENCODER" == dinov3 ]]; then
  if [[ ! -s "$LATENT_CHECKPOINT/config.json" || ! -s "$LATENT_CHECKPOINT/model.safetensors" ]]; then
    echo "Incomplete local DINO checkpoint: $LATENT_CHECKPOINT" >&2
    exit 1
  fi
  CHECKPOINT_FILE=$LATENT_CHECKPOINT/model.safetensors
elif [[ "$LATENT_ENCODER" == internvideo3 ]]; then
  for required in config.json configuration_internvideo3.py modeling_internvideo3.py \
    model-vision-0001-others-save_rank0.safetensors \
    model-projector-0001-others-save_rank0.safetensors; do
    if [[ ! -s "$LATENT_CHECKPOINT/$required" ]]; then
      echo "Incomplete local InternVideo3 checkpoint: $LATENT_CHECKPOINT/$required" >&2
      exit 1
    fi
  done
  CHECKPOINT_FILE=
elif [[ ! -s "$LATENT_CHECKPOINT" ]]; then
  echo "Missing Wan2.2 VAE38 checkpoint: $LATENT_CHECKPOINT" >&2
  exit 1
else
  CHECKPOINT_FILE=$LATENT_CHECKPOINT
fi
if [[ -n "$CHECKPOINT_FILE" ]]; then
  observed_size=$(stat -c %s "$CHECKPOINT_FILE")
  if [[ "$observed_size" != "$EXPECTED_SIZE" ]]; then
    echo "Checkpoint size mismatch: $CHECKPOINT_FILE ($observed_size != $EXPECTED_SIZE)" >&2
    exit 1
  fi
  observed_sha256=$(sha256sum "$CHECKPOINT_FILE" | awk '{print $1}')
  if [[ "$observed_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "Checkpoint SHA256 mismatch: $CHECKPOINT_FILE" >&2
    exit 1
  fi
fi

if [[ "$MODE" == --check ]]; then
  printf '%s latent_encoder=%s checkpoint=%s\n' \
    "$ENCODER" "$LATENT_ENCODER" "$LATENT_CHECKPOINT"
  exit 0
fi
if [[ "$MODE" != run && "$MODE" != --smoke ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 64
fi
if [[ "$MODE" == --smoke && -z "${OUTDIR:-}" ]]; then
  echo "OUTDIR is required for --smoke" >&2
  exit 64
fi

export PYTHON=/data/zhangyang/miniconda3/envs/prts-ft/bin/python
export PYTHONDONTWRITEBYTECODE=1
export LIBERO_DATA_ROOT=/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-$DEFAULT_TEXT_EMBEDDING_CACHE}
export DATASET_STATS=${DATASET_STATS:-$DEFAULT_DATASET_STATS}
export TRAIN_SCRIPT=scripts/latent_space_ablation/train.py
export VJEPA2_ENCODER_CHECKPOINT=$WEIGHTS/checkpoints/vitl.pt
export ENCODER_FAMILY=vjepa2
export ENCODER_MODEL_NAME=vit_large
export ENCODER_CHECKPOINT_KEY=target_encoder
export LATENT_ENCODER LATENT_CHECKPOINT
export IMAGE_MICROBATCH_SIZE=${IMAGE_MICROBATCH_SIZE:-$DEFAULT_IMAGE_MICROBATCH_SIZE}
export VIDEO_MICROBATCH_SIZE=${VIDEO_MICROBATCH_SIZE:-$DEFAULT_VIDEO_MICROBATCH_SIZE}
export FASTWAM_SRC_DIR=${FASTWAM_SRC_DIR:-/data/zhangyang/workspace/FastWAM/src}
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
export GRAD_ACCUM=1
export CONTEXT_LEN=${CONTEXT_LEN:-$DEFAULT_CONTEXT_LEN}
export SEED=7
export ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-$DEFAULT_ACB}
export BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
export OUTDIR=${OUTDIR:-$ROOT/runs/latent_${ENCODER}_independent224_t5len${CONTEXT_LEN}_gbs128_step21360_seed7}

if [[ "$MODE" == --smoke ]]; then
  export NUM_GPUS=${SMOKE_NUM_GPUS:-$DEFAULT_NUM_GPUS}
  export MAX_STEPS=${SMOKE_MAX_STEPS:-10}
  export SAVE_EVERY=0
else
  export NUM_GPUS=${NUM_GPUS:-$DEFAULT_NUM_GPUS}
  export MAX_STEPS=21360
  export SAVE_EVERY=4000
fi

printf 'latent_runtime encoder=%s mode=%s acb=%s image_microbatch=%s video_microbatch=%s\n' \
  "$ENCODER" "$MODE" "$ACTIVATION_CHECKPOINTING_BLOCKS" \
  "$IMAGE_MICROBATCH_SIZE" "$VIDEO_MICROBATCH_SIZE"

exec bash "$ROOT/scripts/train_vjepa_policy.sh"
