#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

EXPERIMENT=${1:?usage: run.sh DINOV2_NATIVE16|WAN22_STRIDE2_T5 [--check|--smoke]}
MODE=${2:-run}
WEIGHTS=${WEIGHTS:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub}

case "$EXPERIMENT" in
  dinov2_native16|DINOV2_NATIVE16)
    GEOMETRY_EXPERIMENT=dinov2_native16
    LATENT_CHECKPOINT=${DINOV2_CHECKPOINT:-$WEIGHTS/dinov2-large-47b73eefe95e}
    CHECKPOINT_FILE="$LATENT_CHECKPOINT/model.safetensors"
    EXPECTED_SIZE=1217522888
    EXPECTED_SHA256=399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23
    DEFAULT_ACB=12
    DEFAULT_IMAGE_MICROBATCH_SIZE=64
    DEFAULT_VIDEO_MICROBATCH_SIZE=32
    PAST_FRAMES=4
    VIDEO_STRIDE=4
    ;;
  wan22_stride2_t5|WAN22_STRIDE2_T5)
    GEOMETRY_EXPERIMENT=wan22_stride2_t5
    LATENT_CHECKPOINT=${WAN22_CHECKPOINT:-$WEIGHTS/wan2.2-ti2v-5b-diffusers-b8fff7315c76/vae/diffusion_pytorch_model.safetensors}
    CHECKPOINT_FILE="$LATENT_CHECKPOINT"
    EXPECTED_SIZE=2818777808
    EXPECTED_SHA256=62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b
    DEFAULT_ACB=8
    DEFAULT_IMAGE_MICROBATCH_SIZE=32
    DEFAULT_VIDEO_MICROBATCH_SIZE=32
    PAST_FRAMES=0
    VIDEO_STRIDE=2
    ;;
  *)
    echo "Unknown geometry experiment: $EXPERIMENT" >&2
    exit 64
    ;;
esac

if [[ "$GEOMETRY_EXPERIMENT" == dinov2_native16 ]]; then
  if [[ ! -s "$LATENT_CHECKPOINT/config.json" || ! -s "$CHECKPOINT_FILE" ]]; then
    echo "Incomplete local DINOv2 checkpoint: $LATENT_CHECKPOINT" >&2
    exit 1
  fi
else
  if [[ ! -s "$CHECKPOINT_FILE" ]]; then
    echo "Missing Wan2.2 VAE38 checkpoint: $CHECKPOINT_FILE" >&2
    exit 1
  fi
fi
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

if [[ "$MODE" == --check ]]; then
  printf 'geometry_experiment=%s latent_checkpoint=%s raw_offsets=' \
    "$GEOMETRY_EXPERIMENT" "$LATENT_CHECKPOINT"
  if [[ "$GEOMETRY_EXPERIMENT" == dinov2_native16 ]]; then
    printf '[%s]\n' "$(seq -s, -4 4 32)"
  else
    printf '[%s]\n' "$(seq -s, 0 2 32)"
  fi
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

export PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"
export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA}
export TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl}
export DATASET_STATS=${DATASET_STATS:-$ROOT/runs/full_encoder_ablation_vjepa2_vitl_independent224_gbs128_step21360_seed7/dataset_stats.json}
export VJEPA2_ENCODER_CHECKPOINT=${VJEPA2_ENCODER_CHECKPOINT:-$WEIGHTS/checkpoints/vitl.pt}
export FASTWAM_SRC_DIR=${FASTWAM_SRC_DIR:-/data/zhangyang/workspace/FastWAM/src}

export NUM_GPUS=${NUM_GPUS:-4}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29561}
export BATCH_SIZE=${BATCH_SIZE:-32}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_STEPS=${MAX_STEPS:-21360}
export SAVE_EVERY=${SAVE_EVERY:-4000}
export LOG_EVERY=${LOG_EVERY:-10}
export SEED=${SEED:-7}
export PRED_DEPTH=${PRED_DEPTH:-24}
export PRED_EMBED_DIM=${PRED_EMBED_DIM:-1024}
export PRED_NUM_HEADS=${PRED_NUM_HEADS:-16}
export ACTION_HIDDEN_SIZE=${ACTION_HIDDEN_SIZE:-512}
export CONDITION_NUM_HEADS=${CONDITION_NUM_HEADS:-8}
export NUM_WORKERS=${NUM_WORKERS:-8}
export PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
export RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-500}
export ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-$DEFAULT_ACB}
export ACTION_ACTIVATION_CHECKPOINTING_BLOCKS=${ACTION_ACTIVATION_CHECKPOINTING_BLOCKS:-1}
export ACTION_MICROBATCH_SIZE=${ACTION_MICROBATCH_SIZE:-$BATCH_SIZE}
export IMAGE_MICROBATCH_SIZE=${IMAGE_MICROBATCH_SIZE:-$DEFAULT_IMAGE_MICROBATCH_SIZE}
export VIDEO_MICROBATCH_SIZE=${VIDEO_MICROBATCH_SIZE:-$DEFAULT_VIDEO_MICROBATCH_SIZE}
export MIXED_PRECISION=${MIXED_PRECISION:-bf16}
export OUTDIR=${OUTDIR:-$ROOT/runs/latent_${GEOMETRY_EXPERIMENT}_independent224_gbs128_step21360_seed7}

if [[ "$MODE" == --smoke ]]; then
  NUM_GPUS=${SMOKE_NUM_GPUS:-1}
  MAX_STEPS=${SMOKE_MAX_STEPS:-10}
  SAVE_EVERY=0
fi

DATASETS=(
  "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot"
  "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot"
  "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot"
  "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot"
)
STATS_ARGS=()
if [[ -n "$DATASET_STATS" ]]; then
  STATS_ARGS+=(--dataset-stats "$DATASET_STATS")
fi

mkdir -p "$OUTDIR"
if [[ -d "$OUTDIR" ]] && find "$OUTDIR" -mindepth 1 -print -quit | grep -q . && [[ "${ALLOW_NONEMPTY_OUTDIR:-0}" != 1 ]]; then
  echo "Refusing to append to non-empty output directory: $OUTDIR" >&2
  echo "Set ALLOW_NONEMPTY_OUTDIR=1 only when intentionally resuming." >&2
  exit 1
fi

echo "geometry_runtime experiment=$GEOMETRY_EXPERIMENT raw_frames=$([[ "$GEOMETRY_EXPERIMENT" == dinov2_native16 ]] && echo 10 || echo 17) gpus=$NUM_GPUS acb=$ACTIVATION_CHECKPOINTING_BLOCKS action_acb=$ACTION_ACTIVATION_CHECKPOINTING_BLOCKS action_microbatch=$ACTION_MICROBATCH_SIZE image_microbatch=$IMAGE_MICROBATCH_SIZE video_microbatch=$VIDEO_MICROBATCH_SIZE out=$OUTDIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROPE_ARGS=(--predictor-rope-frequency-pairing corrected --encoder-interpolate-rope)
"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision "$MIXED_PRECISION" --dynamo_backend no \
  scripts/latent_geometry_ablation/train.py \
    --geometry-experiment "$GEOMETRY_EXPERIMENT" \
    --latent-checkpoint "$LATENT_CHECKPOINT" \
    --fastwam-src-dir "$FASTWAM_SRC_DIR" \
    --dataset-dirs "${DATASETS[@]}" \
    "${STATS_ARGS[@]}" --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
    --lang-dim 4096 --checkpoint "$VJEPA2_ENCODER_CHECKPOINT" \
    --encoder-family vjepa2 --model-name vit_large \
    --encoder-checkpoint-key target_encoder \
    --crop-size 224 --patch-size 16 --tubelet-size 2 \
    --num-frames 33 --past-frames "$PAST_FRAMES" \
    --video-frame-stride "$VIDEO_STRIDE" --context-tubelets 1 \
    --view-layout independent \
    --action-normalization QUANTILE --state-normalization QUANTILE \
    --action-chunk-size 32 --action-dim 7 --proprio-dim 8 \
    --action-hidden-size "$ACTION_HIDDEN_SIZE" --action-num-layers "$PRED_DEPTH" \
    --condition-num-heads "$CONDITION_NUM_HEADS" \
    --action-loss-weight 1.0 --action-num-inference-steps 10 \
    --pred-depth "$PRED_DEPTH" --pred-embed-dim "$PRED_EMBED_DIM" \
    --pred-num-heads "$PRED_NUM_HEADS" --num-mask-tokens 10 \
    --activation-checkpointing-blocks "$ACTIVATION_CHECKPOINTING_BLOCKS" \
    --action-activation-checkpointing-blocks "$ACTION_ACTIVATION_CHECKPOINTING_BLOCKS" \
    --action-microbatch-size "$ACTION_MICROBATCH_SIZE" \
    "${ROPE_ARGS[@]}" --seed "$SEED" --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
    --recycle-workers-every "$RECYCLE_WORKERS_EVERY" \
    --lr 1e-4 --weight-decay 0.01 --max-steps "$MAX_STEPS" --num-epochs 10 \
    --mixed-precision "$MIXED_PRECISION" --log-every "$LOG_EVERY" --save-every "$SAVE_EVERY" \
    --output-dir "$OUTDIR" --wandb-name "$(basename "$OUTDIR")" \
  2>&1 | tee -a "$OUTDIR/train.log"

echo "[done] checkpoint: $OUTDIR/checkpoint_step$(printf '%06d' "$MAX_STEPS").pt"
