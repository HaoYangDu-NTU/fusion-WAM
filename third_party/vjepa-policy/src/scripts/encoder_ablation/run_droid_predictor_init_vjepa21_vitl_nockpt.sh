#!/usr/bin/env bash
set -euo pipefail

# LIBERO fine-tuning initialized from the completed DROID V-JEPA2.1 ViT-L
# predictor pre-training run.  This is predictor initialization only: no
# optimizer, scheduler, step, or Action Expert state is restored.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA}
TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128}
DATASET_STATS=${DATASET_STATS:-$ROOT/runs/droid_init_full_encoder_vjepa2_vitl_independent224_t5len128_maxstate48_gbs128_step21360_seed7/dataset_stats.json}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}
PREDICTOR_INIT=${PREDICTOR_INIT:-/data/zhangyang/workspace/VJEPA-Policy-revised/runs/droid_predictor_pretraining_vjepa2_1_vitl_maxviews4_prts_5fps_bs192_100k/checkpoint_step100000.pt}
OUTDIR=${OUTDIR:-$ROOT/runs/droid_init_full_encoder_vjepa2_1_vitl_independent224_t5len128_maxstate48_gbs128_step21360_seed7_nockpt}

NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29701}
MAX_STEPS=${MAX_STEPS:-21360}
SAVE_EVERY=${SAVE_EVERY:-4000}
SEED=${SEED:-7}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-500}

if (( NUM_GPUS * BATCH_SIZE != GLOBAL_BATCH_SIZE )); then
  echo "Expected global batch size $GLOBAL_BATCH_SIZE, got $((NUM_GPUS * BATCH_SIZE))" >&2
  exit 64
fi
for path in \
  "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot/meta/info.json" \
  "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot/meta/info.json" \
  "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot/meta/info.json" \
  "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot/meta/info.json" \
  "$TEXT_EMBEDDING_CACHE" "$DATASET_STATS" "$ENCODER_CHECKPOINT" "$PREDICTOR_INIT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required training input: $path" >&2
    exit 66
  fi
done

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VJEPA_DDP_FAST=${VJEPA_DDP_FAST:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
mkdir -p "$OUTDIR"

echo "encoder=vjepa2_1/vit_large key=ema_encoder checkpoint=$ENCODER_CHECKPOINT"
echo "predictor_init=$PREDICTOR_INIT view_map=0,1"
echo "global_batch_size=$GLOBAL_BATCH_SIZE crop=224 context_len=128 max_state_dim=48 steps=$MAX_STEPS"
echo "activation_checkpointing_blocks=0 use_activation_checkpointing=false"

exec "$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 --dynamo_backend no \
  scripts/train_vjepa_policy.py \
    --dataset-dirs \
      "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot" \
      "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot" \
      "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot" \
      "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot" \
    --dataset-stats "$DATASET_STATS" \
    --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
    --lang-dim 4096 --checkpoint "$ENCODER_CHECKPOINT" \
    --encoder-family vjepa2_1 --model-name vit_large \
    --encoder-checkpoint-key ema_encoder --encoder-interpolate-rope \
    --crop-size 224 --image-resize-mode stretch --patch-size 16 --tubelet-size 2 \
    --num-frames 33 --past-frames 4 --video-frame-stride 4 --context-tubelets 1 \
    --context-len 128 --view-layout independent \
    --action-normalization QUANTILE --state-normalization QUANTILE \
    --action-chunk-size 32 --action-dim 7 --proprio-dim 8 --max-state-dim 48 \
    --action-hidden-size 512 --action-num-layers 24 --condition-num-heads 8 \
    --action-loss-weight 1.0 --action-num-inference-steps 10 \
    --pred-depth 24 --pred-embed-dim 1024 --pred-num-heads 16 --num-mask-tokens 10 \
    --activation-checkpointing-blocks 0 \
    --predictor-rope-frequency-pairing corrected \
    --predictor-view-map 0 1 \
    --seed "$SEED" --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps 1 \
    --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
    --recycle-workers-every "$RECYCLE_WORKERS_EVERY" \
    --lr 1e-4 --weight-decay 0.01 --max-steps "$MAX_STEPS" --num-epochs 10 \
    --mixed-precision bf16 --log-every 10 --save-every "$SAVE_EVERY" \
    --output-dir "$OUTDIR" --wandb-name "$(basename "$OUTDIR")" \
    --predictor-init "$PREDICTOR_INIT" \
  2>&1 | tee -a "$OUTDIR/train.log"

echo "[done] checkpoint: $OUTDIR/checkpoint_step$(printf '%06d' "$MAX_STEPS").pt"
