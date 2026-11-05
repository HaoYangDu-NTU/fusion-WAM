#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
DATASET_DIR=${DATASET_DIR:-/data-team/rhodes_team/huggingface_cache/rhodes_lerobot/TeleData/Tianji/clean_the_noodle_bowl}
TEXT_CACHE=${TEXT_CACHE:-/data/zhangyang/bench_datasets/tianji_clean_noodle_t5_cache}
DATASET_STATS=${DATASET_STATS:-/data/zhangyang/bench_datasets/tianji_clean_noodle_t5_cache/dataset_stats_quantile.json}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints/vitl.pt}
PREDICTOR_INIT=${PREDICTOR_INIT:-/data/zhangyang/workspace/VJEPA-Policy-revised/runs/droid_predictor_pretraining_prts_5fps_bs192_100k/checkpoint_step100000.pt}
OUTDIR=${OUTDIR:-$ROOT/runs/teledata_tianji_clean_noodle_vjepa2_vitl_droidinit_4view256_t5len128_maxstate48_gbs128_step100000_seed7}

NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-500}
ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-16}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_EVERY=${SAVE_EVERY:-10000}
LOG_EVERY=${LOG_EVERY:-10}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29631}

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != 128 )); then
  echo "Expected global batch size 128, got $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))" >&2
  exit 64
fi
for path in "$DATASET_DIR/meta/info.json" "$TEXT_CACHE" "$DATASET_STATS" \
  "$ENCODER_CHECKPOINT" "$PREDICTOR_INIT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required training input: $path" >&2
    exit 66
  fi
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p "$OUTDIR"

echo "run=$OUTDIR"
echo "global_batch_size=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"

"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 --dynamo_backend no \
  scripts/teledata/train.py \
    --dataset-dirs "$DATASET_DIR" \
    --dataset-stats "$DATASET_STATS" \
    --text-cache-dir "$TEXT_CACHE" \
    --instruction-field task \
    --views \
      observation.images.fourth \
      observation.images.head \
      observation.images.left \
      observation.images.right \
    --view-layout independent \
    --video-backend torchcodec \
    --checkpoint "$ENCODER_CHECKPOINT" \
    --encoder-family vjepa2 \
    --model-name vit_large \
    --encoder-checkpoint-key target_encoder \
    --predictor-init "$PREDICTOR_INIT" \
    --predictor-view-map 0 0 0 0 \
    --crop-size 256 \
    --image-resize-mode letterbox \
    --patch-size 16 \
    --tubelet-size 2 \
    --num-frames 33 \
    --past-frames 4 \
    --video-frame-stride 4 \
    --context-tubelets 1 \
    --require-full-future-window \
    --context-len 128 \
    --lang-dim 4096 \
    --action-normalization QUANTILE \
    --state-normalization QUANTILE \
    --no-clip-normalized \
    --action-chunk-size 32 \
    --action-dim 16 \
    --proprio-dim 16 \
    --max-state-dim 48 \
    --proprio-encoding identity \
    --action-hidden-size 512 \
    --action-num-layers 24 \
    --condition-num-heads 8 \
    --action-loss-weight 1.0 \
    --action-num-inference-steps 10 \
    --pred-depth 24 \
    --pred-embed-dim 1024 \
    --pred-num-heads 16 \
    --num-mask-tokens 10 \
    --use-activation-checkpointing \
    --activation-checkpointing-blocks "$ACTIVATION_CHECKPOINTING_BLOCKS" \
    --predictor-rope-frequency-pairing corrected \
    --encoder-interpolate-rope \
    --seed 7 \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --recycle-workers-every "$RECYCLE_WORKERS_EVERY" \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --max-steps "$MAX_STEPS" \
    --num-epochs 100 \
    --mixed-precision bf16 \
    --log-every "$LOG_EVERY" \
    --save-every "$SAVE_EVERY" \
    --output-dir "$OUTDIR" \
    --wandb-project vjepa_policy_real \
    --wandb-name "$(basename "$OUTDIR")" \
  2>&1 | tee -a "$OUTDIR/train.log"
