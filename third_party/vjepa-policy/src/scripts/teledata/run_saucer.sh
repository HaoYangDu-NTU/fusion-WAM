#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

MODE=${MODE:-scratch}
case "$MODE" in
  scratch)
    INIT_ARGS=()
    LABEL=scratch
    ;;
  droidinit|droid)
    PREDICTOR_INIT=${PREDICTOR_INIT:-/data/zhangyang/workspace/VJEPA-Policy-revised/runs/droid_predictor_pretraining_prts_5fps_bs192_100k/checkpoint_step100000.pt}
    INIT_ARGS=(--predictor-init "$PREDICTOR_INIT" --predictor-view-map 0 0 0 0)
    LABEL=droidinit
    ;;
  *)
    echo "MODE must be scratch or droidinit, got $MODE" >&2
    exit 64
    ;;
esac

PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
DATASET_DIR=${DATASET_DIR:-/data-team/rhodes_team/huggingface_cache/rhodes_lerobot/TeleData/Tianji/saucer0821_no_idle}
TEXT_CACHE=${TEXT_CACHE:-/data/zhangyang/bench_datasets/saucer0821_no_idle_t5_cache}
DATASET_STATS=${DATASET_STATS:-$TEXT_CACHE/dataset_stats_quantile.json}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints/vitl.pt}
OUTDIR=${OUTDIR:-$ROOT/runs/teledata_tianji_saucer0821_no_idle_vjepa2_vitl_${LABEL}_4view256_letterbox_t5len128_maxstate48_gbs64_step40000_seed7_fast}

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-1000}
MAX_STEPS=${MAX_STEPS:-40000}
SAVE_EVERY=${SAVE_EVERY:-10000}
LOG_EVERY=${LOG_EVERY:-10}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29641}
RESUME=${RESUME:-}

# The scratch leg is launched only after the resumed droidinit leg finishes.
# It has no competing training job, so use the available CPU headroom to keep
# all ranks fed; the worker recycle interval remains in force for torchcodec's
# per-process decode leak.
if [[ "$MODE" == "scratch" ]]; then
  NUM_WORKERS=${SCRATCH_NUM_WORKERS:-16}
  LOG_EVERY=${SCRATCH_LOG_EVERY:-100}
fi

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != 64 )); then
  echo "Expected global batch size 64, got $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))" >&2
  exit 64
fi
for path in "$DATASET_DIR/meta/info.json" "$TEXT_CACHE" "$DATASET_STATS" "$ENCODER_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required training input: $path" >&2
    exit 66
  fi
done
if (( ${#INIT_ARGS[@]} )); then
  [[ -e "$PREDICTOR_INIT" ]] || { echo "Missing predictor init: $PREDICTOR_INIT" >&2; exit 66; }
fi
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  [[ -s "$RESUME" ]] || { echo "Missing resume checkpoint: $RESUME" >&2; exit 66; }
  # A policy resume already contains the predictor; do not combine it with
  # DROID-only predictor initialization (train.py rejects both options).
  INIT_ARGS=()
  RESUME_ARGS=(--resume "$RESUME")
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}
# Fresh high-throughput runs may opt into static-graph DDP.  The running
# droidinit job was already created before this switch and is unaffected.
export VJEPA_DDP_FAST=${VJEPA_DDP_FAST:-1}
mkdir -p "$OUTDIR"

echo "run=$OUTDIR"
echo "mode=$MODE"
echo "dataset_fps=20 video_stride=4 effective_video_fps=5"
echo "image_resize_mode=letterbox target=256x256"
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
    "${INIT_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
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
    --activation-checkpointing-blocks 0 \
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
