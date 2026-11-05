#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
DATASET_DIR=${DATASET_DIR:-/data/zhangyang/bench_datasets/robotwin_clean_plus_randomized}
TEXT_CACHE=${TEXT_CACHE:-/data/zhangyang/bench_datasets/robotwin_clean_plus_randomized_t5_len96_cache}
DATASET_STATS=${DATASET_STATS:-$TEXT_CACHE/dataset_stats_quantile.json}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT:-/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}
CLOTHES_CKPT=${CLOTHES_CKPT:-$ROOT/runs/teledata_tianji_clothes0825_no_idle_vjepa2_1_vitl_scratch_4view256_letterbox_t5len128_maxstate48_gbs64_step20000_seed7_nockpt/checkpoint_step020000.pt}
OUTDIR=${OUTDIR:-$ROOT/runs/robotwin2_vjepa2_1_vitl_scratch_3view192_t5len96_gbs128_5epoch_seed7_nockpt}
MAX_STEPS=${MAX_STEPS:-104550}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29647}

mkdir -p "$OUTDIR"
exec >>"$ROOT/runs/robotwin2_vjepa21_queue.log" 2>&1
echo "[$(date -Is)] waiting for clothes scratch checkpoint"
while [[ ! -s "$CLOTHES_CKPT" ]]; do sleep 30; done
while pgrep -f 'scripts/teledata/train.py' >/dev/null; do sleep 30; done

for path in "$DATASET_DIR/meta/info.json" "$DATASET_DIR/meta/stats.json" "$TEXT_CACHE" "$DATASET_STATS" "$ENCODER_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 66; }
done
shopt -s nullglob
cache_files=("$TEXT_CACHE"/*.t5_len96.pt)
(( ${#cache_files[@]} == 23559 )) || { echo "Expected 23559 T5 cache files, found ${#cache_files[@]}"; exit 66; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}

echo "[$(date -Is)] launching RobotWin2 scratch training"
echo "dataset=$DATASET_DIR views=3 crop=192 resize=letterbox context_len=96"
echo "global_batch_size=$((NUM_GPUS * BATCH_SIZE)) max_steps=$MAX_STEPS expected_epochs=5"
echo "activation_checkpointing_blocks=0"
exec "$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --num_machines 1 --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 --dynamo_backend no scripts/teledata/train.py \
    --dataset-dirs "$DATASET_DIR" --dataset-stats "$DATASET_STATS" \
    --text-cache-dir "$TEXT_CACHE" --instruction-field task \
    --views observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist \
    --view-layout independent --video-backend torchcodec \
    --checkpoint "$ENCODER_CHECKPOINT" --encoder-family vjepa2_1 --model-name vit_large \
    --encoder-checkpoint-key ema_encoder --encoder-interpolate-rope \
    --crop-size 192 --image-resize-mode letterbox --patch-size 16 --tubelet-size 2 \
    --num-frames 33 --past-frames 4 --video-frame-stride 4 --context-tubelets 1 \
    --require-full-future-window --context-len 96 --lang-dim 4096 \
    --action-normalization QUANTILE --state-normalization QUANTILE --no-clip-normalized \
    --action-chunk-size 32 --action-dim 14 --proprio-dim 14 --max-state-dim 0 \
    --proprio-encoding identity --action-hidden-size 512 --action-num-layers 24 \
    --condition-num-heads 8 --action-loss-weight 1.0 --action-num-inference-steps 10 \
    --pred-depth 24 --pred-embed-dim 1024 --pred-num-heads 16 --num-mask-tokens 10 \
    --activation-checkpointing-blocks 0 --predictor-rope-frequency-pairing corrected \
    --seed 7 --batch-size "$BATCH_SIZE" --gradient-accumulation-steps 1 \
    --num-workers 8 --prefetch-factor 2 --recycle-workers-every 1000 \
    --lr 1e-4 --weight-decay 0.01 --max-steps "$MAX_STEPS" --num-epochs 5 \
    --mixed-precision bf16 --log-every 10 --save-every 20910 \
    --output-dir "$OUTDIR" --wandb-project vjepa_policy_robotwin \
    --wandb-name "$(basename "$OUTDIR")"
