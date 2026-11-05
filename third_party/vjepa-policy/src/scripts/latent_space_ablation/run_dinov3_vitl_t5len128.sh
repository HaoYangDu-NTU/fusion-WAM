#!/usr/bin/env bash
set -euo pipefail

# Re-run the recorded DINOv3 ViT-L latent ablation with only the T5 cache length
# changed from 32 to 128.  The model/data/runtime values below mirror the old
# DINOv3 run and are intentionally explicit rather than inherited from defaults.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_DEFAULT=/data/zhangyang/miniconda3/envs/prts-ft/bin/python
DATASET_ROOT_DEFAULT=/data/zhangyang/JEPA-Project/datasets/LEROBOT_LIBERO_DATA
WEIGHTS_DEFAULT=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub
TEXT_CACHE_DEFAULT=/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128
DATASET_STATS_DEFAULT=$ROOT/runs/full_encoder_ablation_vjepa2_1_vitl_independent224_t5len128_gbs128_step21360_seed7_8gpu/dataset_stats.json
LATENT_CHECKPOINT_DEFAULT=$WEIGHTS_DEFAULT/dinov3-vitl16-ea8dc2863c51
BASELINE_CHECKPOINT_DEFAULT=$WEIGHTS_DEFAULT/checkpoints/vitl.pt

PYTHON=${PYTHON:-$PYTHON_DEFAULT}
DATASET_ROOT=${LIBERO_DATA_ROOT:-$DATASET_ROOT_DEFAULT}
TEXT_CACHE=${TEXT_EMBEDDING_CACHE:-$TEXT_CACHE_DEFAULT}
DATASET_STATS=${DATASET_STATS:-$DATASET_STATS_DEFAULT}
LATENT_CHECKPOINT=${LATENT_CHECKPOINT:-$LATENT_CHECKPOINT_DEFAULT}
BASELINE_CHECKPOINT=${VJEPA2_ENCODER_CHECKPOINT:-$BASELINE_CHECKPOINT_DEFAULT}
GPU_IDS=${GPU_IDS:-}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
OUTDIR=${OUTDIR:-}
MODE=run
SMOKE_STEPS=10

usage() {
  cat <<'EOF'
Usage: run_dinov3_vitl_t5len128.sh [options]

Formal-run defaults reproduce the recorded DINOv3 ViT-L LIBERO ablation,
except that the instruction cache/context length is 128 instead of 32.

Options:
  --outdir PATH             Fresh run directory under runs/ (or an explicit path)
  --gpus IDS                CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3
  --main-process-port PORT  Accelerate rendezvous port (default: 29500)
  --python PATH             Policy Python (default: prts-ft environment)
  --dataset-root PATH       Directory containing the four LIBERO LeRobot datasets
  --text-cache PATH         T5 length-128 cache
  --dataset-stats PATH      Shared dataset_stats.json
  --latent-checkpoint PATH  Local DINOv3 ViT-L Hugging Face directory
  --baseline-checkpoint PATH V-JEPA2 ViT-L checkpoint used for baseline RNG/init
  --smoke STEPS             Run a diagnostic run with STEPS optimizer steps
  --check                   Validate paths/checkpoint/cache and exit
  -h, --help                Show this help

The formal run is fixed to four GPUs, per-device batch 32, seed 7, 21,360
steps, bf16, crop 224, independent two-view geometry, and the recorded DINOv3
encoder chunking (activation checkpointing 8, image microbatch 320).
EOF
}

while (($#)); do
  case "$1" in
    --outdir) OUTDIR=${2:?--outdir requires a path}; shift 2 ;;
    --gpus) GPU_IDS=${2:?--gpus requires comma-separated IDs}; shift 2 ;;
    --main-process-port) MAIN_PROCESS_PORT=${2:?--main-process-port requires a port}; shift 2 ;;
    --python) PYTHON=${2:?--python requires a path}; shift 2 ;;
    --dataset-root) DATASET_ROOT=${2:?--dataset-root requires a path}; shift 2 ;;
    --text-cache) TEXT_CACHE=${2:?--text-cache requires a path}; shift 2 ;;
    --dataset-stats) DATASET_STATS=${2:?--dataset-stats requires a path}; shift 2 ;;
    --latent-checkpoint) LATENT_CHECKPOINT=${2:?--latent-checkpoint requires a path}; shift 2 ;;
    --baseline-checkpoint) BASELINE_CHECKPOINT=${2:?--baseline-checkpoint requires a path}; shift 2 ;;
    --smoke) MODE=smoke; SMOKE_STEPS=${2:?--smoke requires a positive step count}; shift 2 ;;
    --check) MODE=check; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ! "$MAIN_PROCESS_PORT" =~ ^[0-9]+$ ]] || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
  echo "--main-process-port must be an integer in [1, 65535]" >&2
  exit 64
fi
if [[ "$MODE" == smoke && (! "$SMOKE_STEPS" =~ ^[1-9][0-9]*$) ]]; then
  echo "--smoke requires a positive integer step count" >&2
  exit 64
fi

DATASETS=(
  "$DATASET_ROOT/libero_object_no_noops_1.0.0_lerobot"
  "$DATASET_ROOT/libero_goal_no_noops_1.0.0_lerobot"
  "$DATASET_ROOT/libero_spatial_no_noops_1.0.0_lerobot"
  "$DATASET_ROOT/libero_10_no_noops_1.0.0_lerobot"
)
for path in "${DATASETS[@]}"; do
  [[ -d "$path" ]] || { echo "Missing LIBERO dataset directory: $path" >&2; exit 1; }
done
for path in "$PYTHON" "$BASELINE_CHECKPOINT" "$DATASET_STATS"; do
  [[ -s "$path" ]] || { echo "Missing readable file: $path" >&2; exit 1; }
done
[[ -d "$TEXT_CACHE" ]] || { echo "Missing T5 length-128 cache directory: $TEXT_CACHE" >&2; exit 1; }
[[ -s "$LATENT_CHECKPOINT/config.json" ]] || { echo "Missing DINO config: $LATENT_CHECKPOINT/config.json" >&2; exit 1; }
[[ -s "$LATENT_CHECKPOINT/model.safetensors" ]] || { echo "Missing DINO weights: $LATENT_CHECKPOINT/model.safetensors" >&2; exit 1; }

latent_size=$(stat -c %s "$LATENT_CHECKPOINT/model.safetensors")
latent_sha256=$(sha256sum "$LATENT_CHECKPOINT/model.safetensors" | awk '{print $1}')
if [[ "$latent_size" != 1212559808 || "$latent_sha256" != dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179 ]]; then
  echo "DINOv3 checkpoint size/SHA256 mismatch" >&2
  echo "  size=$latent_size sha256=$latent_sha256" >&2
  exit 1
fi

cache_file=$(find "$TEXT_CACHE" -maxdepth 1 -type f -name '*.t5_len128.pt' -print -quit)
[[ -n "$cache_file" ]] || { echo "No *.t5_len128.pt files found in $TEXT_CACHE" >&2; exit 1; }
"$PYTHON" - "$cache_file" <<'PY'
import sys
import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(payload, dict) or "context" not in payload or "mask" not in payload:
    raise SystemExit(f"invalid text cache payload: {path}")
context, mask = payload["context"], payload["mask"]
if tuple(context.shape) != (128, 4096) or tuple(mask.shape) != (128,):
    raise SystemExit(
        f"expected context=(128,4096), mask=(128,), got {tuple(context.shape)}, {tuple(mask.shape)}"
    )
print(f"validated_t5_cache={path} context_shape={tuple(context.shape)} mask_shape={tuple(mask.shape)}")
PY

if [[ "$MODE" == check ]]; then
  printf 'validated_dinov3_vitl_t5len128 latent_checkpoint=%s text_cache=%s\n' \
    "$LATENT_CHECKPOINT" "$TEXT_CACHE"
  exit 0
fi

if [[ -n "$GPU_IDS" ]]; then
  IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
  if (( ${#gpu_array[@]} != 4 )); then
    echo "Formal DINOv3 run requires exactly four comma-separated GPU IDs" >&2
    exit 64
  fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

if [[ -z "$OUTDIR" ]]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  OUTDIR=$ROOT/runs/latent_dinov3_vitl_independent224_t5len128_gbs128_step21360_seed7_$stamp
fi
if [[ -e "$OUTDIR" ]] && find "$OUTDIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to append to non-empty run directory: $OUTDIR" >&2
  exit 1
fi
mkdir -p "$OUTDIR"

if [[ "$MODE" == smoke ]]; then
  NUM_GPUS=${NUM_GPUS:-4}
  MAX_STEPS=$SMOKE_STEPS
  SAVE_EVERY=0
else
  NUM_GPUS=4
  MAX_STEPS=21360
  SAVE_EVERY=4000
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHON LIBERO_DATA_ROOT="$DATASET_ROOT" TEXT_EMBEDDING_CACHE="$TEXT_CACHE"
export DATASET_STATS TRAIN_SCRIPT=scripts/latent_space_ablation/train_dinov3_vitl_t5len128.py
export VJEPA2_ENCODER_CHECKPOINT="$BASELINE_CHECKPOINT"
export ENCODER_FAMILY=vjepa2 ENCODER_MODEL_NAME=vit_large ENCODER_CHECKPOINT_KEY=target_encoder
export LATENT_ENCODER=dinov3 LATENT_CHECKPOINT="$LATENT_CHECKPOINT"
export IMAGE_MICROBATCH_SIZE=320 VIDEO_MICROBATCH_SIZE=32
export FASTWAM_SRC_DIR=/data/zhangyang/workspace/FastWAM/src
export CROP_SIZE=224 VIEW_LAYOUT=independent ENCODER_INTERPOLATE_ROPE=1
export PREDICTOR_ROPE_FREQUENCY_PAIRING=corrected
export PRED_DEPTH=24 PRED_EMBED_DIM=1024 PRED_NUM_HEADS=16
export ACTION_HIDDEN_SIZE=512 CONDITION_NUM_HEADS=8
export MIXED_PRECISION=bf16 NUM_WORKERS=8 PREFETCH_FACTOR=2 RECYCLE_WORKERS_EVERY=500
export GRAD_ACCUM=1 CONTEXT_LEN=128 SEED=7 ACTIVATION_CHECKPOINTING_BLOCKS=8 BATCH_SIZE=32
export NUM_GPUS MAX_STEPS SAVE_EVERY OUTDIR MAIN_PROCESS_PORT

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != 128 )); then
  echo "DINOv3 comparison requires global batch 128" >&2
  exit 64
fi

printf 'dinov3_vitl_t5len128 mode=%s gpus=%s outdir=%s\n' \
  "$MODE" "${CUDA_VISIBLE_DEVICES:-default}" "$OUTDIR"
exec bash "$ROOT/scripts/train_vjepa_policy.sh"
