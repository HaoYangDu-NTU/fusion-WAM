#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
DROID_DIR=${DROID_DIR:-$ROOT/runs/teledata_tianji_saucer0821_no_idle_vjepa2_vitl_droidinit_4view256_letterbox_t5len128_maxstate48_gbs64_step40000_seed7_fast}
SCRATCH_DIR=${SCRATCH_DIR:-$ROOT/runs/teledata_tianji_saucer0821_no_idle_vjepa2_vitl_scratch_4view256_letterbox_t5len128_maxstate48_gbs64_step40000_seed7_fast}
DROID_OUTDIR=${DROID_OUTDIR:-${DROID_DIR}_8gpu_resume}
SCRATCH_OUTDIR=${SCRATCH_OUTDIR:-${SCRATCH_DIR}_8gpu_resume}
START_STEP=${START_STEP:-20000}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-1000}

check_checkpoint() {
  local path=$1 expected=$2
  [[ -s "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 66; }
  "$PYTHON" - "$path" "$expected" <<'PY'
import sys
import torch

path, expected = sys.argv[1], int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
if int(payload.get("step", -1)) != expected:
    raise SystemExit(f"{path}: expected step {expected}, got {payload.get('step')}")
for key in ("predictor", "action_expert", "proprio_encoder"):
    if key not in payload:
        raise SystemExit(f"{path}: missing {key}")
print(f"validated {path}: step={payload['step']} keys={len(payload['predictor'])}")
PY
}

check_checkpoint "$DROID_DIR/checkpoint_step$(printf '%06d' "$START_STEP").pt" "$START_STEP"
check_checkpoint "$SCRATCH_DIR/checkpoint_step$(printf '%06d' "$START_STEP").pt" "$START_STEP"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}

run_one() {
  local mode=$1 outdir=$2 ckpt=$3 port=$4
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] starting $mode resume=$ckpt"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MODE="$mode" NUM_GPUS=8 BATCH_SIZE=8 GRAD_ACCUM=1 \
    NUM_WORKERS="$NUM_WORKERS" PREFETCH_FACTOR="$PREFETCH_FACTOR" \
    RECYCLE_WORKERS_EVERY="$RECYCLE_WORKERS_EVERY" MAIN_PROCESS_PORT="$port" \
    RESUME="$ckpt" OUTDIR="$outdir" \
    bash scripts/teledata/run_saucer.sh \
    > "$outdir/resume8_launcher.log" 2>&1
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] finished $mode"
}

mkdir -p "$DROID_OUTDIR" "$SCRATCH_OUTDIR"
run_one droidinit "$DROID_OUTDIR" "$DROID_DIR/checkpoint_step$(printf '%06d' "$START_STEP").pt" 29651
run_one scratch "$SCRATCH_OUTDIR" "$SCRATCH_DIR/checkpoint_step$(printf '%06d' "$START_STEP").pt" 29653
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] both resumed runs finished"
