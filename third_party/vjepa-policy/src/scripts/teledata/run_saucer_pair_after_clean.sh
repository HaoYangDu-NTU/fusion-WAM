#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

CLEAN_PID=${CLEAN_PID:-7263}
CLEAN_RUN=${CLEAN_RUN:-$ROOT/runs/teledata_tianji_clean_noodle_vjepa2_vitl_scratch_4view256_letterbox_t5len128_maxstate48_gbs64_step100000_seed7_fast_nockpt}
DATASET_DIR=${DATASET_DIR:-/data-team/rhodes_team/huggingface_cache/rhodes_lerobot/TeleData/Tianji/saucer0821_no_idle}
TEXT_CACHE=${TEXT_CACHE:-/data/zhangyang/bench_datasets/saucer0821_no_idle_t5_cache}
STATS=${STATS:-$TEXT_CACHE/dataset_stats_quantile.json}
TEXT_MODEL=${TEXT_MODEL:-/data/zhangyang/JEPA-Project/hf-weights/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2}
PREP_LOG=${PREP_LOG:-/data/zhangyang/bench_datasets/saucer0821_no_idle_prepare.log}

echo "[queue] waiting for clean run pid=$CLEAN_PID"
while kill -0 "$CLEAN_PID" 2>/dev/null; do
  sleep "${POLL_SECONDS:-30}"
done

FINAL="$CLEAN_RUN/checkpoint_step100000.pt"
if [[ ! -s "$FINAL" ]]; then
  echo "[queue] clean run exited without $FINAL" >&2
  exit 1
fi

echo "[queue] clean run finished; preparing saucer stats and T5 cache"
"${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}" \
  scripts/teledata/prepare_saucer.py \
  --dataset-dir "$DATASET_DIR" \
  --stats-output "$STATS" \
  --text-cache-dir "$TEXT_CACHE" \
  --text-model "$TEXT_MODEL" \
  --context-length 128 \
  --device cpu \
  2>&1 | tee -a "$PREP_LOG"

[[ -s "$STATS" ]] || { echo "[queue] stats preparation failed" >&2; exit 1; }
shopt -s nullglob
cache_files=("$TEXT_CACHE"/*.t5_len128.pt)
(( ${#cache_files[@]} >= 2 )) || { echo "[queue] T5 cache preparation failed" >&2; exit 1; }

echo "[queue] launching droid-initialized and scratch runs on 4 GPUs each"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MODE=droidinit NUM_GPUS=4 BATCH_SIZE=8 GRAD_ACCUM=2 MAIN_PROCESS_PORT=29641 \
  DATASET_DIR="$DATASET_DIR" TEXT_CACHE="$TEXT_CACHE" DATASET_STATS="$STATS" \
  bash scripts/teledata/run_saucer.sh \
  > "$ROOT/runs/teledata_saucer_droidinit_launcher.log" 2>&1 &
droid_pid=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 \
  MODE=scratch NUM_GPUS=4 BATCH_SIZE=8 GRAD_ACCUM=2 MAIN_PROCESS_PORT=29643 \
  DATASET_DIR="$DATASET_DIR" TEXT_CACHE="$TEXT_CACHE" DATASET_STATS="$STATS" \
  bash scripts/teledata/run_saucer.sh \
  > "$ROOT/runs/teledata_saucer_scratch_launcher.log" 2>&1 &
scratch_pid=$!

echo "[queue] droid_pid=$droid_pid scratch_pid=$scratch_pid"
wait "$droid_pid" "$scratch_pid"
