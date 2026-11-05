#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CLEAN_CKPT=${CLEAN_CKPT:-$ROOT/runs/teledata_tianji_clean_noodle_vjepa2_1_vitl_scratch_4view256_letterbox_t5len128_maxstate48_gbs64_step60000_seed7_nockpt/checkpoint_step060000.pt}
DATASET_DIR=${DATASET_DIR:-/data-team/rhodes_team/huggingface_cache/rhodes_lerobot/TeleData/Tianji/saucer0821_no_idle}
TEXT_CACHE=${TEXT_CACHE:-/data/zhangyang/bench_datasets/saucer0821_no_idle_t5_cache}
DATASET_STATS=${DATASET_STATS:-$TEXT_CACHE/dataset_stats_quantile.json}
DROID_SCRIPT=${DROID_SCRIPT:-$ROOT/scripts/teledata/run_tianji_clean_noodle_vjepa21_droidinit_letterbox.sh}
SCRATCH_SCRIPT=${SCRATCH_SCRIPT:-$ROOT/scripts/teledata/run_tianji_clean_noodle_vjepa21_scratch_letterbox.sh}
DROID_OUTDIR=${DROID_OUTDIR:-$ROOT/runs/teledata_tianji_saucer0821_no_idle_vjepa2_1_vitl_droidinit_4view256_letterbox_t5len128_maxstate48_gbs64_step40000_seed7_nockpt}
SCRATCH_OUTDIR=${SCRATCH_OUTDIR:-$ROOT/runs/teledata_tianji_saucer0821_no_idle_vjepa2_1_vitl_scratch_4view256_letterbox_t5len128_maxstate48_gbs64_step40000_seed7_nockpt}
LOG=${LOG:-$ROOT/runs/teledata_saucer0821_vjepa21_pair_queue.log}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for clean_the_noodle_bowl scratch 60k checkpoint"
while [[ ! -s "$CLEAN_CKPT" ]]; do sleep 30; done
echo "[$(date -Is)] clean scratch checkpoint found: $CLEAN_CKPT"

while pgrep -f 'scripts/teledata/train.py' >/dev/null; do sleep 30; done
for path in "$DATASET_DIR/meta/info.json" "$TEXT_CACHE" "$DATASET_STATS"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 66; }
done

echo "[$(date -Is)] starting saucer DROID-init 40k run"
DATASET_DIR="$DATASET_DIR" TEXT_CACHE="$TEXT_CACHE" DATASET_STATS="$DATASET_STATS" \
  OUTDIR="$DROID_OUTDIR" MAX_STEPS=40000 SAVE_EVERY=10000 MAIN_PROCESS_PORT=29643 \
  bash "$DROID_SCRIPT"
echo "[$(date -Is)] saucer DROID-init run completed"

echo "[$(date -Is)] starting saucer scratch 40k run"
DATASET_DIR="$DATASET_DIR" TEXT_CACHE="$TEXT_CACHE" DATASET_STATS="$DATASET_STATS" \
  OUTDIR="$SCRATCH_OUTDIR" MAX_STEPS=40000 SAVE_EVERY=10000 MAIN_PROCESS_PORT=29644 \
  bash "$SCRATCH_SCRIPT"
echo "[$(date -Is)] both saucer runs completed"
