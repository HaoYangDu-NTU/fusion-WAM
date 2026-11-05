#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

DATASET_DIR=${DATASET_DIR:-/data-team/rhodes_team/huggingface_cache/rhodes_lerobot/TeleData/Tianji/clothes0825_no_idle}
TEXT_CACHE=${TEXT_CACHE:-/data/zhangyang/bench_datasets/clothes0825_no_idle_t5_cache}
DATASET_STATS=${DATASET_STATS:-$TEXT_CACHE/dataset_stats_quantile.json}
DROID_SCRIPT=${DROID_SCRIPT:-$ROOT/scripts/teledata/run_tianji_clean_noodle_vjepa21_droidinit_letterbox.sh}
SCRATCH_SCRIPT=${SCRATCH_SCRIPT:-$ROOT/scripts/teledata/run_tianji_clean_noodle_vjepa21_scratch_letterbox.sh}
DROID_OUTDIR=${DROID_OUTDIR:-$ROOT/runs/teledata_tianji_clothes0825_no_idle_vjepa2_1_vitl_droidinit_4view256_letterbox_t5len128_maxstate48_gbs64_step20000_seed7_nockpt}
SCRATCH_OUTDIR=${SCRATCH_OUTDIR:-$ROOT/runs/teledata_tianji_clothes0825_no_idle_vjepa2_1_vitl_scratch_4view256_letterbox_t5len128_maxstate48_gbs64_step20000_seed7_nockpt}
LOG=${LOG:-$ROOT/runs/teledata_clothes0825_vjepa21_pair_queue.log}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] validating clothes inputs"
for path in "$DATASET_DIR/meta/info.json" "$TEXT_CACHE" "$DATASET_STATS"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 66; }
done
shopt -s nullglob
cache_files=("$TEXT_CACHE"/*.t5_len128.pt)
(( ${#cache_files[@]} >= 1 )) || { echo "Missing clothes T5 cache"; exit 66; }

while pgrep -f 'scripts/teledata/train.py' >/dev/null; do sleep 30; done
echo "[$(date -Is)] starting clothes DROID-init 20k run"
DATASET_DIR="$DATASET_DIR" TEXT_CACHE="$TEXT_CACHE" DATASET_STATS="$DATASET_STATS" \
  OUTDIR="$DROID_OUTDIR" MAX_STEPS=20000 SAVE_EVERY=10000 MAIN_PROCESS_PORT=29645 \
  bash "$DROID_SCRIPT"
echo "[$(date -Is)] clothes DROID-init run completed"

echo "[$(date -Is)] starting clothes scratch 20k run"
DATASET_DIR="$DATASET_DIR" TEXT_CACHE="$TEXT_CACHE" DATASET_STATS="$DATASET_STATS" \
  OUTDIR="$SCRATCH_OUTDIR" MAX_STEPS=20000 SAVE_EVERY=10000 MAIN_PROCESS_PORT=29646 \
  bash "$SCRATCH_SCRIPT"
echo "[$(date -Is)] both clothes runs completed"
