#!/usr/bin/env bash
set -euo pipefail

# Run the three encoder variants as separate, sequential 8-GPU jobs.  Keeping
# one variant per process group avoids sharing model memory and makes each
# output directory a self-contained comparison point.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOG_DIR="$ROOT/runs/encoder_ablation_exact_queue"
mkdir -p "$LOG_DIR"

variants=(vjepa2_vitg vjepa2_1_vitl vjepa2_1_vitg)
for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  outdir="$ROOT/runs/full_encoder_ablation_${variant}_independent224_t5len128_gbs128_step21360_seed7_8gpu"
  final="$outdir/checkpoint_step021360.pt"
  if [[ -s "$final" ]]; then
    echo "[$(date -u +%FT%TZ)] already complete: $variant ($final)" | tee -a "$LOG_DIR/queue.log"
    continue
  fi
  port=$((29710 + index))
  mkdir -p "$outdir"
  echo "[$(date -u +%FT%TZ)] starting: $variant" | tee -a "$LOG_DIR/queue.log"
  (
    export OUTDIR="$outdir"
    export MAIN_PROCESS_PORT="$port"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    export VJEPA_DDP_FAST=${VJEPA_DDP_FAST:-1}
    bash "$ROOT/scripts/encoder_ablation/run_exact_t5len128.sh" "$variant"
  ) 2>&1 | tee "$outdir/launcher.log"
  echo "[$(date -u +%FT%TZ)] finished: $variant" | tee -a "$LOG_DIR/queue.log"
done
echo "[$(date -u +%FT%TZ)] encoder queue finished" | tee -a "$LOG_DIR/queue.log"
