#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CURRENT_RUN=${CURRENT_RUN:-$ROOT/runs/droid_init_full_encoder_vjepa2_1_vitl_independent224_t5len128_maxstate48_gbs128_step21360_seed7_nockpt}
DROID_SCRIPT=${DROID_SCRIPT:-$ROOT/scripts/teledata/run_tianji_clean_noodle_vjepa21_droidinit_letterbox.sh}
SCRATCH_SCRIPT=${SCRATCH_SCRIPT:-$ROOT/scripts/teledata/run_tianji_clean_noodle_vjepa21_scratch_letterbox.sh}
QUEUE_LOG=${QUEUE_LOG:-$ROOT/runs/teledata_clean_noodle_vjepa21_pair_queue.log}

mkdir -p "$(dirname "$QUEUE_LOG")"
exec >>"$QUEUE_LOG" 2>&1
echo "[$(date -Is)] waiting for current LIBERO run"

while true; do
  if rg -q "\[done\] training finished at step=21360" "$CURRENT_RUN/train.log" \
    && [[ -s "$CURRENT_RUN/checkpoint_step021360.pt" ]]; then
    break
  fi
  sleep 30
done

if rg -q "Traceback|CUDA out of memory|OutOfMemory|NCCL.*(error|Error)|RuntimeError|Killed|No space left" "$CURRENT_RUN/train.log"; then
  echo "[$(date -Is)] fatal error markers found in current run; refusing queued launch"
  exit 1
fi

echo "[$(date -Is)] current run verified complete; starting DROID-init TeleData run"
bash "$DROID_SCRIPT"
echo "[$(date -Is)] DROID-init run exited; starting scratch TeleData run"
MAX_STEPS=60000 bash "$SCRATCH_SCRIPT"
echo "[$(date -Is)] both TeleData runs completed"
