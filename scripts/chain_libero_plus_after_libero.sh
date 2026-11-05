#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 4 )); then
  echo "Usage: $0 CKPT_DIR CKPT_NAME LIBERO_OUTPUT_DIR PLUS_OUTPUT_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OPENWAM_PYTHON:-/data/fcy/basics/miniconda3/envs/openwam/bin/python}"
CKPT_DIR="$(cd "$1" && pwd)"
CKPT_NAME="$2"
LIBERO_OUTPUT="$(cd "$3" && pwd)"
mkdir -p "$4"
PLUS_OUTPUT="$(cd "$4" && pwd)"
PLUS_CACHE="$ROOT/cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text"
PLUS_DEPLOY="$PLUS_OUTPUT/deployment"
LOG="$PLUS_OUTPUT/chain.log"
mkdir -p "$PLUS_OUTPUT"
exec > >(tee -a "$LOG") 2>&1

echo "$(date -Is) Waiting for standard LIBERO to finish."
while [[ ! -f "$LIBERO_OUTPUT/exit_code" ]]; do
  sleep 30
done

LIBERO_STATUS="$(<"$LIBERO_OUTPUT/exit_code")"
if [[ "$LIBERO_STATUS" != 0 ]]; then
  echo "$(date -Is) Standard LIBERO exited with status $LIBERO_STATUS; Plus was not started."
  exit 1
fi
echo "$(date -Is) Standard LIBERO completed successfully."

export OPENWAM_PYTHON="$PY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$PLUS_CACHE/manifest.json" ]]; then
  echo "$(date -Is) Generating corrected LIBERO-Plus RynnBrain cache."
  "$PY" scripts/experiment.py cache libero-plus --gpus 0,1,2,3
else
  echo "$(date -Is) Reusing existing Plus cache: $PLUS_CACHE"
fi

echo "$(date -Is) Preparing Plus deployment: $PLUS_DEPLOY"
"$PY" - "$CKPT_DIR" "$CKPT_NAME" "$PLUS_DEPLOY" "$PLUS_CACHE" <<'PY'
import sys
from pathlib import Path
from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint

source, checkpoint, target, cache = map(Path, sys.argv[1:])
prepare_checkpoint(source, checkpoint.name, target, cache, corrected=True)
print(target)
PY

export TEXT_EMBEDDING_DIR="$PLUS_CACHE"
echo "$(date -Is) Starting LIBERO-Plus evaluation."
exec "$PY" -u -m benchmarks.vla_harness.run_eval \
  --benchmark libero-plus \
  --ckpt-dir "$PLUS_DEPLOY" \
  --ckpt-name "$CKPT_NAME" \
  --correct-instruction-suffix true \
  --gpus 0,1,2,3 \
  --workers 128 \
  --max-batch-size 32 \
  --render gpu \
  --render-gpus 0,1,2,3 \
  --base-port 8994 \
  --libero-plus-python /data/fcy/basics/miniconda3/envs/libero-plus/bin/python \
  --output-dir "$PLUS_OUTPUT/evaluation"
