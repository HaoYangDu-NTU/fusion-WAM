#!/usr/bin/env bash
# GPU2: one training run, then resumable full LIBERO and LIBERO-Plus.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# No arguments starts a new experiment; pass its directory to resume.
if (( $# == 0 )); then
  QUEUE_DIR="outputs/train_eval/exp_$(date +%Y%m%d_%H%M%S)"
  mkdir -p outputs/train_eval
  mkdir "$QUEUE_DIR"
  printf 'Queue directory: %s/%s\n' "$REPO_ROOT" "$QUEUE_DIR"
  set -- "$QUEUE_DIR"
fi
export OPENWAM_PYTHON="${OPENWAM_PYTHON:-${REPO_ROOT}/../../basics/miniconda3/envs/openwam/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/vla-evaluation-harness/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "$OPENWAM_PYTHON" -u -m benchmarks.vla_harness.run_train_eval \
  --train-script scripts/train_libero_vjepa_1_3b_dev.sh "$@"
