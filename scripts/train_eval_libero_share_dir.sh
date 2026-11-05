#!/usr/bin/env bash
# Start one training run, then resume LIBERO and full Plus until both finish.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# No arguments starts a new experiment; pass its directory to resume.
# Select 5B with OPENWAM_TRAIN_SCRIPT=scripts/train_libero_vjepa_5b_dev_share_dir.sh.
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
TRAIN_SCRIPT="${OPENWAM_TRAIN_SCRIPT:-scripts/train_libero_vjepa_1_3b_dev_share_dir.sh}"
# A resumed queue keeps its recorded training entry, including 5B queues.
if [[ -z "${OPENWAM_TRAIN_SCRIPT:-}" && -f "$1/job.json" ]]; then
  TRAIN_SCRIPT="$("$OPENWAM_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["command"][1])' "$1/job.json")"
fi
exec "$OPENWAM_PYTHON" -u -m benchmarks.vla_harness.run_train_eval \
  --train-script "$TRAIN_SCRIPT" "$@"
