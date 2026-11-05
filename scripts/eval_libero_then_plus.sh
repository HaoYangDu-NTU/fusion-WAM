#!/usr/bin/env bash
# Persistent controller: retry failures; a STOP file is the explicit stop switch.
set -uo pipefail
if (( $# < 3 )); then
  echo 'Usage: bash scripts/eval_libero_then_plus.sh CKPT_DIR CKPT_NAME OUTPUT_DIR [controller options]'
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$(cd "$1" && pwd)"
CKPT_NAME="$2"
mkdir -p "$3"
OUTPUT_DIR="$(cd "$3" && pwd)"
shift 3
cd "$REPO_ROOT"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-${REPO_ROOT}/../../basics/miniconda3/envs/openwam/bin/python}"
exec 9>"$OUTPUT_DIR/supervisor.lock"
flock -n 9 || { echo "Already supervised: $OUTPUT_DIR" >&2; exit 1; }
echo "$$" > "$OUTPUT_DIR/supervisor.pid"
STOP_FILE=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --stop-file) STOP_FILE="${args[i+1]:-}" ;;
    --stop-file=*) STOP_FILE="${args[i]#*=}" ;;
  esac
done
stop_requested() {
  [[ -f "$OUTPUT_DIR/STOP" || -f "$CKPT_DIR/STOP" || -n "$STOP_FILE" && -f "$STOP_FILE" || \
     -f "$OUTPUT_DIR/libero/STOP" || -f "$OUTPUT_DIR/libero-plus/STOP" ]]
}
controller_pid=""
stop_controller() {
  touch "$OUTPUT_DIR/STOP"
  [[ -z "$controller_pid" ]] || kill -TERM "$controller_pid" 2>/dev/null || true
}
trap stop_controller INT TERM
while [[ ! -f "$OUTPUT_DIR/DONE" ]] && ! stop_requested; do
  "$OPENWAM_PYTHON" -u -m benchmarks.vla_harness.run_resumable \
    --ckpt-dir "$CKPT_DIR" --ckpt-name "$CKPT_NAME" --output-dir "$OUTPUT_DIR" "$@" \
    >> "$OUTPUT_DIR/supervisor.log" 2>&1 &
  controller_pid=$!
  wait "$controller_pid"
  status=$?
  if stop_requested; then
    wait "$controller_pid" 2>/dev/null || true
    break
  fi
  controller_pid=""
  [[ -f "$OUTPUT_DIR/DONE" ]] && break
  echo "$(date -Is) controller exited $status; restart in 30s" >> "$OUTPUT_DIR/supervisor.log"
  sleep 30
done
stop_requested && exit 130
[[ -f "$OUTPUT_DIR/DONE" ]]
