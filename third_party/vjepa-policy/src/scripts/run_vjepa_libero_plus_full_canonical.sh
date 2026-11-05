#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_vjepa_libero_plus_full_canonical.sh --mode fixed128 [options]

Run all classified LIBERO-Plus suites with one VJEPA server per GPU and a
single global shard set. The finished SQLite and merged JSON are copied to
--output-dir. The default topology is eight GPUs with 12 EGL simulator clients
per model server and server batch size 12 (96 global shards).

Options:
  --mode MODE             fixed128 (canonical, untruncated T5 cache)
  --gpus LIST             comma-separated CUDA device ids (default: 0,1,2,3,4,5,6,7)
  --sim-gpus LIST         one physical EGL GPU id per model-server GPU (default: --gpus)
  --clients-per-gpu N     EGL simulator clients per model server (default: 12)
  --batch-size N          server inference batch size (default: clients per server)
  --base-port N           first localhost server port (default: 8000)
  --output-dir DIR        persistent result directory (required)
  --eval-id ID            reproducible eval id (default: generated UUID)
  --policy-name NAME      label used by the live progress monitor
  --server-config PATH    override the mode's default model-server config
  --text-cache-dir DIR    override the mode's language cache directory
  --cache-inventory PATH  validate an external cache against this manifest
  --benchmark-config PATH override the LIBERO-Plus benchmark config
  --resume-local-dir DIR  reuse an interrupted /dev/shm run directory
  --resume-started-at UTC original UTC start time for resumed wall timing
  --rollout-offset N      completed rollout seconds before resume (default: 0)
EOF
  exit "${1:-0}"
}

WORKSPACE="${VLA_WORKSPACE:-/data/zhangyang/workspace}"
CONDA_ROOT="${VLA_CONDA_ROOT:-/data/zhangyang/miniconda3}"
HARNESS="$WORKSPACE/vla-evaluation-harness"
POLICY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HARNESS/scripts/libero_plus_runtime_env.sh"
libero_plus_setup_imagemagick
BENCH_ENV="$CONDA_ROOT/envs/libero-plus-eval"
SERVER_PYTHON="${VJEPA_SERVER_PYTHON:-$CONDA_ROOT/envs/VLA_JEPA/bin/python}"
LIBERO_CONFIG_PATH="$WORKSPACE/.libero-plus"
SERVER_CONFIG="$HARNESS/configs/model_servers/vjepa_policy/vjepa2_libero_plus.yaml"
FULL_SERVER_CONFIG="$HARNESS/configs/model_servers/vjepa_policy/vjepa2_libero_plus_full_language.yaml"
T5LEN128_SERVER_CONFIG="$HARNESS/configs/model_servers/vjepa_policy/vjepa2_libero_plus_t5len128.yaml"
BENCHMARK_CONFIG="$HARNESS/configs/benchmarks/libero_plus/all.yaml"
MODE=""
GPU_LIST="0,1,2,3,4,5,6,7"
SIM_GPU_LIST=""
CLIENTS_PER_GPU=12
BATCH_SIZE=""
BASE_PORT=8000
OUTPUT_DIR=""
EVAL_ID=""
POLICY_NAME="${VJEPA_POLICY_NAME:-}"
SERVER_CONFIG_OVERRIDE="${VJEPA_SERVER_CONFIG:-}"
TEXT_CACHE_OVERRIDE="${VJEPA_TEXT_CACHE_DIR:-}"
CACHE_INVENTORY_OVERRIDE=""
BENCHMARK_CONFIG_OVERRIDE=""
RESUME_LOCAL_DIR=""
RESUME_STARTED_AT=""
ROLLOUT_OFFSET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --sim-gpus) SIM_GPU_LIST="$2"; shift 2 ;;
    --clients-per-gpu) CLIENTS_PER_GPU="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --eval-id) EVAL_ID="$2"; shift 2 ;;
    --policy-name) POLICY_NAME="$2"; shift 2 ;;
    --server-config) SERVER_CONFIG_OVERRIDE="$2"; shift 2 ;;
    --text-cache-dir) TEXT_CACHE_OVERRIDE="$2"; shift 2 ;;
    --cache-inventory) CACHE_INVENTORY_OVERRIDE="$2"; shift 2 ;;
    --benchmark-config) BENCHMARK_CONFIG_OVERRIDE="$2"; shift 2 ;;
    --resume-local-dir) RESUME_LOCAL_DIR="$2"; shift 2 ;;
    --resume-started-at) RESUME_STARTED_AT="$2"; shift 2 ;;
    --rollout-offset) ROLLOUT_OFFSET="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

if [[ "$MODE" != "fixed128" ]]; then
  echo "Error: this local canonical runner requires --mode fixed128" >&2
  usage 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output-dir is required" >&2
  usage 1
fi
for value in "$CLIENTS_PER_GPU" "$BASE_PORT"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: numeric options must be positive integers" >&2
    exit 1
  fi
done
if [[ ! "$ROLLOUT_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "Error: --rollout-offset must be a non-negative integer" >&2
  exit 1
fi
if [[ -z "$BATCH_SIZE" ]]; then
  BATCH_SIZE="$CLIENTS_PER_GPU"
elif [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: numeric options must be positive integers" >&2
  exit 1
fi
if (( BATCH_SIZE > CLIENTS_PER_GPU )); then
  echo "Error: --batch-size cannot exceed --clients-per-gpu" >&2
  exit 1
fi
if [[ ! -f "$BENCHMARK_CONFIG" || ! -f "$SERVER_CONFIG" || ! -f "$FULL_SERVER_CONFIG" || ! -f "$T5LEN128_SERVER_CONFIG" ]]; then
  echo "Error: benchmark/server config is missing" >&2
  exit 1
fi
if [[ -n "$SERVER_CONFIG_OVERRIDE" && ! -f "$SERVER_CONFIG_OVERRIDE" ]]; then
  echo "Error: overridden server config is missing: $SERVER_CONFIG_OVERRIDE" >&2
  exit 1
fi
if [[ -n "$CACHE_INVENTORY_OVERRIDE" && ! -f "$CACHE_INVENTORY_OVERRIDE" ]]; then
  echo "Error: cache inventory manifest is missing: $CACHE_INVENTORY_OVERRIDE" >&2
  exit 1
fi
if [[ -n "$BENCHMARK_CONFIG_OVERRIDE" ]]; then
  if [[ ! -f "$BENCHMARK_CONFIG_OVERRIDE" ]]; then
    echo "Error: overridden benchmark config is missing: $BENCHMARK_CONFIG_OVERRIDE" >&2
    exit 1
  fi
  BENCHMARK_CONFIG="$BENCHMARK_CONFIG_OVERRIDE"
fi
if [[ -n "$RESUME_LOCAL_DIR" && -z "$RESUME_STARTED_AT" ]]; then
  echo "Error: --resume-started-at is required with --resume-local-dir" >&2
  exit 1
fi
if [[ -n "$RESUME_STARTED_AT" ]] && ! date -u -d "$RESUME_STARTED_AT" +%s >/dev/null 2>&1; then
  echo "Error: invalid --resume-started-at timestamp: $RESUME_STARTED_AT" >&2
  exit 1
fi
if [[ ! -x "$BENCH_ENV/bin/vla-eval" || ! -x "$SERVER_PYTHON" ]]; then
  echo "Error: expected conda environments are not ready" >&2
  exit 1
fi
LIBERO_ROOT="$WORKSPACE/LIBERO-plus/libero/libero"
if [[ ! -d "$LIBERO_ROOT" ]]; then
  echo "Error: LIBERO-Plus source tree is missing: $LIBERO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$LIBERO_ROOT/bddl_files" || ! -d "$LIBERO_ROOT/init_files" ]]; then
  echo "Warning: LIBERO-Plus bddl/init files were not found during preflight; benchmark import will validate them" >&2
fi
LIBERO_ASSETS="${VLA_LIBERO_PLUS_ASSETS:-}"
if [[ -d "$LIBERO_ASSETS/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets" ]]; then
  LIBERO_ASSETS="$LIBERO_ASSETS/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
fi
if [[ -z "$LIBERO_ASSETS" || ! -d "$LIBERO_ASSETS" ]]; then
  if [[ -d "$LIBERO_ROOT/assets" ]]; then
    LIBERO_ASSETS="$(readlink -f "$LIBERO_ROOT/assets" 2>/dev/null || true)"
  fi
fi
if [[ -z "$LIBERO_ASSETS" || ! -d "$LIBERO_ASSETS" ]]; then
  # The downloaded Plus asset archive is often unpacked under a randomized
  # workspace directory while the checkout keeps a stale temporary symlink.
  for candidate in "$WORKSPACE"/libero-plus-assets.*/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets; do
    if [[ -d "$candidate" ]]; then
      LIBERO_ASSETS="$candidate"
      break
    fi
  done
fi
if [[ -z "$LIBERO_ASSETS" || ! -d "$LIBERO_ASSETS" ]]; then
  echo "Error: LIBERO-Plus assets are missing; set VLA_LIBERO_PLUS_ASSETS" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "Error: --gpus cannot be empty" >&2
  exit 1
fi
if [[ -z "$SIM_GPU_LIST" ]]; then
  SIM_GPU_LIST="$GPU_LIST"
fi
IFS=',' read -r -a SIM_GPUS <<< "$SIM_GPU_LIST"
if (( ${#SIM_GPUS[@]} != ${#GPUS[@]} )); then
  echo "Error: --sim-gpus must contain one GPU id per --gpus entry" >&2
  exit 1
fi
SEEN_GPUS=","
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid GPU id: $gpu" >&2
    exit 1
  fi
  if [[ "$SEEN_GPUS" == *",$gpu,"* ]]; then
    echo "Error: duplicate GPU id: $gpu" >&2
    exit 1
  fi
  SEEN_GPUS+="$gpu,"
done
for gpu in "${SIM_GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid simulator GPU id: $gpu" >&2
    exit 1
  fi
done
NUM_GPUS="${#GPUS[@]}"
NUM_SHARDS=$((NUM_GPUS * CLIENTS_PER_GPU))
if (( BASE_PORT + NUM_GPUS - 1 > 65535 )); then
  echo "Error: server port range exceeds 65535" >&2
  exit 1
fi

if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
fi
if [[ ! "$EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Error: eval id may contain only letters, digits, '.', '_' and '-'" >&2
  exit 1
fi
OUTPUT_DIR="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUTPUT_DIR")"
if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Error: output directory is not empty: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "fixed32" ]]; then
  TEXT_CACHE_DIR="${VJEPA_LIBERO_PLUS_CACHE:-$WORKSPACE/VJEPA-WAM/data/text_embeds/libero_plus_task_language_t5xxl}"
  ACTIVE_SERVER_CONFIG="$SERVER_CONFIG"
elif [[ "$MODE" == "fixed128" ]]; then
  TEXT_CACHE_DIR="${VJEPA_LIBERO_PLUS_LEN128_CACHE:-$WORKSPACE/VJEPA-WAM/data/text_embeds/libero_plus_t5xxl_len128}"
  ACTIVE_SERVER_CONFIG="$T5LEN128_SERVER_CONFIG"
else
  TEXT_CACHE_DIR="${VJEPA_LIBERO_PLUS_EFFECTIVE_CACHE:-$WORKSPACE/VJEPA-WAM/data/text_embeds/libero_plus_t5xxl_effective}"
  ACTIVE_SERVER_CONFIG="$FULL_SERVER_CONFIG"
fi
if [[ -n "$TEXT_CACHE_OVERRIDE" ]]; then
  TEXT_CACHE_DIR="$TEXT_CACHE_OVERRIDE"
fi
if [[ -n "$SERVER_CONFIG_OVERRIDE" ]]; then
  ACTIVE_SERVER_CONFIG="$SERVER_CONFIG_OVERRIDE"
fi
if [[ ! -f "$ACTIVE_SERVER_CONFIG" ]]; then
  echo "Error: active model-server config is missing: $ACTIVE_SERVER_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$TEXT_CACHE_DIR" ]]; then
  echo "Error: language cache is missing: $TEXT_CACHE_DIR" >&2
  exit 1
fi

for command in curl nvidia-smi setsid; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Error: required command is missing: $command" >&2
    exit 1
  fi
done

AVAILABLE_GPUS=",$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -),"
for gpu in "${GPUS[@]}"; do
  if [[ "$AVAILABLE_GPUS" != *",$gpu,"* ]]; then
    echo "Error: GPU $gpu is not available (found $AVAILABLE_GPUS)" >&2
    exit 1
  fi
done
for gpu in "${SIM_GPUS[@]}"; do
  if [[ "$AVAILABLE_GPUS" != *",$gpu,"* ]]; then
    echo "Error: simulator GPU $gpu is not available (found $AVAILABLE_GPUS)" >&2
    exit 1
  fi
done

export PATH="$BENCH_ENV/bin:$PATH"
export PYTHONPATH="$POLICY_ROOT/src:$HARNESS/src${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

LOCAL_TAG="${MODE}-${EVAL_ID}"
if [[ -n "$POLICY_NAME" ]]; then
  LOCAL_TAG="${POLICY_NAME}-${LOCAL_TAG}"
fi
LOCAL_DIR="/dev/shm/vjepa-libero-plus-${LOCAL_TAG}"
if [[ -n "$RESUME_LOCAL_DIR" ]]; then
  LOCAL_DIR="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RESUME_LOCAL_DIR")"
  case "$LOCAL_DIR" in
    /dev/shm/vjepa-libero-plus-*) ;;
    *) echo "Error: resume directory must be below /dev/shm/vjepa-libero-plus-*: $LOCAL_DIR" >&2; exit 1 ;;
  esac
  if [[ ! -d "$LOCAL_DIR" || ! -f "$LOCAL_DIR/recording-$EVAL_ID.sqlite" ]]; then
    echo "Error: resume directory or recording database is missing: $LOCAL_DIR" >&2
    exit 1
  fi
  rm -f -- "$LOCAL_DIR/monitor.done"
  RESUME_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  for log in "$LOCAL_DIR"/model-server-gpu*.log "$LOCAL_DIR"/shard-*.log; do
    [[ -f "$log" ]] || continue
    mv -- "$log" "${log%.log}.pre-resume-${RESUME_STAMP}.log"
  done
else
  if [[ -e "$LOCAL_DIR" ]]; then
    echo "Error: local run directory already exists; use a fresh eval id: $LOCAL_DIR" >&2
    exit 1
  fi
  mkdir -p "$LOCAL_DIR"
fi
# Keep all runtime configuration in the harness-owned temporary run directory.
# In particular, do not repair the checkout's potentially stale assets symlink.
RUNTIME_LIBERO_CONFIG="$LOCAL_DIR/libero-config"
mkdir -p "$RUNTIME_LIBERO_CONFIG"
"$BENCH_ENV/bin/python" - "$RUNTIME_LIBERO_CONFIG/config.yaml" "$LIBERO_ROOT" "$LIBERO_ASSETS" <<'PY'
from pathlib import Path
import sys

config_path, root, assets = map(Path, sys.argv[1:])
config_path.write_text(
    "\n".join(
        [
            f"assets: {assets}",
            f"bddl_files: {root / 'bddl_files'}",
            f"benchmark_root: {root}",
            f"datasets: {root.parent / 'datasets'}",
            f"init_states: {root / 'init_files'}",
            "",
        ]
    ),
    encoding="utf-8",
)
PY
export LIBERO_CONFIG_PATH="$RUNTIME_LIBERO_CONFIG"
if [[ -d "$LIBERO_ROOT/bddl_files" && -d "$LIBERO_ROOT/init_files" ]]; then
  if ! "$BENCH_ENV/bin/python" -c 'from wand.image import Image; import libero.libero.envs.env_wrapper' >/dev/null 2>&1; then
    echo "Error: LIBERO-Plus Wand/ImageMagick import failed; install libmagickwand-dev or set VLA_IMAGEMAGICK_HOME" >&2
    exit 1
  fi
fi
SERVER_PIDS=()
SHARD_PIDS=()
MONITOR_PID=""
MONITOR_DONE="$LOCAL_DIR/monitor.done"
MONITOR_LOG="$LOCAL_DIR/monitor.log"
TOTAL_START_NS="$(date +%s%N)"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "$RESUME_STARTED_AT" ]]; then
  START_UTC="$(date -u -d "$RESUME_STARTED_AT" +%Y-%m-%dT%H:%M:%SZ)"
  TOTAL_START_NS="$(( $(date -u -d "$START_UTC" +%s) * 1000000000 ))"
fi

cleanup() {
  if [[ -n "$MONITOR_DONE" && -d "$(dirname "$MONITOR_DONE")" ]]; then
    : >"$MONITOR_DONE"
  fi
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "$MONITOR_PID" ]]; then
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  for pid in "${SHARD_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "${SHARD_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -- "-$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
stop_servers() {
  for index in "${!SERVER_PIDS[@]}"; do
    pid="${SERVER_PIDS[$index]}"
    if [[ -n "$pid" ]]; then
      kill -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      SERVER_PIDS[$index]=""
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

MONITOR_LABEL="${POLICY_NAME:-VJEPA-Policy}"
"$BENCH_ENV/bin/python" "$HARNESS/scripts/monitor_vjepa_eval.py" \
  --db "$LOCAL_DIR/recording-$EVAL_ID.sqlite" \
  --eval-id "$EVAL_ID" \
  --policy "$MONITOR_LABEL" \
  --total 10030 \
  --done-file "$MONITOR_DONE" \
  --interval "${VJEPA_MONITOR_INTERVAL:-60}" \
  --log "$MONITOR_LOG" &
MONITOR_PID=$!

for index in "${!GPUS[@]}"; do
  port=$((BASE_PORT + index))
  if ! "$BENCH_ENV/bin/python" - "$port" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f"port {sys.argv[1]} is already occupied: {exc}")
finally:
    sock.close()
PY
  then
    exit 1
  fi
done

echo "Validating $MODE cache provenance and tensor shapes"
if [[ -n "$CACHE_INVENTORY_OVERRIDE" ]]; then
  CACHE_MANIFEST="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CACHE_INVENTORY_OVERRIDE")"
  "$SERVER_PYTHON" - "$MODE" "$CACHE_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

mode, manifest_path = sys.argv[1:]
manifest_path = Path(manifest_path)
manifest = json.loads(manifest_path.read_text())
if mode != "fixed128":
    raise SystemExit("external cache inventory is supported only for fixed128 mode")
if manifest.get("task_count") != 10030 or manifest.get("unique_prompt_count") != 1549:
    raise SystemExit("canonical cache inventory must cover 10,030 tasks and 1,549 prompts")
if manifest.get("canonical_task_language") not in (True, None):
    raise SystemExit("external cache inventory must use canonical task-language prompts")
if manifest.get("cache_format", "fixed") != "fixed" or manifest.get("context_length") != 128:
    raise SystemExit("external cache inventory must describe fixed t5_len=128 entries")
print(f"validated external cache inventory {manifest_path}")
PY
else
  CACHE_MANIFEST="$TEXT_CACHE_DIR/manifest.json"
  "$SERVER_PYTHON" - "$MODE" "$TEXT_CACHE_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

mode = sys.argv[1]
root = Path(sys.argv[2])
metadata = json.loads((root / "cache_metadata.json").read_text())
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"cache manifest is missing: {manifest_path}")
if Path(metadata["manifest"]).resolve() != manifest_path.resolve():
    raise SystemExit("cache metadata points to a different manifest")
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if digest != metadata.get("manifest_sha256"):
    raise SystemExit("cache manifest SHA256 does not match cache_metadata.json")
manifest = json.loads(manifest_path.read_text())
if manifest.get("task_count") != 10030:
    raise SystemExit("cache manifest must cover all 10,030 LIBERO-Plus tasks")
if manifest.get("canonical_task_language") not in (True, None):
    raise SystemExit("cache manifest must use canonical LIBERO-Plus task-language instructions")
if metadata.get("unique_prompt_count") != 1549 or manifest.get("unique_prompt_count") != 1549:
    raise SystemExit("canonical cache must contain all 1,549 unique prompts")
if not Path(metadata.get("model_snapshot", "")).is_dir():
    raise SystemExit("recorded T5 model snapshot is missing")

if mode == "fixed32":
    if (
        metadata.get("cache_format", "fixed") != "fixed"
        or metadata.get("context_length") != 32
        or metadata.get("truncation") != "right"
        or metadata.get("truncated_unique_prompts")
        != manifest.get("statistics", {}).get("unique_prompts", {}).get("over_context")
    ):
        raise SystemExit("fixed32 cache metadata does not match the classified LIBERO-Plus cache")
elif mode == "fixed128":
    if (
        metadata.get("cache_format", "fixed") != "fixed"
        or metadata.get("context_length") != 128
        or metadata.get("truncation") != "none"
        or metadata.get("truncated_unique_prompts") != 0
    ):
        raise SystemExit("fixed128 canonical cache metadata does not match the 128-token cache")
else:
    if (
        metadata.get("cache_format") != "effective"
        or metadata.get("context_length") != 80
        or metadata.get("truncation") != "none"
        or metadata.get("truncated_unique_prompts") != 0
    ):
        raise SystemExit("effective cache metadata must declare untruncated effective format")
print(f"validated {mode} cache metadata")
PY
fi
VERIFY_ARGS=(
  verify
  --manifest "$CACHE_MANIFEST"
  --cache-dir "$TEXT_CACHE_DIR"
)
if [[ -n "$CACHE_INVENTORY_OVERRIDE" ]]; then
  VERIFY_ARGS+=(--allow-unexpected)
fi
if [[ "$MODE" == "effective" ]]; then
  VERIFY_ARGS+=(--variable-length)
fi
PYTHONPATH="$HARNESS/src" "$SERVER_PYTHON" "$HARNESS/scripts/build_vjepa_libero_plus_cache.py" "${VERIFY_ARGS[@]}"

cd "$HARNESS"

echo "policy=${POLICY_NAME:-VJEPA-Policy} mode=$MODE eval_id=$EVAL_ID server_gpus=$GPU_LIST sim_gpus=$SIM_GPU_LIST clients_per_server=$CLIENTS_PER_GPU shards=$NUM_SHARDS batch=$BATCH_SIZE"
echo "server_config=$ACTIVE_SERVER_CONFIG text_cache=$TEXT_CACHE_DIR render_backend=egl"
echo "started=$START_UTC local_db=$LOCAL_DIR/recording-$EVAL_ID.sqlite"

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  port=$((BASE_PORT + index))
  log="$LOCAL_DIR/model-server-gpu${gpu}.log"
  echo "Starting GPU $gpu server at ws://127.0.0.1:$port (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$BENCH_ENV/bin/vla-eval" serve \
    -c "$ACTIVE_SERVER_CONFIG" --python "$SERVER_PYTHON" \
    --arg "port=$port" --arg "max_batch_size=$BATCH_SIZE" \
    --arg "text_cache_dir=$TEXT_CACHE_DIR" \
    >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
done

for index in "${!GPUS[@]}"; do
  port=$((BASE_PORT + index))
  health="http://127.0.0.1:$port/health"
  deadline=$((SECONDS + 900))
  until curl --noproxy '*' --fail --silent --show-error --max-time 1 "$health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then
      status=0
      wait "${SERVER_PIDS[$index]}" || status=$?
      SERVER_PIDS[$index]=""
      echo "Error: server on port $port exited before readiness (status $status)" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Error: server on port $port did not become ready" >&2
      exit 1
    fi
    sleep 1
  done
  echo "Server ready on port $port"
done
for index in "${!SERVER_PIDS[@]}"; do
  if ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then
    echo "Error: server $index exited during startup" >&2
    exit 1
  fi
done

echo "Launching $NUM_SHARDS EGL simulator shards ($CLIENTS_PER_GPU per model server)"
ROLLOUT_START_NS="$(date +%s%N)"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  server_index=$((shard % NUM_GPUS))
  sim_gpu="${SIM_GPUS[$server_index]}"
  port=$((BASE_PORT + server_index))
  log="$LOCAL_DIR/shard-$(printf '%03d' "$shard").log"
  env -u LIBGL_ALWAYS_SOFTWARE \
    CUDA_VISIBLE_DEVICES="$sim_gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=device MUJOCO_EGL_DEVICE_ID="$sim_gpu" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 "$BENCH_ENV/bin/vla-eval" run \
    -c "$BENCHMARK_CONFIG" --eval-id "$EVAL_ID" --output-dir "$LOCAL_DIR" \
    --render gpu --server-url "ws://127.0.0.1:$port" \
    --param send_wrist_image=true --param send_state=true --param quat_no_antipodal=true \
    --no-docker --shard-id "$shard" --num-shards "$NUM_SHARDS" \
    >"$log" 2>&1 &
  SHARD_PIDS+=("$!")
done

failed=0
for index in "${!SHARD_PIDS[@]}"; do
  if ! wait "${SHARD_PIDS[$index]}"; then
    failed=$((failed + 1))
  fi
  SHARD_PIDS[$index]=""
done
ROLLOUT_END_NS="$(date +%s%N)"
if (( failed != 0 )); then
  echo "Error: $failed/$NUM_SHARDS shards failed; raw run retained at $LOCAL_DIR" >&2
  exit 1
fi
stop_servers

DB="$LOCAL_DIR/recording-$EVAL_ID.sqlite"
if [[ ! -f "$DB" ]]; then
  echo "Error: recording database missing: $DB" >&2
  exit 1
fi

"$BENCH_ENV/bin/python" - "$DB" <<'PY'
import json
import sqlite3
import sys

db = sys.argv[1]
expected = {
    "libero_plus_spatial": 2402,
    "libero_plus_object": 2518,
    "libero_plus_goal": 2591,
    "libero_plus_10": 2519,
}
con = sqlite3.connect(db)
checkpoint = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
if checkpoint[0] != 0 or checkpoint[1] != checkpoint[2]:
    raise SystemExit(f"SQLite WAL checkpoint failed: {checkpoint}")
if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("SQLite integrity_check failed")
rows = con.execute("SELECT eval_id, safe_name FROM eval_metadata ORDER BY safe_name").fetchall()
seen = set()
for eval_id, safe_name in rows:
    count = 0
    unique = set()
    errors = 0
    statuses = set()
    suites = set()
    task_ids = set()
    episode_ids = set()
    for task_name, episode_id, status, context in con.execute(
        "SELECT task_name, episode_id, status, context FROM episode_results WHERE eval_id = ?",
        (eval_id,),
    ):
        count += 1
        ctx = json.loads(context or "{}")
        suite = ctx.get("suite")
        task_id = ctx.get("task_id")
        unique.add((suite, task_id, episode_id))
        suites.add(suite)
        task_ids.add(task_id)
        statuses.add(status)
        episode_ids.add(episode_id)
        errors += int(status == "error")
    if len(suites) != 1:
        raise SystemExit(f"coverage failure {safe_name}: mixed suites={suites}")
    suite = suites.pop()
    suite_suffix = suite[len("libero_") :] if isinstance(suite, str) and suite.startswith("libero_") else suite
    suite_key = f"libero_plus_{suite_suffix}"
    expected_count = expected.get(suite_key)
    expected_ids = set(range(expected_count)) if expected_count is not None else set()
    if (
        expected_count is None
        or count != expected_count
        or len(unique) != count
        or task_ids != expected_ids
        or episode_ids != {0}
        or not statuses <= {"success", "fail", "error"}
    ):
        raise SystemExit(
            f"coverage failure {safe_name}: rows={count} unique={len(unique)} "
            f"task_ids={len(task_ids)} expected={expected_count} statuses={statuses} errors={errors}"
        )
    seen.add(suite_key)
    print(f"validated {safe_name}: rows={count} unique={len(unique)} errors={errors}")
if len(rows) != 4 or seen != set(expected):
    raise SystemExit(f"expected four suite metadata rows, got {seen}")
total = con.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0]
if total != sum(expected.values()):
    raise SystemExit(f"expected {sum(expected.values())} total rows, got {total}")
con.close()
PY

echo "Merging results to $OUTPUT_DIR"
"$BENCH_ENV/bin/vla-eval" merge --db "$DB" --output-dir "$OUTPUT_DIR" >"$LOCAL_DIR/merge.log" 2>&1
cp "$DB" "$OUTPUT_DIR/recording-$EVAL_ID.sqlite"
mkdir -p "$OUTPUT_DIR/logs"
cp "$LOCAL_DIR"/model-server-*.log "$OUTPUT_DIR/logs/"
cp "$LOCAL_DIR"/shard-*.log "$OUTPUT_DIR/logs/"
cp "$LOCAL_DIR/merge.log" "$OUTPUT_DIR/logs/"

END_NS="$(date +%s%N)"
END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WALL_SECONDS="$(( (END_NS - TOTAL_START_NS) / 1000000000 ))"
ROLLOUT_SECONDS="$(( ROLLOUT_OFFSET + (ROLLOUT_END_NS - ROLLOUT_START_NS) / 1000000000 ))"
METADATA_BENCHMARK_CONFIG="$BENCHMARK_CONFIG"
if [[ -n "$RESUME_LOCAL_DIR" ]]; then
  cp "$BENCHMARK_CONFIG" "$OUTPUT_DIR/resume_benchmark_config.yaml"
  METADATA_BENCHMARK_CONFIG="$OUTPUT_DIR/resume_benchmark_config.yaml"
fi
"$BENCH_ENV/bin/python" - \
  "$OUTPUT_DIR/run_metadata.json" "$OUTPUT_DIR/recording-$EVAL_ID.sqlite" \
  "$MODE" "${POLICY_NAME:-VJEPA-Policy}" "$EVAL_ID" "$START_UTC" "$END_UTC" \
  "$WALL_SECONDS" "$ROLLOUT_SECONDS" "$GPU_LIST" "$SIM_GPU_LIST" \
  "$NUM_GPUS" "$NUM_SHARDS" "$BATCH_SIZE" \
  "$ACTIVE_SERVER_CONFIG" "$METADATA_BENCHMARK_CONFIG" \
  "$TEXT_CACHE_DIR" "${CACHE_INVENTORY_OVERRIDE:-}" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(
    metadata_path,
    db_path,
    mode,
    policy_name,
    eval_id,
    started_at,
    finished_at,
    wall_seconds,
    rollout_seconds,
    gpu_list,
    sim_gpu_list,
    num_servers,
    num_shards,
    batch_size,
    server_config,
    benchmark_config,
    text_cache_dir,
    cache_inventory,
) = sys.argv[1:]
wall_seconds = int(wall_seconds)
rollout_seconds = int(rollout_seconds)
num_servers = int(num_servers)
num_shards = int(num_shards)
batch_size = int(batch_size)

db_uri = f"file:{Path(db_path).resolve()}?mode=ro&immutable=1"
con = sqlite3.connect(db_uri, uri=True)
status_counts = dict(
    con.execute("SELECT status, COUNT(*) FROM episode_results GROUP BY status")
)
total_episodes, episode_seconds = con.execute(
    "SELECT COUNT(*), COALESCE(SUM(elapsed_sec), 0) FROM episode_results"
).fetchone()
con.close()

metadata = {
    "mode": mode,
    "policy_name": policy_name,
    "eval_id": eval_id,
    "started_at": started_at,
    "finished_at": finished_at,
    "wall_seconds": wall_seconds,
    "rollout_wall_seconds": rollout_seconds,
    "gpus": [int(value) for value in gpu_list.split(",")],
    "simulator_gpus": [int(value) for value in sim_gpu_list.split(",")],
    "num_model_servers": num_servers,
    "num_simulator_shards": num_shards,
    "simulator_shards_per_server": [
        sum(shard % num_servers == index for shard in range(num_shards))
        for index in range(num_servers)
    ],
    "batch_size": batch_size,
    "clients_per_server": (
        num_shards // num_servers if num_shards % num_servers == 0 else None
    ),
    "render_backend": "egl",
    "shard_assignment": "global_round_robin",
    "total_episodes": total_episodes,
    "success": status_counts.get("success", 0),
    "fail": status_counts.get("fail", 0),
    "error": status_counts.get("error", 0),
    "success_rate": status_counts.get("success", 0) / total_episodes,
    "episode_elapsed_seconds_sum": episode_seconds,
    "episodes_per_minute": total_episodes * 60 / rollout_seconds,
    "effective_parallelism": episode_seconds / rollout_seconds,
    "server_config": str(Path(server_config).resolve()),
    "benchmark_config": str(Path(benchmark_config).resolve()),
    "text_cache_dir": str(Path(text_cache_dir).resolve()),
    "cache_inventory": str(Path(cache_inventory).resolve()) if cache_inventory else None,
}
Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n")
PY
echo "finished=$END_UTC wall_seconds=$WALL_SECONDS rollout_seconds=$ROLLOUT_SECONDS output=$OUTPUT_DIR"

# Let the monitor emit the final authoritative line before the temporary DB is
# removed.  It exits promptly after observing this marker.
: >"$MONITOR_DONE"
if [[ -n "$MONITOR_PID" ]]; then
  wait "$MONITOR_PID" 2>/dev/null || true
  MONITOR_PID=""
fi
cp "$MONITOR_LOG" "$OUTPUT_DIR/logs/"

case "$LOCAL_DIR" in
  /dev/shm/vjepa-libero-plus-*) rm -rf -- "$LOCAL_DIR" ;;
  *) echo "Error: refusing to clean unexpected local path: $LOCAL_DIR" >&2; exit 1 ;;
esac
