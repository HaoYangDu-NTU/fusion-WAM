#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

RUN_DIR=${1:?usage: evaluate_policy.sh RUN_DIR [CHECKPOINT]}
CHECKPOINT=${2:-$RUN_DIR/checkpoint_step021360.pt}
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE}"

POLICY_PYTHON=${POLICY_PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/data/zhangyang/miniconda3/envs/libero/bin/python}
LIBERO_MUJOCO_GL=${LIBERO_MUJOCO_GL:-egl}
POLICY_SERVER_SCRIPT=${POLICY_SERVER_SCRIPT:-examples/libero/serve_policy.py}
EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_r16_50trials}
DATASET_STATS=${DATASET_STATS:-$RUN_DIR/dataset_stats.json}
BASE_PORT=${BASE_PORT:-14001}
REPLAN_STEPS=${REPLAN_STEPS:-16}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
EVAL_SEED=${EVAL_SEED:-7}
POLICY_PRECISION=${POLICY_PRECISION:-bf16}
POLICY_T5_LEN=${POLICY_T5_LEN:-32}
POLICY_CROP_SIZE=${POLICY_CROP_SIZE:-224}
POLICY_VIEW_LAYOUT=${POLICY_VIEW_LAYOUT:-independent}
POLICY_ENCODER_INTERPOLATE_ROPE=${POLICY_ENCODER_INTERPOLATE_ROPE:-1}
POLICY_CORRECTED_PREDICTOR_ROPE=${POLICY_CORRECTED_PREDICTOR_ROPE:-1}
ENCODER_FAMILY=${ENCODER_FAMILY:-vjepa2}
ENCODER_MODEL_NAME=${ENCODER_MODEL_NAME:-vit_giant_xformers}
ENCODER_CHECKPOINT_KEY=${ENCODER_CHECKPOINT_KEY:-encoder}
PRED_DEPTH=${PRED_DEPTH:-24}
PRED_EMBED_DIM=${PRED_EMBED_DIM:-1024}
PRED_NUM_HEADS=${PRED_NUM_HEADS:-16}
ACTION_HIDDEN_SIZE=${ACTION_HIDDEN_SIZE:-512}
ACTION_NUM_LAYERS=${ACTION_NUM_LAYERS:-24}
LIBERO_SHARDS_PER_SUITE=${LIBERO_SHARDS_PER_SUITE:-1}
EVAL_GPU_LIST=${EVAL_GPUS:-0,1,2,3}
EVAL_SIM_GPU_LIST=${EVAL_SIM_GPUS:-$EVAL_GPU_LIST}
IFS=',' read -r -a GPU_IDS <<<"$EVAL_GPU_LIST"
IFS=',' read -r -a SIM_GPU_IDS <<<"$EVAL_SIM_GPU_LIST"
ALL_SUITES=(libero_spatial libero_object libero_goal libero_10)
IFS=',' read -r -a SUITES <<<"${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing policy checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -s "$DATASET_STATS" ]]; then
  echo "Missing normalization statistics: $DATASET_STATS" >&2
  exit 1
fi
if [[ ! -f "$POLICY_SERVER_SCRIPT" ]]; then
  echo "Missing policy server script: $POLICY_SERVER_SCRIPT" >&2
  exit 1
fi
if (( ${#SUITES[@]} == 0 )); then
  echo "EVAL_SUITES must contain at least one suite" >&2
  exit 64
fi
declare -A requested_suites=()
for suite in "${SUITES[@]}"; do
  if [[ ! " ${ALL_SUITES[*]} " =~ " $suite " ]]; then
    echo "Unknown LIBERO suite: $suite" >&2
    exit 64
  fi
  if [[ -v "requested_suites[$suite]" ]]; then
    echo "EVAL_SUITES must not contain duplicates: $suite" >&2
    exit 64
  fi
  requested_suites[$suite]=1
done
if [[ ! "$LIBERO_SHARDS_PER_SUITE" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIBERO_SHARDS_PER_SUITE must be a positive integer" >&2
  exit 64
fi
num_workers=$((${#SUITES[@]} * LIBERO_SHARDS_PER_SUITE))
if (( ${#GPU_IDS[@]} < num_workers )); then
  echo "EVAL_GPUS must contain at least $num_workers comma-separated GPU IDs" >&2
  exit 1
fi
if (( ${#SIM_GPU_IDS[@]} < num_workers )); then
  echo "EVAL_SIM_GPUS must contain at least $num_workers comma-separated GPU IDs" >&2
  exit 1
fi
mkdir -p "$EVAL_DIR"
SUMMARY_LOG="$EVAL_DIR/summary.txt"

{
  echo "LIBERO evaluation"
  echo "checkpoint: $CHECKPOINT"
  echo "policy_server_script: $POLICY_SERVER_SCRIPT"
  echo "started: $(date -Is)"
  echo "replan_steps: $REPLAN_STEPS"
  echo "frame_stride: 4"
  echo "trials_per_task: $NUM_TRIALS_PER_TASK"
  echo "seed: $EVAL_SEED"
  echo "mujoco_gl: $LIBERO_MUJOCO_GL"
  echo "t5_len: $POLICY_T5_LEN"
  echo "requested_suites: ${SUITES[*]}"
  echo "task_shards_per_suite: $LIBERO_SHARDS_PER_SUITE"
  echo "server_gpus: ${GPU_IDS[*]:0:num_workers}"
  echo "simulator_gpus: ${SIM_GPU_IDS[*]:0:num_workers}"
  echo "view_layout: $POLICY_VIEW_LAYOUT"
  echo "crop_size: $POLICY_CROP_SIZE"
  echo "encoder_endpoint_rope: $POLICY_ENCODER_INTERPOLATE_ROPE"
  echo "corrected_predictor_rope: $POLICY_CORRECTED_PREDICTOR_ROPE"
  echo
} >"$SUMMARY_LOG"

run_suite() (
  set -uo pipefail
  suite=$1
  server_gpu=$2
  sim_gpu=$3
  port=$4
  task_shard_id=$5
  num_task_shards=$6
  if (( num_task_shards == 1 )); then
    worker_name=$suite
    shard_dir="$EVAL_DIR/$suite"
  else
    worker_name="${suite}_shard${task_shard_id}"
    shard_dir="$EVAL_DIR/$suite/shard${task_shard_id}"
  fi
  server_log="$EVAL_DIR/${worker_name}_server.log"
  client_log="$EVAL_DIR/${worker_name}.log"
  status_file="$EVAL_DIR/${worker_name}.status"
  results_file="$shard_dir/${suite}_eval_results.txt"
  server_pid=""

  cleanup() {
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  if [[ -s "$results_file" ]]; then
    echo 0 >"$status_file"
    echo "[$(date -Is)] [$worker_name] reusing $results_file"
    exit 0
  fi

  rope_args=()
  if [[ "$POLICY_CORRECTED_PREDICTOR_ROPE" == 1 ]]; then
    rope_args+=(--corrected-predictor-rope)
  else
    rope_args+=(--no-corrected-predictor-rope)
  fi
  if [[ "$POLICY_ENCODER_INTERPOLATE_ROPE" == 1 ]]; then
    rope_args+=(--encoder-interpolate-rope)
  else
    rope_args+=(--no-encoder-interpolate-rope)
  fi

  echo running >"$status_file"
  CUDA_VISIBLE_DEVICES="$server_gpu" "$POLICY_PYTHON" "$POLICY_SERVER_SCRIPT" \
    --ckpt "$CHECKPOINT" \
    --pretrained-encoder "$VJEPA2_ENCODER_CHECKPOINT" \
    --encoder-family "$ENCODER_FAMILY" \
    --model-name "$ENCODER_MODEL_NAME" \
    --encoder-checkpoint-key "$ENCODER_CHECKPOINT_KEY" \
    --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
    --dataset-stats "$DATASET_STATS" \
    --crop-size "$POLICY_CROP_SIZE" \
    --view-layout "$POLICY_VIEW_LAYOUT" \
    "${rope_args[@]}" \
    --pred-depth "$PRED_DEPTH" \
    --pred-embed-dim "$PRED_EMBED_DIM" \
    --pred-num-heads "$PRED_NUM_HEADS" \
    --action-hidden-size "$ACTION_HIDDEN_SIZE" \
    --action-num-layers "$ACTION_NUM_LAYERS" \
    --precision "$POLICY_PRECISION" \
    --t5-len "$POLICY_T5_LEN" \
    --seed "$EVAL_SEED" \
    --port "$port" \
    >"$server_log" 2>&1 &
  server_pid=$!

  ready=false
  for _ in $(seq 1 300); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[$worker_name] policy server exited during startup" >&2
      tail -100 "$server_log" >&2
      echo 1 >"$status_file"
      exit 1
    fi
    if curl --noproxy '*' -fs --max-time 1 "http://127.0.0.1:$port/healthz" \
      >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then
    echo "[$worker_name] policy server did not become ready" >&2
    tail -100 "$server_log" >&2
    echo 1 >"$status_file"
    exit 1
  fi

  set +e
  env -u LIBGL_ALWAYS_SOFTWARE \
    CUDA_VISIBLE_DEVICES="$sim_gpu" MUJOCO_GL="$LIBERO_MUJOCO_GL" \
    PYOPENGL_PLATFORM=egl EGL_PLATFORM=device MUJOCO_EGL_DEVICE_ID="$sim_gpu" \
    "$LIBERO_PYTHON" \
    examples/libero/run_libero_client.py \
      --args.host 127.0.0.1 \
      --args.port "$port" \
      --args.task-suite-name "$suite" \
      --args.replan-steps "$REPLAN_STEPS" \
      --args.frame-stride 4 \
      --args.resize-size 256 \
      --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
      --args.task-shard-id "$task_shard_id" \
      --args.num-task-shards "$num_task_shards" \
      --args.seed "$EVAL_SEED" \
      --args.video-out-path "$shard_dir" \
      >"$client_log" 2>&1
  client_status=$?
  set -e

  if (( client_status == 0 )) && [[ -s "$results_file" ]]; then
    echo 0 >"$status_file"
    exit 0
  fi
  if (( client_status == 0 )); then
    client_status=2
  fi
  echo "$client_status" >"$status_file"
  exit "$client_status"
)

merge_suite_shards() {
  suite=$1
  output_file="$EVAL_DIR/$suite/${suite}_eval_results.txt"
  shard_results=()
  for task_shard_id in $(seq 0 $((LIBERO_SHARDS_PER_SUITE - 1))); do
    worker_name="${suite}_shard${task_shard_id}"
    status_file="$EVAL_DIR/${worker_name}.status"
    result_file="$EVAL_DIR/$suite/shard${task_shard_id}/${suite}_eval_results.txt"
    if [[ ! -f "$status_file" || "$(<"$status_file")" != 0 || ! -s "$result_file" ]]; then
      echo "[$suite] incomplete shard $task_shard_id: $result_file" >&2
      return 1
    fi
    shard_results+=("$result_file")
  done

  "$POLICY_PYTHON" - "$suite" "$NUM_TRIALS_PER_TASK" "$output_file" "${shard_results[@]}" <<'PY'
from pathlib import Path
import re
import sys

suite, trials, output, *inputs = sys.argv[1:]
trials = int(trials)
row_pattern = re.compile(
    r"^  \[(\d+)\] (\d+)/(\d+) \(([0-9.]+)\) - (.*)$"
)
tasks = {}
for input_path in inputs:
    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        match = row_pattern.match(line)
        if match is None:
            continue
        task_id, successes, episodes, _, description = match.groups()
        task_id = int(task_id)
        if task_id in tasks:
            raise SystemExit(f"duplicate {suite} task id {task_id}")
        successes, episodes = int(successes), int(episodes)
        if episodes != trials:
            raise SystemExit(
                f"{suite} task {task_id} has {episodes} episodes, expected {trials}"
            )
        tasks[task_id] = (successes, episodes, description)

expected_ids = set(range(10))
if set(tasks) != expected_ids:
    raise SystemExit(
        f"{suite} task coverage mismatch: got {sorted(tasks)}, expected {sorted(expected_ids)}"
    )
total_success = sum(row[0] for row in tasks.values())
total_episodes = sum(row[1] for row in tasks.values())
lines = [
    f"Task suite name: {suite}",
    f"Total success rate: {total_success / total_episodes}",
    f"Total episodes: {total_episodes}",
    f"Total success: {total_success}",
    "Task results:",
]
for task_id in sorted(tasks):
    successes, episodes, description = tasks[task_id]
    lines.append(
        f"  [{task_id:02d}] {successes}/{episodes} "
        f"({successes / episodes:.4f}) - {description}"
    )
output_path = Path(output)
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
temporary.replace(output_path)
PY
}

worker_pids=()
worker_index=0
for suite in "${SUITES[@]}"; do
  for task_shard_id in $(seq 0 $((LIBERO_SHARDS_PER_SUITE - 1))); do
    if (( LIBERO_SHARDS_PER_SUITE == 1 )); then
      worker_name=$suite
    else
      worker_name="${suite}_shard${task_shard_id}"
    fi
    run_suite \
      "$suite" \
      "${GPU_IDS[$worker_index]}" \
      "${SIM_GPU_IDS[$worker_index]}" \
      "$((BASE_PORT + worker_index))" \
      "$task_shard_id" \
      "$LIBERO_SHARDS_PER_SUITE" \
      >"$EVAL_DIR/${worker_name}_runner.log" 2>&1 &
    worker_pids+=("$!")
    worker_index=$((worker_index + 1))
    sleep 2
  done
done

set +e
for worker_pid in "${worker_pids[@]}"; do
  wait "$worker_pid"
done
set -e

if (( LIBERO_SHARDS_PER_SUITE > 1 )); then
  for suite in "${SUITES[@]}"; do
    merge_suite_shards "$suite" || true
  done
fi

failed=()
requested_failed=()
total_success=0
total_episodes=0
for suite in "${ALL_SUITES[@]}"; do
  results_file="$EVAL_DIR/$suite/${suite}_eval_results.txt"
  status=missing
  if [[ -s "$results_file" ]]; then
    status=0
  elif [[ -f "$EVAL_DIR/${suite}.status" ]]; then
    status=$(<"$EVAL_DIR/${suite}.status")
  fi
  {
    echo "[$suite] exit_status: $status"
    if [[ -s "$results_file" ]]; then
      sed 's/^/  /' "$results_file"
    else
      echo "  missing results: $results_file"
    fi
    echo
  } >>"$SUMMARY_LOG"
  if [[ "$status" != 0 ]] || [[ ! -s "$results_file" ]]; then
    failed+=("$suite")
    if [[ -v "requested_suites[$suite]" ]]; then
      requested_failed+=("$suite")
    fi
    continue
  fi
  successes=$(awk -F': ' '$1 == "Total success" {print $2}' "$results_file")
  episodes=$(awk -F': ' '$1 == "Total episodes" {print $2}' "$results_file")
  total_success=$((total_success + successes))
  total_episodes=$((total_episodes + episodes))
done

{
  echo "finished: $(date -Is)"
  echo "successful_suites: $((${#ALL_SUITES[@]} - ${#failed[@]}))/${#ALL_SUITES[@]}"
  echo "failed_suites: ${failed[*]:-none}"
  if (( total_episodes > 0 )); then
    awk -v successes="$total_success" -v episodes="$total_episodes" \
      'BEGIN {printf "overall: %d/%d = %.2f%%\n", successes, episodes, 100 * successes / episodes}'
  fi
} >>"$SUMMARY_LOG"

if (( ${#requested_failed[@]} > 0 )); then
  echo "Requested LIBERO evaluation incomplete: ${requested_failed[*]}" >&2
  exit 1
fi

echo "[done] LIBERO evaluation summary: $SUMMARY_LOG"
