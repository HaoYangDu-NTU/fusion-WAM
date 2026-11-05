#!/usr/bin/env bash
set -euo pipefail

# Standard LIBERO evaluation with exactly one persistent server and one active
# suite client per GPU.  The client is restarted after a short episode batch to
# avoid the native MuJoCo/EGL lifetime abort observed on this host; tasks are not
# sharded and the original 0..49 init-state order is preserved.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

VARIANT=${1:?usage: eval_one_server_per_suite.sh VARIANT RUN_DIR [CHECKPOINT]}
RUN_DIR=${2:?usage: eval_one_server_per_suite.sh VARIANT RUN_DIR [CHECKPOINT]}
CHECKPOINT=${3:-$RUN_DIR/checkpoint_step021360.pt}
POLICY_PYTHON=${POLICY_PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/data/zhangyang/miniconda3/envs/libero/bin/python}
TEXT_EMBEDDING_CACHE=${TEXT_EMBEDDING_CACHE:-/data/zhangyang/workspace/VJEPA-WAM/data/text_embeds/libero_t5xxl_len128}
DATASET_STATS=${DATASET_STATS:-$RUN_DIR/dataset_stats.json}
EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_r16_50trials_seed7_egl_1server1client}
BASE_PORT=${BASE_PORT:-15100}
CLIENT_EPISODES_PER_PROCESS=${CLIENT_EPISODES_PER_PROCESS:-3}
EVAL_SEED=${EVAL_SEED:-7}
REPLAN_STEPS=${REPLAN_STEPS:-16}
TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
WEIGHTS=/data/zhangyang/workspace/VJEPA-WAM/pretrained_weights/hub/checkpoints
IFS=',' read -r -a GPU_IDS <<<"${EVAL_GPUS:-0,1,2,3}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

case "$VARIANT" in
  vjepa2_vitg)
    ENCODER_FAMILY=vjepa2
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=target_encoder
    ENCODER_CHECKPOINT="$WEIGHTS/vitg.pt"
    ;;
  vjepa2_1_vitl)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=ema_encoder
    ENCODER_CHECKPOINT="$WEIGHTS/vjepa2_1_vitl_dist_vitG_384.pt"
    ;;
  vjepa2_1_vitg)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=target_encoder
    ENCODER_CHECKPOINT="$WEIGHTS/vjepa2_1_vitg_384.pt"
    ;;
  *)
    echo "Unknown encoder variant: $VARIANT" >&2
    exit 64
    ;;
esac

if (( ${#GPU_IDS[@]} != 4 )); then
  echo "EVAL_GPUS must contain exactly four GPU IDs" >&2
  exit 64
fi
if [[ ! "$CLIENT_EPISODES_PER_PROCESS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLIENT_EPISODES_PER_PROCESS must be positive" >&2
  exit 64
fi
for required in "$CHECKPOINT" "$DATASET_STATS" "$ENCODER_CHECKPOINT"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$EVAL_DIR"
SUMMARY_LOG="$EVAL_DIR/summary.txt"

write_suite_result() {
  local suite=$1
  "$POLICY_PYTHON" - "$suite" "$TRIALS_PER_TASK" "$EVAL_DIR" <<'PY'
from pathlib import Path
import re
import sys

suite, expected_trials, eval_dir = sys.argv[1:]
expected_trials = int(expected_trials)
eval_dir = Path(eval_dir)
rows = []
for task_id in range(10):
    task_dir = eval_dir / suite / f"task{task_id:02d}"
    successes = 0
    episodes = 0
    description = None
    for result_path in sorted(task_dir.glob("batch*/libero_*_eval_results.txt")):
        text = result_path.read_text(encoding="utf-8")
        success_match = re.search(r"^Total success: (\d+)$", text, re.MULTILINE)
        episode_match = re.search(r"^Total episodes: (\d+)$", text, re.MULTILINE)
        row_match = re.search(
            r"^  \[00\] \d+/\d+ \([0-9.]+\) - (.*)$", text, re.MULTILINE
        )
        if not (success_match and episode_match and row_match):
            raise SystemExit(f"invalid batch result: {result_path}")
        successes += int(success_match.group(1))
        episodes += int(episode_match.group(1))
        current_description = row_match.group(1)
        if description is None:
            description = current_description
        elif description != current_description:
            raise SystemExit(f"task description mismatch in {task_dir}")
    if episodes != expected_trials:
        raise SystemExit(
            f"{suite} task {task_id}: {episodes} episodes, expected {expected_trials}"
        )
    rows.append((task_id, successes, episodes, description))

total_success = sum(row[1] for row in rows)
total_episodes = sum(row[2] for row in rows)
lines = [
    f"Task suite name: {suite}",
    f"Total success rate: {total_success / total_episodes}",
    f"Total episodes: {total_episodes}",
    f"Total success: {total_success}",
    "Task results:",
]
for task_id, successes, episodes, description in rows:
    lines.append(
        f"  [{task_id:02d}] {successes}/{episodes} "
        f"({successes / episodes:.4f}) - {description}"
    )
output = eval_dir / suite / f"{suite}_eval_results.txt"
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
temporary.replace(output)
PY
}

run_suite() (
  set -uo pipefail
  local suite=$1
  local gpu=$2
  local port=$3
  local suite_dir="$EVAL_DIR/$suite"
  local results_file="$suite_dir/${suite}_eval_results.txt"
  local server_log="$EVAL_DIR/${suite}_server.log"
  local runner_log="$EVAL_DIR/${suite}_runner.log"
  local status_file="$EVAL_DIR/${suite}.status"
  local server_pid=""

  cleanup() {
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  if [[ -s "$results_file" ]]; then
    echo 0 >"$status_file"
    echo "[$(date -Is)] [$suite] reusing completed $results_file" >>"$runner_log"
    exit 0
  fi

  mkdir -p "$suite_dir"
  echo running >"$status_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$POLICY_PYTHON" examples/libero/serve_policy.py \
    --ckpt "$CHECKPOINT" \
    --pretrained-encoder "$ENCODER_CHECKPOINT" \
    --encoder-family "$ENCODER_FAMILY" \
    --model-name "$ENCODER_MODEL_NAME" \
    --encoder-checkpoint-key "$ENCODER_CHECKPOINT_KEY" \
    --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
    --dataset-stats "$DATASET_STATS" \
    --crop-size 224 \
    --view-layout independent \
    --corrected-predictor-rope \
    --encoder-interpolate-rope \
    --pred-depth 24 \
    --pred-embed-dim 1024 \
    --pred-num-heads 16 \
    --action-hidden-size 512 \
    --action-num-layers 24 \
    --precision bf16 \
    --t5-len 128 \
    --seed "$EVAL_SEED" \
    --port "$port" \
    >"$server_log" 2>&1 &
  server_pid=$!

  local ready=false
  for _ in $(seq 1 300); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[$suite] server exited during startup" >>"$runner_log"
      tail -100 "$server_log" >>"$runner_log"
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
    echo "[$suite] server readiness timed out" >>"$runner_log"
    echo 1 >"$status_file"
    exit 1
  fi

  for task_id in $(seq 0 9); do
    for offset in $(seq 0 "$CLIENT_EPISODES_PER_PROCESS" $((TRIALS_PER_TASK - 1))); do
      local remaining=$((TRIALS_PER_TASK - offset))
      local count=$CLIENT_EPISODES_PER_PROCESS
      if (( remaining < count )); then
        count=$remaining
      fi
      local batch_dir="$suite_dir/task$(printf '%02d' "$task_id")/batch$(printf '%02d' "$offset")"
      local batch_result="$batch_dir/${suite}_eval_results.txt"
      local client_log="$batch_dir/client.log"
      mkdir -p "$batch_dir"
      if [[ -s "$batch_result" ]]; then
        continue
      fi
      echo "[$(date -Is)] task=$task_id offset=$offset count=$count" >>"$runner_log"
      set +e
      env -u LIBGL_ALWAYS_SOFTWARE \
        LIBERO_FIXED_TASK_ID="$task_id" \
        LIBERO_EPISODE_OFFSET="$offset" \
        LIBERO_FIXED_SUITE="$suite" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        EGL_PLATFORM=device \
        MUJOCO_EGL_DEVICE_ID="$gpu" \
        PYTHONUNBUFFERED=1 \
        "$LIBERO_PYTHON" scripts/encoder_ablation/run_libero_task_batch.py \
          --args.host 127.0.0.1 \
          --args.port "$port" \
          --args.task-suite-name "$suite" \
          --args.replan-steps "$REPLAN_STEPS" \
          --args.frame-stride 4 \
          --args.resize-size 256 \
          --args.num-trials-per-task "$count" \
          --args.seed "$EVAL_SEED" \
          --args.video-out-path "$batch_dir" \
          >"$client_log" 2>&1
      local client_rc=$?
      set -e
      if (( client_rc != 0 )) || [[ ! -s "$batch_result" ]]; then
        echo "[$(date -Is)] task=$task_id offset=$offset failed rc=$client_rc" >>"$runner_log"
        tail -100 "$client_log" >>"$runner_log"
        echo "$client_rc" >"$status_file"
        exit "${client_rc:-1}"
      fi
    done
  done

  if write_suite_result "$suite" >>"$runner_log" 2>&1; then
    echo 0 >"$status_file"
    exit 0
  fi
  echo 2 >"$status_file"
  exit 2
)

{
  echo "LIBERO evaluation"
  echo "checkpoint: $CHECKPOINT"
  echo "variant: $VARIANT"
  echo "started: $(date -Is)"
  echo "topology: one server + one active suite client per GPU"
  echo "server_gpus: ${GPU_IDS[*]}"
  echo "simulator_gpus: ${GPU_IDS[*]}"
  echo "suite_mapping: spatial=${GPU_IDS[0]} object=${GPU_IDS[1]} goal=${GPU_IDS[2]} libero_10=${GPU_IDS[3]}"
  echo "task_shards_per_suite: 1"
  echo "client_episode_batch: $CLIENT_EPISODES_PER_PROCESS"
  echo "mujoco_gl: egl"
  echo "t5_len: 128"
  echo "crop_size: 224"
  echo "replan_steps: $REPLAN_STEPS"
  echo "frame_stride: 4"
  echo "trials_per_task: $TRIALS_PER_TASK"
  echo "seed: $EVAL_SEED"
  echo
} >"$SUMMARY_LOG"

worker_pids=()
for index in "${!SUITES[@]}"; do
  run_suite "${SUITES[$index]}" "${GPU_IDS[$index]}" "$((BASE_PORT + index))" &
  worker_pids+=("$!")
  sleep 3
done

worker_failed=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    worker_failed=1
  fi
done

total_success=0
total_episodes=0
failed=()
for suite in "${SUITES[@]}"; do
  results_file="$EVAL_DIR/$suite/${suite}_eval_results.txt"
  suite_status=missing
  if [[ -s "$results_file" ]]; then
    suite_status=0
  elif [[ -f "$EVAL_DIR/${suite}.status" ]]; then
    suite_status=$(<"$EVAL_DIR/${suite}.status")
  fi
  {
    echo "[$suite] exit_status: $suite_status"
    if [[ -s "$results_file" ]]; then
      sed 's/^/  /' "$results_file"
    else
      echo "  missing results: $results_file"
    fi
    echo
  } >>"$SUMMARY_LOG"
  if [[ "$suite_status" != 0 ]] || [[ ! -s "$results_file" ]]; then
    failed+=("$suite")
    continue
  fi
  successes=$(awk -F': ' '$1 == "Total success" {print $2}' "$results_file")
  episodes=$(awk -F': ' '$1 == "Total episodes" {print $2}' "$results_file")
  total_success=$((total_success + successes))
  total_episodes=$((total_episodes + episodes))
done

{
  echo "finished: $(date -Is)"
  echo "successful_suites: $((4 - ${#failed[@]}))/4"
  echo "failed_suites: ${failed[*]:-none}"
  if (( total_episodes > 0 )); then
    awk -v successes="$total_success" -v episodes="$total_episodes" \
      'BEGIN {printf "overall: %d/%d = %.2f%%\n", successes, episodes, 100 * successes / episodes}'
  fi
} >>"$SUMMARY_LOG"

if (( worker_failed != 0 || ${#failed[@]} != 0 )); then
  echo "LIBERO evaluation incomplete: ${failed[*]:-worker failure}" >&2
  exit 1
fi
echo "[done] $SUMMARY_LOG"
