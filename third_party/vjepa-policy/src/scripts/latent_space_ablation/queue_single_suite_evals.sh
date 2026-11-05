#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

SNAPSHOT=$(realpath -e "${1:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}")
CURRENT_EVAL_PID=${2:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}
ACTION_V2=$(realpath -e "${3:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}")
ACTION_V21=$(realpath -e "${4:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}")
DINO_RUN=$(realpath -m "${5:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}")
WAN_RUN=$(realpath -m "${6:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}")
TRAIN_CHAIN_PID=${7:?usage: queue_single_suite_evals.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO_RUN WAN_RUN TRAIN_CHAIN_PID}
PORT=${EVAL_PORT_START:-15810}
IFS=',' read -r -a GPUS <<<"${EVAL_GPU_POOL:-0,1,2,3}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

source "$SNAPSHOT/scripts/ablation_runtime.sh"
export PYTHONDONTWRITEBYTECODE=1

if (( ${#GPUS[@]} != ${#SUITES[@]} )); then
  echo "EVAL_GPU_POOL must contain exactly four comma-separated GPU IDs" >&2
  exit 64
fi
if [[ ! "$CURRENT_EVAL_PID" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$TRAIN_CHAIN_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Current evaluation and training-chain PIDs must be positive integers" >&2
  exit 64
fi
CURRENT_EVAL_START=$(process_start_time "$CURRENT_EVAL_PID" || true)
TRAIN_CHAIN_START=$(process_start_time "$TRAIN_CHAIN_PID" || true)
if [[ -z "$TRAIN_CHAIN_START" ]]; then
  echo "Training chain is not running: $TRAIN_CHAIN_PID" >&2
  exit 1
fi
(cd "$SNAPSHOT" && sha256sum --check --quiet SHA256SUMS)

wait_for_run() {
  local run_dir=$1
  local final_checkpoint=$run_dir/checkpoint_step021360.pt
  local train_pid
  local train_start

  while [[ ! -s "$run_dir/train.pid" || ! -s "$run_dir/train.start" ]]; do
    if ! process_matches_start_time "$TRAIN_CHAIN_PID" "$TRAIN_CHAIN_START"; then
      echo "Training chain exited before creating $run_dir/train.pid" >&2
      exit 1
    fi
    sleep 30
  done
  train_pid=$(<"$run_dir/train.pid")
  train_start=$(<"$run_dir/train.start")
  while [[ ! -s "$final_checkpoint" ]]; do
    if ! process_matches_start_time "$train_pid" "$train_start"; then
      echo "Training pid $train_pid exited without $final_checkpoint" >&2
      exit 1
    fi
    sleep 60
  done
  while process_matches_start_time "$train_pid" "$train_start"; do
    sleep 10
  done
}

run_single_suite() {
  local kind=$1
  local encoder=$2
  local run_dir=$3
  local suite=$4
  local gpu=$5
  local eval_dir=$run_dir/libero_eval_r16_50trials_seed7
  local script
  local attempt

  if [[ "$kind" == action ]]; then
    script=$SNAPSHOT/scripts/encoder_ablation/eval.sh
  else
    script=$SNAPSHOT/scripts/latent_space_ablation/eval.sh
  fi
  for attempt in 1 2 3; do
    printf '[%s] kind=%s encoder=%s suite=%s gpu=%s port=%s attempt=%s\n' \
      "$(date -Is)" "$kind" "$encoder" "$suite" "$gpu" "$PORT" "$attempt"
    if EVAL_SUITES="$suite" \
      EVAL_GPUS="$gpu" \
      BASE_PORT="$PORT" \
      EVAL_DIR="$eval_dir" \
      bash "$script" "$encoder" "$run_dir" \
        "$run_dir/checkpoint_step021360.pt"; then
      PORT=$((PORT + 1))
      return 0
    fi
    sleep 15
  done
  echo "Suite failed after three attempts: $kind $encoder $suite" >&2
  return 1
}

run_group() {
  local kind=$1
  local encoder=$2
  local run_dir=$3
  local index

  for index in "${!SUITES[@]}"; do
    run_single_suite \
      "$kind" "$encoder" "$run_dir" "${SUITES[$index]}" "${GPUS[$index]}"
  done
  grep -Fxq 'successful_suites: 4/4' \
    "$run_dir/libero_eval_r16_50trials_seed7/summary.txt"
}

if [[ -n "$CURRENT_EVAL_START" ]]; then
  printf '[%s] waiting for current action evaluation pid=%s\n' \
    "$(date -Is)" "$CURRENT_EVAL_PID"
  while process_matches_start_time "$CURRENT_EVAL_PID" "$CURRENT_EVAL_START"; do
    sleep 30
  done
fi

run_group action vjepa2_vitl "$ACTION_V2"
run_group action vjepa2_1_vitl "$ACTION_V21"

wait_for_run "$DINO_RUN"
run_group latent dinov3_vitl "$DINO_RUN"

wait_for_run "$WAN_RUN"
run_group latent wan22_vae38 "$WAN_RUN"

printf '[%s] all queued evaluations complete\n' "$(date -Is)"
