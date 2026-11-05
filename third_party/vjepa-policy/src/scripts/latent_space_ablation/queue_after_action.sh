#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
SOURCE_ROOT=$(realpath -e "${SOURCE_ROOT:-$ROOT}")
source "$SOURCE_ROOT/scripts/ablation_runtime.sh"
verify_source_snapshot "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1

ACTION_QUEUE_PID=${1:?usage: queue_after_action.sh ACTION_QUEUE_PID ACTION_RUN_DIR [RUN_ROOT]}
ACTION_RUN_DIR=$(realpath -m "${2:?usage: queue_after_action.sh ACTION_QUEUE_PID ACTION_RUN_DIR [RUN_ROOT]}")
RUN_ROOT=$(realpath -m "${3:-$ROOT/runs}")
GPU_IDS=0,1,2,3
GPUS=(0 1 2 3)

if [[ ! "$ACTION_QUEUE_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid action queue PID: $ACTION_QUEUE_PID" >&2
  exit 64
fi
mkdir -p "$RUN_ROOT"
echo $$ >"$RUN_ROOT/latent_after_action_queue.pid"

wait_for_eval_pid_file() {
  local controller_pid=$1
  local run_dir=$2
  local label=$3
  local pid_file=$run_dir/eval_queue_launcher.pid
  local dead_checks=0
  local controller_start

  controller_start=$(process_start_time "$controller_pid")

  while [[ ! -s "$pid_file" ]]; do
    if process_matches_start_time "$controller_pid" "$controller_start"; then
      dead_checks=0
    else
      dead_checks=$((dead_checks + 1))
      if (( dead_checks >= 3 )); then
        echo "$label controller $controller_pid exited without $pid_file" >&2
        return 1
      fi
    fi
    sleep 60
  done
}

wait_for_eval_completion() {
  local run_dir=$1
  local label=$2
  local eval_pid
  local eval_start
  local summary=$run_dir/libero_eval_r16_50trials_seed7/summary.txt

  eval_pid=$(<"$run_dir/eval_queue_launcher.pid")
  if [[ ! "$eval_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid $label evaluation PID: $eval_pid" >&2
    return 1
  fi
  if [[ -s "$run_dir/eval_queue_launcher.start" ]]; then
    eval_start=$(<"$run_dir/eval_queue_launcher.start")
  else
    eval_start=$(process_start_time "$eval_pid")
  fi
  echo "[$(date -Is)] waiting for $label evaluation PID $eval_pid"
  while process_matches_start_time "$eval_pid" "$eval_start"; do
    sleep 60
  done
  if [[ ! -s "$summary" ]] || ! grep -Fxq 'successful_suites: 4/4' "$summary"; then
    echo "$label evaluation did not complete successfully: $summary" >&2
    return 1
  fi
}

check_gpu_health() {
  for gpu in "${GPUS[@]}"; do
    uncorrected=$(nvidia-smi --id="$gpu" \
      --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader,nounits | tr -d ' ')
    recovery=$(nvidia-smi -q -i "$gpu" | \
      awk -F ': ' '/GPU Recovery Action/ && !found {print $2; found=1}' | xargs)
    if [[ "$uncorrected" != 0 || "$recovery" != None ]]; then
      echo "GPU $gpu is unhealthy: uncorrected_ecc=$uncorrected recovery=$recovery" >&2
      return 1
    fi
  done
}

wait_for_healthy_idle_gpus() {
  check_gpu_health

  while :; do
    busy=()
    for gpu in "${GPUS[@]}"; do
      used=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
        --format=csv,noheader | sed '/^[[:space:]]*$/d')
      if [[ -n "$used" ]]; then
        busy+=("$gpu:$used")
      fi
    done
    if (( ${#busy[@]} == 0 )); then
      check_gpu_health
      return 0
    fi
    echo "[$(date -Is)] waiting for GPUs $GPU_IDS: ${busy[*]}"
    sleep 30
  done
}

run_smoke() {
  local encoder=$1
  local run_dir=$2
  local port=$3
  local acb=$4
  local image_microbatch=$5
  local video_microbatch=$6

  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty smoke directory: $run_dir" >&2
    return 1
  fi
  printf '[%s] smoke encoder=%s acb=%s image_microbatch=%s video_microbatch=%s\n' \
    "$(date -Is)" "$encoder" "$acb" "$image_microbatch" "$video_microbatch"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    MAIN_PROCESS_PORT="$port" \
    OUTDIR="$run_dir" \
    SMOKE_NUM_GPUS=4 \
    SMOKE_MAX_STEPS=10 \
    ACTIVATION_CHECKPOINTING_BLOCKS="$acb" \
    IMAGE_MICROBATCH_SIZE="$image_microbatch" \
    VIDEO_MICROBATCH_SIZE="$video_microbatch" \
    bash "$SOURCE_ROOT/scripts/latent_space_ablation/run.sh" \
      "$encoder" --smoke || return $?
  if [[ ! -s "$run_dir/checkpoint_step000010.pt" ]] || \
     ! rg -q 'step=10/10 loss=' "$run_dir/train.log" || \
     rg -qi '(loss|world|action)=([+-]?(nan|inf(inity)?))' "$run_dir/train.log"; then
    echo "Smoke did not produce a finite step-10 checkpoint: $run_dir" >&2
    return 1
  fi
}

launch_and_evaluate() {
  local encoder=$1
  local run_dir=$2
  local train_port=$3
  local eval_port=$4
  local acb=$5
  local image_microbatch=$6
  local video_microbatch=$7

  printf '[%s] formal encoder=%s acb=%s image_microbatch=%s video_microbatch=%s\n' \
    "$(date -Is)" "$encoder" "$acb" "$image_microbatch" "$video_microbatch"
  MAIN_PROCESS_PORT="$train_port" \
    EVAL_BASE_PORT="$eval_port" \
    ACTIVATION_CHECKPOINTING_BLOCKS="$acb" \
    IMAGE_MICROBATCH_SIZE="$image_microbatch" \
    VIDEO_MICROBATCH_SIZE="$video_microbatch" \
    bash "$SOURCE_ROOT/scripts/latent_space_ablation/launch_one.sh" \
      "$encoder" "$GPU_IDS" "$run_dir"
  wait_for_eval_pid_file "$(<"$run_dir/train.pid")" "$run_dir" "$encoder"
  wait_for_eval_completion "$run_dir" "$encoder"
}

echo "[$(date -Is)] waiting for V-JEPA2.1 action-only launch and evaluation"
wait_for_eval_pid_file "$ACTION_QUEUE_PID" "$ACTION_RUN_DIR" action-only-vjepa2.1
wait_for_eval_completion "$ACTION_RUN_DIR" action-only-vjepa2.1

wait_for_healthy_idle_gpus
dino_acb=7
dino_imb=320
dino_vmb=32
dino_smoke=$RUN_ROOT/preflight_latent_dinov2_vitl_4gpu_bs32_step10_acb7_imb320_seed7
if ! run_smoke dinov2_vitl "$dino_smoke" 29571 \
  "$dino_acb" "$dino_imb" "$dino_vmb"; then
  if [[ -s "$dino_smoke/train.log" ]] && \
     rg -q 'CUDA out of memory|OutOfMemoryError' "$dino_smoke/train.log"; then
    dino_acb=8
    dino_imb=160
    dino_smoke=$RUN_ROOT/preflight_latent_dinov2_vitl_4gpu_bs32_step10_acb8_imb160_seed7
    wait_for_healthy_idle_gpus
    run_smoke dinov2_vitl "$dino_smoke" 29575 \
      "$dino_acb" "$dino_imb" "$dino_vmb"
  else
    echo "DINOv2 smoke failed for a non-OOM reason" >&2
    exit 1
  fi
fi

dino_run=$RUN_ROOT/latent_dinov2_vitl_independent224_gbs128_step21360_seed7
wait_for_healthy_idle_gpus
launch_and_evaluate dinov2_vitl "$dino_run" 29572 15610 \
  "$dino_acb" "$dino_imb" "$dino_vmb"

wait_for_healthy_idle_gpus
wan_acb=0
wan_imb=32
wan_vmb=32
wan_smoke=$RUN_ROOT/preflight_latent_wan22_vae38_4gpu_bs32_step10_acb0_vmb32_seed7
if ! run_smoke wan22_vae38 "$wan_smoke" 29573 \
  "$wan_acb" "$wan_imb" "$wan_vmb"; then
  if [[ -s "$wan_smoke/train.log" ]] && \
     rg -q 'CUDA out of memory|OutOfMemoryError' "$wan_smoke/train.log"; then
    wan_acb=1
    wan_vmb=16
    wan_smoke=$RUN_ROOT/preflight_latent_wan22_vae38_4gpu_bs32_step10_acb1_vmb16_seed7
    wait_for_healthy_idle_gpus
    run_smoke wan22_vae38 "$wan_smoke" 29576 \
      "$wan_acb" "$wan_imb" "$wan_vmb"
  else
    echo "Wan2.2 smoke failed for a non-OOM reason" >&2
    exit 1
  fi
fi

wan_run=$RUN_ROOT/latent_wan22_vae38_independent224_gbs128_step21360_seed7
wait_for_healthy_idle_gpus
launch_and_evaluate wan22_vae38 "$wan_run" 29574 15620 \
  "$wan_acb" "$wan_imb" "$wan_vmb"

echo "[$(date -Is)] latent DINOv2 and Wan2.2 sequence completed"
