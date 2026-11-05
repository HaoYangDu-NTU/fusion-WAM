#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

usage='usage: continue_full_ablation.sh SNAPSHOT CURRENT_EVAL_PID ACTION_V2 ACTION_V21 DINO2_SMOKE_PREFIX DINO2_RUN DINO3_RUN WAN_RUN TRAIN_CHAIN_PID'
SNAPSHOT=$(realpath -e "${1:?$usage}")
CURRENT_EVAL_PID=${2:?$usage}
ACTION_V2=$(realpath -e "${3:?$usage}")
ACTION_V21=$(realpath -e "${4:?$usage}")
DINO2_SMOKE_PREFIX=$(realpath -m "${5:?$usage}")
DINO2_RUN=$(realpath -m "${6:?$usage}")
DINO3_RUN=$(realpath -m "${7:?$usage}")
WAN_RUN=$(realpath -m "${8:?$usage}")
TRAIN_CHAIN_PID=${9:?$usage}

GPU_IDS=${GPU_IDS:-0,1,2,3}
IFS=',' read -r -a GPUS <<<"$GPU_IDS"
SUITES=(libero_spatial libero_object libero_goal libero_10)
PORT=${EVAL_PORT_START:-16010}

source "$SNAPSHOT/scripts/ablation_runtime.sh"
export PYTHONDONTWRITEBYTECODE=1

if (( ${#GPUS[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
  exit 64
fi
if (( $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) != 4 )); then
  echo "GPU_IDS must not contain duplicates: $GPU_IDS" >&2
  exit 64
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
    echo "Invalid GPU ID: $gpu" >&2
    exit 64
  fi
done
if [[ ! "$CURRENT_EVAL_PID" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$TRAIN_CHAIN_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Evaluation and training-chain PIDs must be positive integers" >&2
  exit 64
fi
if [[ "$DINO2_SMOKE_PREFIX" == "$DINO2_RUN" ]]; then
  echo "DINOv2 smoke prefix and formal run directory must differ" >&2
  exit 64
fi

(cd "$SNAPSHOT" && sha256sum --check --quiet SHA256SUMS)
bash "$SNAPSHOT/scripts/latent_space_ablation/run.sh" dinov2_vitl --check

check_gpu_health() {
  local gpu
  local recovery
  local uncorrected

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

wait_for_idle_gpus() {
  local busy
  local gpu
  local used

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
    printf '[%s] waiting for GPUs %s: %s\n' \
      "$(date -Is)" "$GPU_IDS" "${busy[*]}"
    sleep 30
  done
}

run_group() {
  local kind=$1
  local encoder=$2
  local run_dir=$3
  local eval_dir=$run_dir/libero_eval_osmesa_r16_50trials_seed7
  local eval_gpus=0,1,2,3
  local script
  local suites_csv
  local attempt
  local base_port

  if [[ "$kind" == action ]]; then
    script=$SNAPSHOT/scripts/encoder_ablation/eval.sh
  else
    script=$SNAPSHOT/scripts/latent_space_ablation/eval.sh
  fi
  suites_csv=$(IFS=,; printf '%s' "${SUITES[*]}")

  for attempt in 1 2 3; do
    base_port=$PORT
    PORT=$((PORT + 4))
    printf '[%s] kind=%s encoder=%s suites=%s gpus=%s ports=%s-%s attempt=%s\n' \
      "$(date -Is)" "$kind" "$encoder" "$suites_csv" "$eval_gpus" \
      "$base_port" "$((base_port + 3))" "$attempt"
    if env \
      LD_LIBRARY_PATH= \
      LIBERO_TORCH_LIB=/usr/lib/x86_64-linux-gnu \
      LIBERO_MUJOCO_GL=osmesa \
      PYOPENGL_PLATFORM=osmesa \
      LP_NUM_THREADS=8 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      NUMBA_NUM_THREADS=1 \
      EVAL_SUITES="$suites_csv" \
      EVAL_GPUS="$eval_gpus" \
      BASE_PORT="$base_port" \
      EVAL_DIR="$eval_dir" \
      bash "$script" "$encoder" "$run_dir" \
        "$run_dir/checkpoint_step021360.pt" && \
      grep -Fxq 'successful_suites: 4/4' "$eval_dir/summary.txt"; then
      return 0
    fi
    if (( attempt < 3 )); then
      sleep 15
    fi
  done
  echo "Group failed after three attempts: $kind $encoder" >&2
  return 1
}

verify_finite_training() {
  local run_dir=$1
  local final_step=$2
  local checkpoint=$run_dir/checkpoint_step$(printf '%06d' "$final_step").pt

  test -s "$checkpoint"
  rg -q "step=$final_step/$final_step loss=[+-]?[0-9]" "$run_dir/train.log"
  if rg -qi \
    '(loss|world|action|cos_sim|grad_norm)=([+-]?(nan|inf(inity)?))' \
    "$run_dir/train.log"; then
    echo "Training produced a non-finite metric: $run_dir" >&2
    return 1
  fi
}

wait_for_existing_run() {
  local label=$1
  local run_dir=$2
  local final_checkpoint=$run_dir/checkpoint_step021360.pt
  local train_pid
  local train_start
  local chain_start

  chain_start=$(process_start_time "$TRAIN_CHAIN_PID" || true)
  while [[ ! -s "$run_dir/train.pid" || ! -s "$run_dir/train.start" ]]; do
    if [[ -s "$final_checkpoint" ]]; then
      break
    fi
    if [[ -z "$chain_start" ]] || \
       ! process_matches_start_time "$TRAIN_CHAIN_PID" "$chain_start"; then
      echo "$label was not launched before training chain $TRAIN_CHAIN_PID exited" >&2
      return 1
    fi
    sleep 30
  done

  if [[ -s "$run_dir/train.pid" && -s "$run_dir/train.start" ]]; then
    train_pid=$(<"$run_dir/train.pid")
    train_start=$(<"$run_dir/train.start")
    while process_matches_start_time "$train_pid" "$train_start"; do
      sleep 60
    done
  fi
  verify_finite_training "$run_dir" 21360
}

run_dinov2_smoke() {
  local acb=$1
  local image_microbatch=$2
  local run_dir=$3
  local train_port=$4

  if [[ -e "$run_dir" && ! -d "$run_dir" ]]; then
    echo "Smoke path exists and is not a directory: $run_dir" >&2
    return 1
  fi
  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty smoke directory: $run_dir" >&2
    return 1
  fi
  mkdir -p "$run_dir"
  printf '[%s] DINOv2 smoke acb=%s image_microbatch=%s\n' \
    "$(date -Is)" "$acb" "$image_microbatch"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    MAIN_PROCESS_PORT="$train_port" \
    OUTDIR="$run_dir" \
    SMOKE_NUM_GPUS=4 \
    SMOKE_MAX_STEPS=10 \
    ACTIVATION_CHECKPOINTING_BLOCKS="$acb" \
    IMAGE_MICROBATCH_SIZE="$image_microbatch" \
    VIDEO_MICROBATCH_SIZE=32 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$SNAPSHOT/scripts/latent_space_ablation/run.sh" dinov2_vitl --smoke \
    >"$run_dir/smoke_launcher.log" 2>&1 || return $?
  verify_finite_training "$run_dir" 10
}

launch_dinov2() {
  local acb=$1
  local image_microbatch=$2
  local train_pid
  local train_start
  local train_status

  if [[ -e "$DINO2_RUN" && ! -d "$DINO2_RUN" ]]; then
    echo "Formal path exists and is not a directory: $DINO2_RUN" >&2
    return 1
  fi
  if [[ -d "$DINO2_RUN" ]] && \
     find "$DINO2_RUN" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty formal directory: $DINO2_RUN" >&2
    return 1
  fi
  mkdir -p "$DINO2_RUN"
  printf 'acb=%s\nimage_microbatch=%s\n' "$acb" "$image_microbatch" \
    >"$DINO2_RUN/runtime_selection.txt"

  (cd "$SNAPSHOT" && sha256sum --check --quiet SHA256SUMS)
  wait_for_idle_gpus
  printf '[%s] starting formal DINOv2 training acb=%s image_microbatch=%s\n' \
    "$(date -Is)" "$acb" "$image_microbatch"
  setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    MAIN_PROCESS_PORT=29582 \
    OUTDIR="$DINO2_RUN" \
    ACTIVATION_CHECKPOINTING_BLOCKS="$acb" \
    IMAGE_MICROBATCH_SIZE="$image_microbatch" \
    VIDEO_MICROBATCH_SIZE=32 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$SNAPSHOT/scripts/latent_space_ablation/run.sh" dinov2_vitl \
    >"$DINO2_RUN/launcher.log" 2>&1 < /dev/null &
  train_pid=$!
  printf '%s\n' "$train_pid" >"$DINO2_RUN/train.pid"
  train_start=$(process_start_time "$train_pid")
  printf '%s\n' "$train_start" >"$DINO2_RUN/train.start"

  setsid bash "$SNAPSHOT/scripts/watch_training_memory.sh" \
    "$train_pid" "$DINO2_RUN" 760 \
    >"$DINO2_RUN/watchdog_launcher.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$DINO2_RUN/watchdog.pid"

  set +e
  wait "$train_pid"
  train_status=$?
  set -e
  printf '[%s] DINOv2 training exited status=%s\n' \
    "$(date -Is)" "$train_status"
  (( train_status == 0 ))
  verify_finite_training "$DINO2_RUN" 21360
}

current_eval_start=$(process_start_time "$CURRENT_EVAL_PID" || true)
if [[ -n "$current_eval_start" ]]; then
  printf '[%s] waiting for current action evaluation pid=%s\n' \
    "$(date -Is)" "$CURRENT_EVAL_PID"
  while process_matches_start_time "$CURRENT_EVAL_PID" "$current_eval_start"; do
    sleep 30
  done
fi

wait_for_idle_gpus
run_group action vjepa2_vitl "$ACTION_V2"
run_group action vjepa2_1_vitl "$ACTION_V21"

selected_acb=
selected_image_microbatch=
for smoke_spec in 7:320:29581 8:320:29583 8:160:29584; do
  IFS=: read -r smoke_acb smoke_image_microbatch smoke_port <<<"$smoke_spec"
  smoke_dir=${DINO2_SMOKE_PREFIX}_acb${smoke_acb}_imb${smoke_image_microbatch}
  wait_for_idle_gpus
  if run_dinov2_smoke \
    "$smoke_acb" "$smoke_image_microbatch" "$smoke_dir" "$smoke_port"; then
    selected_acb=$smoke_acb
    selected_image_microbatch=$smoke_image_microbatch
    break
  fi
  if ! rg -q 'CUDA out of memory|OutOfMemoryError' \
    "$smoke_dir/smoke_launcher.log" "$smoke_dir/train.log"; then
    echo "DINOv2 smoke failed for a non-OOM reason: $smoke_dir" >&2
    exit 1
  fi
  printf '[%s] DINOv2 smoke OOM; retrying with more recomputation/chunking\n' \
    "$(date -Is)"
done
if [[ -z "$selected_acb" || -z "$selected_image_microbatch" ]]; then
  echo "DINOv2 smoke exhausted all memory configurations" >&2
  exit 1
fi

launch_dinov2 "$selected_acb" "$selected_image_microbatch"

wait_for_existing_run DINOv3 "$DINO3_RUN"
run_group latent dinov3_vitl "$DINO3_RUN"
run_group latent dinov2_vitl "$DINO2_RUN"
wait_for_existing_run Wan2.2 "$WAN_RUN"
run_group latent wan22_vae38 "$WAN_RUN"

printf '[%s] action-only, DINOv2, DINOv3, and Wan2.2 continuation complete\n' \
  "$(date -Is)"
