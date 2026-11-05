#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

ACTION_RUN_DIR=$(realpath -e "${1:?usage: queue_one_after_eval.sh ACTION_RUN_DIR ENCODER GPU_IDS RUN_DIR SMOKE_DIR}")
ENCODER=${2:?usage: queue_one_after_eval.sh ACTION_RUN_DIR ENCODER GPU_IDS RUN_DIR SMOKE_DIR}
GPU_IDS=${3:?usage: queue_one_after_eval.sh ACTION_RUN_DIR ENCODER GPU_IDS RUN_DIR SMOKE_DIR}
RUN_DIR=$(realpath -m "${4:?usage: queue_one_after_eval.sh ACTION_RUN_DIR ENCODER GPU_IDS RUN_DIR SMOKE_DIR}")
SMOKE_DIR=$(realpath -m "${5:?usage: queue_one_after_eval.sh ACTION_RUN_DIR ENCODER GPU_IDS RUN_DIR SMOKE_DIR}")
SOURCE_ROOT=$(realpath -e "${SOURCE_ROOT:-$ROOT}")
SMOKE_TRAIN_PORT=${SMOKE_TRAIN_PORT:-29571}
FORMAL_TRAIN_PORT=${FORMAL_TRAIN_PORT:-29572}
FORMAL_EVAL_PORT=${FORMAL_EVAL_PORT:-15610}
ACTION_EVAL_PID_FILE=$ACTION_RUN_DIR/eval_queue_launcher.pid
ACTION_SUMMARY=$ACTION_RUN_DIR/libero_eval_r16_50trials_seed7/summary.txt

if [[ ! -s "$SOURCE_ROOT/SHA256SUMS" ]]; then
  echo "Immutable SOURCE_ROOT manifest is missing: $SOURCE_ROOT/SHA256SUMS" >&2
  exit 1
fi
if ! (cd "$SOURCE_ROOT" && sha256sum --check --quiet SHA256SUMS); then
  echo "Immutable SOURCE_ROOT manifest verification failed: $SOURCE_ROOT" >&2
  exit 1
fi
source "$SOURCE_ROOT/scripts/ablation_runtime.sh"
export PYTHONDONTWRITEBYTECODE=1

NEXT_VARIABLES=(NEXT_ENCODER NEXT_RUN_DIR NEXT_SMOKE_DIR)
NEXT_VARIABLE_COUNT=0
for variable in "${NEXT_VARIABLES[@]}"; do
  if [[ -v "$variable" ]]; then
    NEXT_VARIABLE_COUNT=$((NEXT_VARIABLE_COUNT + 1))
  fi
done
if (( NEXT_VARIABLE_COUNT != 0 && NEXT_VARIABLE_COUNT != 3 )); then
  echo "NEXT_ENCODER, NEXT_RUN_DIR, and NEXT_SMOKE_DIR must be set together" >&2
  exit 64
fi
CHAIN_NEXT=false
if (( NEXT_VARIABLE_COUNT == 3 )); then
  if [[ -z "$NEXT_ENCODER" || -z "$NEXT_RUN_DIR" || -z "$NEXT_SMOKE_DIR" ]]; then
    echo "NEXT_ENCODER, NEXT_RUN_DIR, and NEXT_SMOKE_DIR must all be non-empty" >&2
    exit 64
  fi
  NEXT_RUN_DIR=$(realpath -m "$NEXT_RUN_DIR")
  NEXT_SMOKE_DIR=$(realpath -m "$NEXT_SMOKE_DIR")
  NEXT_QUEUE_SCRIPT=$SOURCE_ROOT/scripts/latent_space_ablation/queue_one_after_eval.sh
  if [[ ! -f "$NEXT_QUEUE_SCRIPT" ]]; then
    echo "Immutable SOURCE_ROOT is missing the chained queue script: $NEXT_QUEUE_SCRIPT" >&2
    exit 1
  fi
  CHAIN_NEXT=true
fi

case "$ENCODER" in
  dinov2_vitl|dinov3_vitl)
    ACB=7
    IMAGE_MICROBATCH_SIZE=320
    VIDEO_MICROBATCH_SIZE=32
    ;;
  wan22_vae38)
    ACB=0
    IMAGE_MICROBATCH_SIZE=32
    VIDEO_MICROBATCH_SIZE=32
    ;;
  *)
    echo "Unknown latent encoder: $ENCODER" >&2
    exit 64
    ;;
esac

IFS=',' read -r -a GPUS <<<"$GPU_IDS"
if (( ${#GPUS[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs: $GPU_IDS" >&2
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

for port in "$SMOKE_TRAIN_PORT" "$FORMAL_TRAIN_PORT"; do
  if [[ ! "$port" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
    echo "Invalid training port: $port" >&2
    exit 64
  fi
done
if [[ ! "$FORMAL_EVAL_PORT" =~ ^[1-9][0-9]*$ ]] || \
   (( FORMAL_EVAL_PORT > 65532 )); then
  echo "Invalid four-suite evaluation base port: $FORMAL_EVAL_PORT" >&2
  exit 64
fi
if [[ "$RUN_DIR" == "$SMOKE_DIR" ]]; then
  echo "RUN_DIR and SMOKE_DIR must be different: $RUN_DIR" >&2
  exit 64
fi
for spec in "formal:$RUN_DIR" "smoke:$SMOKE_DIR"; do
  label=${spec%%:*}
  directory=${spec#*:}
  if [[ -e "$directory" && ! -d "$directory" ]]; then
    echo "$label path exists and is not a directory: $directory" >&2
    exit 1
  fi
  if [[ -d "$directory" ]] && find "$directory" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to append to non-empty $label directory: $directory" >&2
    exit 1
  fi
done

verify_inputs() {
  verify_source_snapshot "$SOURCE_ROOT"
  bash "$SOURCE_ROOT/scripts/latent_space_ablation/run.sh" "$ENCODER" --check
}

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
    echo "[$(date -Is)] waiting for GPUs $GPU_IDS: ${busy[*]}"
    sleep 30
  done
}

verify_action_eval() {
  if [[ ! -s "$ACTION_SUMMARY" ]] || \
     ! grep -Fxq 'successful_suites: 4/4' "$ACTION_SUMMARY"; then
    echo "[BLOCKED] upstream LIBERO evaluation is not successful_suites 4/4" >&2
    echo "summary: $ACTION_SUMMARY" >&2
    if [[ -s "$ACTION_SUMMARY" ]]; then
      sed -n '1,260p' "$ACTION_SUMMARY" >&2
    else
      echo "summary is missing or empty" >&2
    fi
    if [[ -s "$ACTION_RUN_DIR/eval_launcher.log" ]]; then
      echo "last action evaluation launcher lines:" >&2
      tail -100 "$ACTION_RUN_DIR/eval_launcher.log" >&2
    fi
    return 1
  fi
}

verify_inputs
if [[ ! -s "$ACTION_EVAL_PID_FILE" ]]; then
  echo "Missing action evaluation PID file: $ACTION_EVAL_PID_FILE" >&2
  exit 1
fi
ACTION_EVAL_PID=$(<"$ACTION_EVAL_PID_FILE")
if [[ ! "$ACTION_EVAL_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid action evaluation PID: $ACTION_EVAL_PID" >&2
  exit 64
fi
ACTION_EVAL_START=$(process_start_time "$ACTION_EVAL_PID" || true)
if [[ -n "$ACTION_EVAL_START" ]]; then
  echo "[$(date -Is)] waiting for upstream evaluation pid=$ACTION_EVAL_PID start=$ACTION_EVAL_START"
  while process_matches_start_time "$ACTION_EVAL_PID" "$ACTION_EVAL_START"; do
    sleep 60
  done
else
  echo "[$(date -Is)] upstream evaluation PID $ACTION_EVAL_PID already exited; validating summary"
fi
verify_action_eval

# The queue may wait for hours. Revalidate immutable code and weights before GPU work.
verify_inputs
wait_for_idle_gpus

echo "[$(date -Is)] smoke encoder=$ENCODER gpus=$GPU_IDS acb=$ACB image_microbatch=$IMAGE_MICROBATCH_SIZE video_microbatch=$VIDEO_MICROBATCH_SIZE"
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  MAIN_PROCESS_PORT="$SMOKE_TRAIN_PORT" \
  OUTDIR="$SMOKE_DIR" \
  SMOKE_NUM_GPUS=4 \
  SMOKE_MAX_STEPS=10 \
  ACTIVATION_CHECKPOINTING_BLOCKS="$ACB" \
  IMAGE_MICROBATCH_SIZE="$IMAGE_MICROBATCH_SIZE" \
  VIDEO_MICROBATCH_SIZE="$VIDEO_MICROBATCH_SIZE" \
  bash "$SOURCE_ROOT/scripts/latent_space_ablation/run.sh" "$ENCODER" --smoke

SMOKE_CHECKPOINT=$SMOKE_DIR/checkpoint_step000010.pt
if [[ ! -s "$SMOKE_CHECKPOINT" ]] || \
   ! rg -q 'step=10/10 loss=[+-]?[0-9]' "$SMOKE_DIR/train.log" || \
   rg -qi '(loss|world|action|cos_sim|grad_norm)=([+-]?(nan|inf(inity)?))' "$SMOKE_DIR/train.log"; then
  echo "Smoke did not produce a finite step-10 checkpoint/log: $SMOKE_DIR" >&2
  exit 1
fi

verify_inputs
wait_for_idle_gpus
echo "[$(date -Is)] formal encoder=$ENCODER gpus=$GPU_IDS acb=$ACB image_microbatch=$IMAGE_MICROBATCH_SIZE video_microbatch=$VIDEO_MICROBATCH_SIZE"
MAIN_PROCESS_PORT="$FORMAL_TRAIN_PORT" \
  EVAL_BASE_PORT="$FORMAL_EVAL_PORT" \
  ACTIVATION_CHECKPOINTING_BLOCKS="$ACB" \
  IMAGE_MICROBATCH_SIZE="$IMAGE_MICROBATCH_SIZE" \
  VIDEO_MICROBATCH_SIZE="$VIDEO_MICROBATCH_SIZE" \
  bash "$SOURCE_ROOT/scripts/latent_space_ablation/launch_one.sh" \
    "$ENCODER" "$GPU_IDS" "$RUN_DIR"

if [[ ! -s "$RUN_DIR/train.pid" || ! -s "$RUN_DIR/eval_queue_launcher.pid" ]]; then
  echo "Formal launch did not record training and evaluation PIDs: $RUN_DIR" >&2
  exit 1
fi
echo "[$(date -Is)] formal training and automatic LIBERO evaluation queued: $RUN_DIR"

if [[ "$CHAIN_NEXT" == true ]]; then
  next_encoder=$NEXT_ENCODER
  next_run_dir=$NEXT_RUN_DIR
  next_smoke_dir=$NEXT_SMOKE_DIR
  unset NEXT_ENCODER NEXT_RUN_DIR NEXT_SMOKE_DIR
  echo "[$(date -Is)] chaining $next_encoder after automatic evaluation of $RUN_DIR"
  exec env \
    SOURCE_ROOT="$SOURCE_ROOT" \
    SMOKE_TRAIN_PORT="$SMOKE_TRAIN_PORT" \
    FORMAL_TRAIN_PORT="$FORMAL_TRAIN_PORT" \
    FORMAL_EVAL_PORT="$FORMAL_EVAL_PORT" \
    bash "$NEXT_QUEUE_SCRIPT" \
      "$RUN_DIR" "$next_encoder" "$GPU_IDS" "$next_run_dir" "$next_smoke_dir"
fi
