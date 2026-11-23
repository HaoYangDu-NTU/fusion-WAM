#!/usr/bin/env bash
# Wait for a specific scratch training run, then smoke-test and run official Target50.
set -euo pipefail

infra_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval_root="$(cd "$infra_dir/../../.." && pwd)"
openwam_root="$(cd "${OPENWAM_ROOT:-$eval_root/../OpenWAM_v2}" && pwd)"
server_python="${OPENWAM_PYTHON:-/data/fcy/basics/miniconda3/envs/openwam/bin/python}"
eval_python="${ROBOCASA365_PYTHON:-$eval_root/.benchmark_runtime/robocasa365/robocasa/.venv/bin/python}"
check_only=false
if [[ "${1:-}" == --check ]]; then check_only=true; shift; fi
if [[ $# != 2 ]]; then
    echo "Usage: $0 [--check] TRAIN_CHECKPOINT_DIR EVAL_OUTPUT_DIR" >&2
    exit 2
fi
checkpoint_dir="$(realpath "$1")"
train_root="$(dirname "$checkpoint_dir")"
out="$(realpath -m "$2")"
test -x "$server_python" && test -x "$eval_python"
test -f "$checkpoint_dir/config.yaml" && test -f "$checkpoint_dir/normalization_stats.npy"
test -r "$train_root/pid"
training_pid="$(cat "$train_root/pid")"
run_spec="$("$server_python" - "$checkpoint_dir/config.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
if (
    cfg.training.finetune_ckpt_path is not None
    or not cfg.model.video_backbone.from_scratch
    or cfg.dataloader.type != 'robocasa365'
    or not cfg.model.video_backbone.encoder.seperate_encode_view
    or not cfg.training.max_steps
):
    raise SystemExit("Expected the RoboCasa365 M29 scratch training configuration")
print(int(cfg.training.max_steps), int(cfg.dataloader.num_frames) - 1,
      int(cfg.dataloader.get("independent_view_count", 2)))
PY
)"
read -r step action_horizon view_count <<<"$run_spec"
checkpoint="$checkpoint_dir/checkpoint_step_$step.safetensors"
if "$check_only"; then
    echo "Target50 waits for $train_root/exit_code = 0 and final artifact $checkpoint"
    exit 0
fi

mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9
if [[ -e "$out/target50/episodes.jsonl" || -e "$out/smoke/episodes.jsonl" ]]; then
    echo "Evaluation output already exists: $out" >&2
    exit 1
fi

echo "Waiting for training PID $training_pid and final step $step"
while [[ ! -s "$train_root/exit_code" ]]; do
    if ! kill -0 "$training_pid" 2>/dev/null; then
        echo "Training exited without a status at $train_root/exit_code" >&2
        exit 1
    fi
    sleep 15
done
if [[ "$(cat "$train_root/exit_code")" != 0 || ! -s "$checkpoint" ]] ||
   ! grep -Fq "Saved final: " "$train_root/train.log"; then
    echo "Training did not finish successfully with $checkpoint" >&2
    exit 1
fi
sha256sum "$checkpoint" >"$out/checkpoint.sha256"
cp "$checkpoint_dir/config.yaml" "$out/training_config.yaml"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export ROBOCASA365_PYTHON="$eval_python"
export ROBOCASA365_EVAL_VIEW_COUNT="$view_count"
export ROBOCASA365_POLICY_PORT="$("$server_python" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
(
    cd "$openwam_root"
    exec "$server_python" scripts/deploy.py --ckpt-dir "$checkpoint_dir" \
        --ckpt-name "$(basename "$checkpoint")" --host 127.0.0.1 \
        --port "$ROBOCASA365_POLICY_PORT" --device "${SERVER_DEVICE:-cuda:0}" \
        --inference-horizon "$action_horizon" \
        --compile-enabled false inference.encode_uncached_vlm_prompts=true \
        optimization.dit_cache.enabled=false
) >"$out/server.log" 2>&1 &
server_pid=$!
cleanup() {
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

deadline=$((SECONDS + 1200))
until grep -Fq "WebSocket server started on ws://127.0.0.1:$ROBOCASA365_POLICY_PORT" "$out/server.log"; do
    if ! kill -0 "$server_pid" 2>/dev/null || ((SECONDS >= deadline)); then
        echo "Policy server startup failed. See $out/server.log" >&2
        exit 1
    fi
    sleep 2
done

policy=benchmarks.robocasa365.infra.openwam_policy:make_policy
echo "Running checkpoint and simulator smoke test"
bash "$infra_dir/run.sh" eval --policy "$policy" --smoke --trials 1 --smoke-steps 4 \
    --output "$out/smoke" >"$out/smoke.log" 2>&1
echo "Starting official Target50 evaluation"
bash "$infra_dir/run.sh" eval --policy "$policy" --output "$out/target50" \
    >"$out/target50.log" 2>&1
echo "Target50 evaluation complete: $out/target50/summary.json"
