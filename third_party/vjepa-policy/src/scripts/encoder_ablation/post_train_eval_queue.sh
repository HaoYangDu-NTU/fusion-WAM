#!/usr/bin/env bash
set -euo pipefail

# Detached post-training queue. It waits for the three exact checkpoints,
# validates their saved topology, then runs direct LIBERO (two four-GPU groups
# in parallel) and LIBERO-Plus (one complete eight-GPU EGL pass per policy).
# The script never substitutes a different checkpoint when a target is absent.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT="$ROOT/runs"
LOG_DIR="$RUN_ROOT/encoder_ablation_post_eval"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/supervisor.log") 2>&1

PYTHON=${PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
HARNESS_RUNNER="$ROOT/scripts/run_vjepa_libero_plus_full_canonical.sh"
BENCHMARK_CONFIG="$ROOT/scripts/encoder_ablation/libero_plus_all_canonical.yaml"
ASSETS=/data/zhangyang/workspace/libero-plus-assets.vCbeiy
PLUS_SERVER_PYTHON=${VJEPA_SERVER_PYTHON:-/data/zhangyang/miniconda3/envs/prts-ft/bin/python}
TRAIN_PID=${1:-}

variants=(vjepa2_vitg vjepa2_1_vitl vjepa2_1_vitg)
declare -A run_dirs=(
  [vjepa2_vitg]="$RUN_ROOT/full_encoder_ablation_vjepa2_vitg_independent224_t5len128_gbs128_step21360_seed7_8gpu"
  [vjepa2_1_vitl]="$RUN_ROOT/full_encoder_ablation_vjepa2_1_vitl_independent224_t5len128_gbs128_step21360_seed7_8gpu"
  [vjepa2_1_vitg]="$RUN_ROOT/full_encoder_ablation_vjepa2_1_vitg_independent224_t5len128_gbs128_step21360_seed7_8gpu"
)

checkpoint_for() { echo "${run_dirs[$1]}/checkpoint_step021360.pt"; }

validate_checkpoint() {
  local variant=$1 ckpt=$2
  "$PYTHON" - "$variant" "$ckpt" <<'PY'
import sys
import torch

variant, path = sys.argv[1:]
expected = {
    "vjepa2_vitg": ("vjepa2", "vit_giant_xformers", "target_encoder"),
    "vjepa2_1_vitl": ("vjepa2_1", "vit_large", "ema_encoder"),
    "vjepa2_1_vitg": ("vjepa2_1", "vit_giant_xformers", "target_encoder"),
}
ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
top = ckpt.get("model_topology") or {}
serving = (ckpt.get("policy_serving") or {}).get("config") or {}
family, model, key = expected[variant]
checks = {
    "step": ckpt.get("step") == 21360,
    "global_batch_size": ckpt.get("global_batch_size") == 128,
    "max_steps": ckpt.get("max_steps") == 21360,
    "t5_len": serving.get("t5_len") == 128,
    "family": top.get("encoder_family") == family,
    "model": top.get("encoder_model_name") == model,
    "checkpoint_key": top.get("encoder_checkpoint_key") == key,
    "context_tubelets": serving.get("context_tubelets") == 1,
    "num_frames": serving.get("num_frames") == 10,
    "action_chunk_size": top.get("action_chunk_size") == 32,
    "views": top.get("predictor_num_views") == 2,
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    raise SystemExit(f"{variant}: checkpoint metadata mismatch: {bad}")
print(f"validated {variant}: step=21360 global_batch=128 t5_len=128 encoder={family}/{model}/{key}")
PY
}

wait_for_checkpoint() {
  local variant=$1 ckpt
  ckpt=$(checkpoint_for "$variant")
  echo "[$(date -u +%FT%TZ)] waiting for $variant: $ckpt"
  while [[ ! -s "$ckpt" ]]; do
    # A completed supervisor with no target checkpoint is a hard failure, not
    # a reason to silently evaluate an older run.
    if grep -q 'encoder queue finished' "$RUN_ROOT/encoder_ablation_exact_queue/queue.log" 2>/dev/null; then
      echo "queue ended before $variant checkpoint appeared" >&2
      return 1
    fi
    if [[ -n "$TRAIN_PID" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "training PID $TRAIN_PID exited before $variant checkpoint appeared" >&2
      return 1
    fi
    sleep 60
  done
  validate_checkpoint "$variant" "$ckpt"
}

run_libero() {
  local variant=$1 gpu_list=$2 tag=$3
  local dir=${run_dirs[$variant]}
  local base_port=14100
  if [[ "$tag" == group4_7 ]]; then
    base_port=14200
  fi
  EVAL_GPUS="$gpu_list" EVAL_SIM_GPUS="$gpu_list" \
    EVAL_DIR="$dir/libero_eval_r16_50trials_seed7_egl_${tag}" \
    BASE_PORT="$base_port" \
    POLICY_PYTHON="$PLUS_SERVER_PYTHON" \
    bash "$ROOT/scripts/encoder_ablation/eval_exact_t5len128.sh" \
      "$variant" "$dir" "$(checkpoint_for "$variant")"
}

run_plus() {
  local variant=$1 index=$2
  local dir=${run_dirs[$variant]}
  local out="$dir/libero_plus_eval_canonical_8gpu_12clients"
  if [[ -s "$out/run_metadata.json" ]]; then
    echo "[$(date -u +%FT%TZ)] reusing completed Plus result for $variant: $out"
    return 0
  fi
  VLA_LIBERO_PLUS_ASSETS="$ASSETS" \
    VJEPA_SERVER_PYTHON="$PLUS_SERVER_PYTHON" \
    VJEPA_POLICY_NAME="encoder-ablation-$variant" \
    bash "$HARNESS_RUNNER" --mode fixed128 \
      --gpus 0,1,2,3,4,5,6,7 --sim-gpus 0,1,2,3,4,5,6,7 \
      --clients-per-gpu 12 --batch-size 12 \
      --base-port "$((32000 + index * 100))" \
      --eval-id "encoder-ablation-${variant}-canonical-8x12" \
      --benchmark-config "$BENCHMARK_CONFIG" \
      --server-config "$ROOT/scripts/encoder_ablation/plus_server_${variant}.yaml" \
      --output-dir "$out"
}

echo "[$(date -u +%FT%TZ)] post-training evaluation queue started"
for variant in "${variants[@]}"; do
  wait_for_checkpoint "$variant"
done

echo "[$(date -u +%FT%TZ)] starting LIBERO pair on GPU groups 0-3 and 4-7"
run_libero vjepa2_vitg 0,1,2,3 group0_3 >"$LOG_DIR/libero_vjepa2_vitg.log" 2>&1 & p0=$!
run_libero vjepa2_1_vitl 4,5,6,7 group4_7 >"$LOG_DIR/libero_vjepa2_1_vitl.log" 2>&1 & p1=$!
set +e
finished_pid=""
wait -n -p finished_pid "$p0" "$p1"; first_status=$?
set -e
if [[ "$finished_pid" == "$p0" ]]; then
  s0=$first_status
  if (( s0 != 0 )); then
    set +e; wait "$p1"; s1=$?; set -e
    echo "LIBERO pair failed: vjepa2_vitg=$s0 vjepa2_1_vitl=$s1" >&2
    exit 1
  fi
  echo "[$(date -u +%FT%TZ)] first LIBERO policy complete; starting third on freed GPU group 0-3"
  run_libero vjepa2_1_vitg 0,1,2,3 group0_3 >"$LOG_DIR/libero_vjepa2_1_vitg.log" 2>&1 & p2=$!
  set +e
  wait "$p1"; s1=$?
  wait "$p2"; s2=$?
  set -e
else
  s1=$first_status
  if (( s1 != 0 )); then
    set +e; wait "$p0"; s0=$?; set -e
    echo "LIBERO pair failed: vjepa2_vitg=$s0 vjepa2_1_vitl=$s1" >&2
    exit 1
  fi
  echo "[$(date -u +%FT%TZ)] second LIBERO policy complete; starting third on freed GPU group 4-7"
  run_libero vjepa2_1_vitg 4,5,6,7 group4_7 >"$LOG_DIR/libero_vjepa2_1_vitg.log" 2>&1 & p2=$!
  set +e
  wait "$p0"; s0=$?
  wait "$p2"; s2=$?
  set -e
fi
if (( s0 != 0 || s1 != 0 || s2 != 0 )); then
  echo "LIBERO evaluations failed: vjepa2_vitg=$s0 vjepa2_1_vitl=$s1 vjepa2_1_vitg=$s2" >&2
  exit 1
fi
echo "[$(date -u +%FT%TZ)] all LIBERO evaluations complete"

for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  echo "[$(date -u +%FT%TZ)] starting canonical LIBERO-Plus for $variant"
  run_plus "$variant" "$index"
  "$PYTHON" - "$variant" "${run_dirs[$variant]}/libero_plus_eval_canonical_8gpu_12clients/run_metadata.json" <<'PY'
import json
import sys

variant, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    meta = json.load(handle)
required = {"mode": "fixed128", "render_backend": "egl", "total_episodes": 10030}
for key, value in required.items():
    if meta.get(key) != value:
        raise SystemExit(f"{variant}: Plus metadata {key}={meta.get(key)!r}, expected {value!r}")
print(f"validated {variant} Plus: {meta['success']}/{meta['total_episodes']} ({meta['success_rate']:.2%}), EGL, fixed128")
PY
done
echo "[$(date -u +%FT%TZ)] post-training evaluation queue finished"
