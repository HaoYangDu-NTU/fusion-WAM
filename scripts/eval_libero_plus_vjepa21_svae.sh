#!/usr/bin/env bash
set -euo pipefail
cd /workspace/code/OpenWAM

# Same final S-VAE checkpoint already evaluated on LIBERO (1920/2000).
OPENWAM_PYTHON="${OPENWAM_PYTHON:-/workspace/basics/miniconda3/envs/openwam/bin/python}"
SOURCE_DIR="${SOURCE_DIR:-/workspace/code/OpenWAM/outputs/libero_vjepa21_cached_text_svae/2026-09-21_10-23-40}"
CKPT_NAME="${CKPT_NAME:-checkpoint_step_10690.safetensors}"
CORRECT_INSTRUCTION_SUFFIX="${CORRECT_INSTRUCTION_SUFFIX:-True}"
text_cache_args=()
if [[ -n "${TEXT_EMBEDDING_DIR:-}" ]]; then
  text_cache_args=(--text-embedding-dir "$TEXT_EMBEDDING_DIR")
fi
remaining=()
while (($#)); do
  case "$1" in
    --correct-instruction-suffix|--correct_instruction_suffix)
      CORRECT_INSTRUCTION_SUFFIX="${2:?Expected True or False}"; shift 2 ;;
    --correct-instruction-suffix=*|--correct_instruction_suffix=*)
      CORRECT_INSTRUCTION_SUFFIX="${1#*=}"; shift ;;
    *) remaining+=("$1"); shift ;;
  esac
done
case "${CORRECT_INSTRUCTION_SUFFIX,,}" in
  true) TEXT_MODE=corrected_text; CORRECT_INSTRUCTION_SUFFIX=True ;;
  false) TEXT_MODE=original_text; CORRECT_INSTRUCTION_SUFFIX=False ;;
  *) echo 'correct_instruction_suffix must be True or False' >&2; exit 2 ;;
esac
DEPLOY_DIR="${DEPLOY_DIR:-/workspace/code/OpenWAM/outputs/libero-plus/vjepa21_svae_step10690_harness_${TEXT_MODE}/checkpoint_for_eval}"


"$OPENWAM_PYTHON" -m benchmarks.vla_harness.prepare_libero_plus_svae \
  --source-dir "$SOURCE_DIR" --ckpt-name "$CKPT_NAME" \
  --output-dir "$DEPLOY_DIR" --correct-instruction-suffix "$CORRECT_INSTRUCTION_SUFFIX" \
      "${text_cache_args[@]}"

# --dry-run only prints a plan; actual evaluation requires idle GPUs and the
# exact Plus text cache. Text encoding is never started by this script.
exec "$OPENWAM_PYTHON" -m benchmarks.vla_harness.run_eval \
  --benchmark libero-plus --ckpt-dir "$DEPLOY_DIR" --ckpt-name "$CKPT_NAME" \
  --correct-instruction-suffix "$CORRECT_INSTRUCTION_SUFFIX" \
  --gpus 0,1,2,3 --workers 32 --max-batch-size 8 --max-wait-time 0.01 \
  --render gpu --render-gpus 3 --image-format raw "${remaining[@]}"
