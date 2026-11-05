#!/usr/bin/env bash
# Raw V-JEPA and S-VAE, old L-shape and independent-view checkpoints.
# Usage: bash scripts/eval_libero_vjepa.sh CKPT_DIR CKPT_NAME [run_eval options]
set -euo pipefail
if [[ "${1:-}" == --help || $# -lt 2 ]]; then
  echo 'Usage: bash scripts/eval_libero_vjepa.sh CKPT_DIR CKPT_NAME [--benchmark libero|libero-plus] [--smoke|--dry-run] [options]'
  [[ "${1:-}" == --help ]] && exit 0
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$(cd "$1" && pwd)"
CKPT_NAME="$2"
shift 2
cd "$REPO_ROOT"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-${REPO_ROOT}/../../basics/miniconda3/envs/openwam/bin/python}"
BENCHMARK=libero
CORRECT_INSTRUCTION_SUFFIX="${CORRECT_INSTRUCTION_SUFFIX:-}"
POLICY_BACKEND=openwam
text_cache_args=()
if [[ -n "${TEXT_EMBEDDING_DIR:-}" ]]; then
  text_cache_args=(--text-embedding-dir "$TEXT_EMBEDDING_DIR")
fi
remaining=()
while (($#)); do
  case "$1" in
    --policy-backend) POLICY_BACKEND="${2:?Expected openwam or fm-vjepa}"; shift 2 ;;
    --policy-backend=*) POLICY_BACKEND="${1#*=}"; shift ;;
    --benchmark) BENCHMARK="${2:?Expected libero or libero-plus}"; shift 2 ;;
    --benchmark=*) BENCHMARK="${1#*=}"; shift ;;
    --correct-instruction-suffix|--correct_instruction_suffix)
      CORRECT_INSTRUCTION_SUFFIX="${2:?Expected True or False}"; shift 2 ;;
    --correct-instruction-suffix=*|--correct_instruction_suffix=*)
      CORRECT_INSTRUCTION_SUFFIX="${1#*=}"; shift ;;
    *) remaining+=("$1"); shift ;;
  esac
done
if [[ -z "$CORRECT_INSTRUCTION_SUFFIX" ]]; then
  if [[ "$BENCHMARK" == libero-plus ]]; then
    CORRECT_INSTRUCTION_SUFFIX=True
  else
    CORRECT_INSTRUCTION_SUFFIX=False
  fi
fi
case "${CORRECT_INSTRUCTION_SUFFIX,,}" in
  true) TEXT_MODE=corrected_text; CORRECT_INSTRUCTION_SUFFIX=True ;;
  false) TEXT_MODE=original_text; CORRECT_INSTRUCTION_SUFFIX=False ;;
  *) echo 'correct_instruction_suffix must be True or False' >&2; exit 2 ;;
esac
if [[ "$POLICY_BACKEND" == fm-vjepa || "$POLICY_BACKEND" == vpp2 ]]; then
  exec "$OPENWAM_PYTHON" -m benchmarks.vla_harness.run_eval \
    --policy-backend "$POLICY_BACKEND" --benchmark "$BENCHMARK" --ckpt-dir "$SOURCE_DIR" --ckpt-name "$CKPT_NAME" \
    --correct-instruction-suffix "$CORRECT_INSTRUCTION_SUFFIX" "${remaining[@]}"
fi
[[ "$POLICY_BACKEND" == openwam ]] || { echo "Unknown policy backend: $POLICY_BACKEND" >&2; exit 2; }
DEPLOY_DIR="$SOURCE_DIR"
case "$BENCHMARK" in
  libero) ;;
  libero-plus)
    DEPLOY_DIR="${PLUS_DEPLOY_DIR:-${SOURCE_DIR}/eval_libero_plus/${CKPT_NAME%.safetensors}/${TEXT_MODE}}"

    "$OPENWAM_PYTHON" -m benchmarks.vla_harness.prepare_libero_plus \
      --source-dir "$SOURCE_DIR" --ckpt-name "$CKPT_NAME" \
      --output-dir "$DEPLOY_DIR" --correct-instruction-suffix "$CORRECT_INSTRUCTION_SUFFIX" \
      "${text_cache_args[@]}"
    ;;
  *) echo 'benchmark must be libero or libero-plus' >&2; exit 2 ;;
esac
exec "$OPENWAM_PYTHON" -m benchmarks.vla_harness.run_eval \
  --benchmark "$BENCHMARK" --ckpt-dir "$DEPLOY_DIR" --ckpt-name "$CKPT_NAME" \
  --correct-instruction-suffix "$CORRECT_INSTRUCTION_SUFFIX" "${remaining[@]}"
