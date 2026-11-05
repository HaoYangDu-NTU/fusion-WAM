#!/usr/bin/env bash
# Official VPP2 policy on all 10,030 Plus tasks, with durable per-task resume.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-/data/fcy/basics/miniconda3/envs/openwam/bin/python}"
export VPP2_TORCH_LOAD_MMAP=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$OPENWAM_PYTHON" -u -m benchmarks.vla_harness.run_resumable \
  --policy-backend vpp2 --benchmarks libero-plus \
  --ckpt-dir weights/vpp2 --ckpt-name libero/action_step030000.pt \
  --gpus 0,1,2,3 --workers 128 --max-batch-size 16 --render-gpus 0,1,2,3 \
  --base-port 9240 --dynamic-tasks --correct-instruction-suffix True \
  --output-dir outputs/vpp2_plus/full_w128_b16 "$@"
