#!/usr/bin/env bash
set -euo pipefail
INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${INFRA_DIR}/../../.." && pwd)"
CODE_DIR="${PROJECT_DIR}/.benchmark_runtime/robocasa365"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "${PROJECT_DIR}"
exec "${ROBOCASA365_PYTHON:-${CODE_DIR}/robocasa/.venv/bin/python}" -m benchmarks.robocasa365.infra "$@"
