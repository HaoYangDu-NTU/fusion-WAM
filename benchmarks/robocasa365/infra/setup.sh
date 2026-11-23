#!/usr/bin/env bash
# Isolated simulator + CPU-only training-interface smoke environment; no model training.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${HERE}/../../.." && pwd)/.benchmark_runtime/robocasa365"
mkdir -p "${CODE_DIR}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
PREFIX="${CODE_DIR}/robocasa/.venv"
checkout() {
    local name="$1" remote="$2" commit="$3"
    if [[ ! -d "${CODE_DIR}/${name}/.git" ]]; then
        git clone "$remote" "${CODE_DIR}/${name}"
        git -C "${CODE_DIR}/${name}" checkout --detach "$commit"
    fi
    [[ "$(git -C "${CODE_DIR}/${name}" rev-parse HEAD)" == "$commit" ]] || { echo "Revision mismatch: $name" >&2; exit 1; }
}
checkout robocasa https://github.com/robocasa/robocasa.git a07e365c958c4216cd6bbd5f30b47f09a65c6f00
checkout robosuite https://github.com/ARISE-Initiative/robosuite.git 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
if [[ ! -x "${PREFIX}/bin/python" ]]; then
    if [[ -f "${HERE}/conda-explicit-linux-64.txt" ]]; then
        "$CONDA_BIN" create -y -p "$PREFIX" --file "${HERE}/conda-explicit-linux-64.txt"
    else
        "$CONDA_BIN" create -y -p "$PREFIX" python=3.11 pip --override-channels -c conda-forge
    fi
fi
PY="${PREFIX}/bin/python"
if [[ -f "${HERE}/requirements.lock.txt" ]]; then
    "$PY" -m pip install -r "${HERE}/requirements.lock.txt" --extra-index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install --no-deps -e "${CODE_DIR}/robosuite" -e "${CODE_DIR}/robocasa"
else
    "$PY" -m pip install 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install -e "${CODE_DIR}/robosuite" -e "${CODE_DIR}/robocasa" pytest websockets==15.0.1
fi
# Explicit source roots also avoid namespace collisions when launched from CODE_DIR.
"$PY" - "$CODE_DIR" <<'PYSOURCES'
import sys,sysconfig
from pathlib import Path
code=Path(sys.argv[1])
(Path(sysconfig.get_paths()['purelib'])/'robocasa365_sources.pth').write_text(
    str(code/'robocasa')+'\n'+str(code/'robosuite')+'\n')
PYSOURCES
if [[ ! -f "${CODE_DIR}/robocasa/robocasa/macros_private.py" ]]; then
    "$PY" -m robocasa.scripts.setup_macros
fi
if [[ ! -f "${CODE_DIR}/robosuite/robosuite/macros_private.py" ]]; then
    "$PY" -m robosuite.scripts.setup_macros
fi
if [[ "${DOWNLOAD_ASSETS:-0}" == 1 ]]; then
    "$PY" -m robocasa.scripts.download_kitchen_assets
fi
"$PY" -m pip check
bash "${HERE}/run.sh" env
