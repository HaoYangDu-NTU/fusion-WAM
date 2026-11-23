#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
CODE="$PROJECT/.benchmark_runtime/molmospaces"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CODE/.molmo-tools/interpreters}"
UV="${MOLMO_UV:-$CODE/.molmo-tools/bin/uv}"
SIM="$CODE/molmospaces-official"
OPENPI="$CODE/openpi-molmo-jointpos"
ARTIFACTS="$CODE/molmospaces-artifacts"
INDEX="${MOLMO_PIP_INDEX:-https://pypi.org/simple}"
WHEEL=mujoco-3.7.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
mkdir -p "$ARTIFACTS/wheels"
if [[ ! -x "$UV" ]]; then
    python -m pip install --target "$CODE/.molmo-tools" uv
fi
if [[ ! -d "$SIM/.git" ]]; then
    git clone --filter=blob:none --sparse --depth 1 --branch v0.2.9 https://github.com/allenai/molmospaces.git "$SIM"
    git -C "$SIM" sparse-checkout set molmo_spaces scripts
fi
if [[ ! -d "$OPENPI/.git" ]]; then
    git clone https://github.com/omarrayyann/openpi.git "$OPENPI"
    git -C "$OPENPI" checkout --detach 711487f019e5f03b254d427d4523b1f0805a4814
fi
[[ "$(git -C "$SIM" rev-parse HEAD)" == 735138876887eb64650742631b8dda841dd264bf ]]
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == 711487f019e5f03b254d427d4523b1f0805a4814 ]]
wheel_path="$ARTIFACTS/wheels/$WHEEL"
if [[ ! -f "$wheel_path" ]]; then
    curl -fsSL --retry 5 -C - \
        "https://raw.githubusercontent.com/allenai/molmospaces/735138876887eb64650742631b8dda841dd264bf/bin/wheels/$WHEEL" \
        -o "$wheel_path.part"
    mv "$wheel_path.part" "$wheel_path"
fi
[[ "$(git hash-object "$wheel_path")" == b5a66e66b530c301cbca69f083b2bd8af15f194a ]]
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$ARTIFACTS/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$ARTIFACTS/resource-cache}"
export MLSPACES_FORCE_INSTALL=False
for renderer in classic filament; do
    env_dir="$CODE/molmospaces/.venv-$renderer"
    if [[ ! -x "$env_dir/bin/python" ]]; then
        "$UV" venv --managed-python --python 3.11 --seed "$env_dir"
    fi
    if [[ "$renderer" == classic ]]; then
        simulator=("molmo-spaces[mujoco]==0.2.9" "mujoco==3.5.0" "mujoco-warp==3.5.0")
    else
        simulator=("molmo-spaces==0.2.9" "$wheel_path" "mujoco-warp==3.7.0")
    fi
    "$UV" pip install --index-url "$INDEX" --python "$env_dir/bin/python" pip \
        "${simulator[@]}" -e "$OPENPI/packages/openpi-client" "websockets>=13"
    if ! check_output=$("$env_dir/bin/python" -m pip check 2>&1); then
        if [[ "$check_output" != "decord 0.6.0 is not supported on this platform" ]]; then
            printf '%s\n' "$check_output" >&2
            exit 1
        fi
        # The upstream wheel carries a cp36 tag internally but its Python wrapper uses ctypes.
        "$env_dir/bin/python" -c 'import decord; assert decord.__version__ == "0.6.0"'
        printf '%s (import verified)\n' "$check_output"
    fi
    "$UV" pip freeze --python "$env_dir/bin/python" > "$ARTIFACTS/$renderer-packages.txt"
done
if [[ "${1:-}" != "--eval-only" ]]; then
    if [[ ! -x "$ARTIFACTS/.venv-droid/bin/python" ]]; then
        "$UV" venv --managed-python --python 3.11 --seed "$ARTIFACTS/.venv-droid"
    fi
    "$UV" pip install --index-url "$INDEX" --python "$ARTIFACTS/.venv-droid/bin/python" \
        tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9 tensorflow-metadata==1.15.0
    "$UV" pip freeze --python "$ARTIFACTS/.venv-droid/bin/python" > "$ARTIFACTS/droid-packages.txt"
fi
if [[ "${MOLMO_INSTALL_ASSETS:-1}" == 1 ]]; then
    "$CODE/molmospaces/.venv-classic/bin/python" -m molmo_spaces.molmo_spaces_constants
fi
