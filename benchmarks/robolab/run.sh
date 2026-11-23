#!/usr/bin/env bash
# Run a Python command against the project's prepared RoboLab checkout.
set -eu
project=$(cd "$(dirname "$0")/../.." && pwd)
rt="$project/.benchmark_runtime/robolab"
python_bin="${ROBOLAB_PYTHON:-$rt/sim-env/bin/python}"
if [[ "$python_bin" == "$rt/sim-env/bin/python" ]]; then
    # NVIDIA's Isaac Lab wheel ships the extension packages under source/.
    src="$rt/sim-env/lib/python3.11/site-packages/isaaclab/source"
    for component in "$src"/isaaclab*; do
        export PYTHONPATH="$component${PYTHONPATH:+:$PYTHONPATH}"
    done
    export LD_LIBRARY_PATH="$rt/tools/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$rt/RoboLab${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
cd "$rt/RoboLab"
exec "$python_bin" "$@"
