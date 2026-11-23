#!/usr/bin/env bash
set -euo pipefail
scripts="$(cd "$(dirname "$0")" && pwd)"
runtime="$scripts/../../.benchmark_runtime/robocasa_gr1"
mkdir -p "$runtime/reference"
cd "$runtime"
python_bin="${PYTHON:-python}"
revision=09c6de8af50168090e7e9cc01e1ec3bce788de24
upstream=https://raw.githubusercontent.com/NVlabs/LongLive/31c70d9071cd7851d82939806dc9fac645fcabe5/Long-WAM
curl -fsSL --retry 3 "$upstream/configs/data/gr1_teleop_inventory.json" -o reference/inventory.json
curl -fsSL --retry 3 "$upstream/tools/data/build_robocasa_gr1_tabletop_stats.py" -o reference/build_stats.py
for attempt in 1 2 3; do
    if hf download nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim --repo-type dataset \
        --revision "$revision" --include 'LeRobot/gr1_unified.*/*' --max-workers 12 --local-dir teleop; then
        break
    fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
    sleep 30
done
"$python_bin" "$scripts/verify_teleop.py" teleop/LeRobot reference/inventory.json
"$python_bin" reference/build_stats.py --inventory reference/inventory.json --data-root teleop/LeRobot --output global_action_minmax.json
