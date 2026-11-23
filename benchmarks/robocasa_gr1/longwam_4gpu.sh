#!/usr/bin/env bash
set -euo pipefail
runtime="$(cd "$(dirname "$0")/../../.benchmark_runtime/robocasa_gr1" && pwd)"
export ROBOCASA_GR1_TABLETOP_DATA_ROOT="$runtime/teleop/LeRobot"
export ROBOCASA_GR1_TABLETOP_ACTION_STATS="$runtime/global_action_minmax.json"
export LONGWAM_DATA_ROOT="$runtime/data"
export LONGWAM_OUTPUT_ROOT="$runtime/outputs"
export NPROC_PER_NODE=4 NNODES=1
exec bash "$runtime/LongLive/Long-WAM/scripts/robocasa_gr1/train.sh" \
    context=p384 batch_size=5 gradient_accumulation_steps=24 num_workers=8 "$@"
