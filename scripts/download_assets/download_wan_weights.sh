#!/usr/bin/env bash
set -euo pipefail
cd /data/fcy/code/openwam-exp
HF=/data/fcy/basics/miniconda3/envs/openwam/bin/hf

echo "[$(date '+%F %T %Z')] start Wan2.2-TI2V-5B"
"$HF" download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /data/fcy/code/openwam-exp/encoder/wan22-ti2v-5b \
  --max-workers 8

echo "[$(date '+%F %T %Z')] start Wan2.1-VACE-1.3B"
"$HF" download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir /data/fcy/code/openwam-exp/encoder/wan21-vace-1.3b \
  --max-workers 8

echo "[$(date '+%F %T %Z')] start Wan2.1-I2V-14B-480P"
"$HF" download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir /data/fcy/code/openwam-exp/encoder/wan21-i2v-14b-480p \
  --max-workers 8

echo "[$(date '+%F %T %Z')] all Wan downloads complete"
