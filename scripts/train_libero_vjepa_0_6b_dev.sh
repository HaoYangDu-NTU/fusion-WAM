#!/usr/bin/env bash
# LIBERO + V-JEPA 2.1 + 0.6B policy (0.5B predictor + 0.1B action expert), with development switches.
# M25 training switches; distilled V-JEPA-L by default, frozen and excluded from 0.6B.
# Example: x/x prediction, no temporal/spatial merging, two-frame context:
#   VIDEO_PREDICTION_TYPE=x ACTION_PREDICTION_TYPE=x USE_AVG_T_POOLING=false \
#   USE_WAN_STYLE_PATCH_GRID=false HISTORY_NUM_TUBELETS=1 \
#   bash scripts/train_libero_vjepa_0_6b_dev.sh training.batch_size=8
# Extra Hydra arguments at the end override the settings below.
set -euo pipefail

TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-umt5-xxl}"
VLM_TEXT_EMBEDDING_METHOD="${VLM_TEXT_EMBEDDING_METHOD:-avg_all_layers}"
ulimit -n 65536
ulimit -c 0
cd /workspace/code/OpenWAM

# Independent video/action outputs: v = velocity, x = clean sample prediction.
# Both modes still use the existing velocity-space flow-matching loss.
VIDEO_PREDICTION_TYPE="${VIDEO_PREDICTION_TYPE:-x}"
ACTION_PREDICTION_TYPE="${ACTION_PREDICTION_TYPE:-x}"

# Frozen V-JEPA 2.1 encoder: G (1408 channels) or distilled L (1024 channels).
# L uses the EMA student weights; the manifest controls the encoder geometry.
VJEPA_ENCODER="${VJEPA_ENCODER:-L}"
case "$VJEPA_ENCODER" in
  G) VJEPA_ENCODER_MODEL_PATH=/workspace/model/vjepa ;;
  L) VJEPA_ENCODER_MODEL_PATH=/workspace/model/vjepa/vitl_dist_vitG_384 ;;
  *) echo "VJEPA_ENCODER must be L or G" >&2; exit 2 ;;
esac

# Encoder processing. These switches are independent.
ADD_INPUT_BOTTLENECK="${ADD_INPUT_BOTTLENECK:-false}" # split clean/future; bottleneck only the future
BOTTLENECK_DIM="${BOTTLENECK_DIM:-512}"
USE_AVG_T_POOLING="${USE_AVG_T_POOLING:-false}"          # extra temporal mean-pool
USE_WAN_STYLE_PATCH_GRID="${USE_WAN_STYLE_PATCH_GRID:-true}"  # true: (1,2,2); false: (1,1,1)
HISTORY_NUM_TUBELETS="${HISTORY_NUM_TUBELETS:-1}" # 0: current only; H>0: 2H frames at stride 4, episode-local
SEPERATE_ENCODE_VIEW="${SEPERATE_ENCODE_VIEW:-true}"   # true: two 224x224 views, local RoPE + view embedding
VJEPA21_FORWARD="${VJEPA21_FORWARD:-video}"            # video | mixed; history always uses video

# Optional S-VAE: null disables it; otherwise provide a checkpoint path.
# Its trained pooling mode must match USE_AVG_T_POOLING. Target dim null infers
# the width from the checkpoint; set 48 (for example) to check it explicitly.
SVAE_PATH="${SVAE_PATH:-null}"
SVAE_TARGET_DIM="${SVAE_TARGET_DIM:-null}"

# Flow schedules and joint attention.
NOISE_SCHEDULE_DISTRIBUTION="${NOISE_SCHEDULE_DISTRIBUTION:-logit_normal}" # uniform | logit_normal | logit_logistic | mode
SHIFT_VIDEO="${SHIFT_VIDEO:-42.0}"
SHIFT_ACTION="${SHIFT_ACTION:-5.0}"
ACTION_SCALE="${ACTION_SCALE:-1.0}" # positive scale for normalized actions/proprio; 1.0 preserves baseline
ATTENTION_MASK_MODE="${ATTENTION_MASK_MODE:-mutual}"   # mutual | action_sees_video | video_sees_action | isolated
VIDEO_ATTENTION_MASK_MODE="${VIDEO_ATTENTION_MASK_MODE:-clean_condition_causal}" # clean_condition_causal | per_frame_causal | bidirectional
USE_PROPRIOCEPTION="${USE_PROPRIOCEPTION:-true}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-/workspace/code/OpenWAM/outputs/libero_vjepa_0_6b_dev}"
# Short suffix: video prediction, action prediction, video shift (e.g. _vx_ax_s12.0).
# Hydra interpolation follows the final values, including trailing CLI overrides.
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH%/}"'_v${model.architecture.video_prediction_type}_a${model.architecture.action_prediction_type}_s${model.video_backbone.shift_video}'

for flag in ADD_INPUT_BOTTLENECK USE_AVG_T_POOLING USE_WAN_STYLE_PATCH_GRID SEPERATE_ENCODE_VIEW USE_PROPRIOCEPTION; do
  case "${!flag}" in
    true|false) ;;
    *) echo "$flag must be true or false" >&2; exit 2 ;;
  esac
done
for flag in VIDEO_PREDICTION_TYPE ACTION_PREDICTION_TYPE; do
  case "${!flag}" in
    v|x) ;;
    *) echo "$flag must be v or x" >&2; exit 2 ;;
  esac
done
case "$NOISE_SCHEDULE_DISTRIBUTION" in
  uniform|logit_normal|logit_logistic|mode) ;;
  *) echo "NOISE_SCHEDULE_DISTRIBUTION must be uniform, logit_normal, logit_logistic, or mode" >&2; exit 2 ;;
esac
if [[ ! "$HISTORY_NUM_TUBELETS" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "HISTORY_NUM_TUBELETS must be a non-negative integer" >&2
  exit 2
fi
case "$VJEPA21_FORWARD" in
  video|mixed) ;;
  *) echo "VJEPA21_FORWARD must be video or mixed" >&2; exit 2 ;;
esac

# Activate Conda fully, including environment variables and activation hooks.
source /workspace/basics/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/basics/miniconda3/envs/openwam
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
# All four H100 GPUs on this server attach to NUMA node 1.
exec taskset -c 48-95,144-191 /workspace/basics/miniconda3/envs/openwam/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/train.py \
  dataloader=libero \
  model/video_backbone=wan22_style_0_5b \
  model.action_backbone.dim=320 model.action_backbone.ffn_dim=1280 \
  model.architecture.variant=joint_self_attn \
  "model.architecture.video_prediction_type=$VIDEO_PREDICTION_TYPE" \
  "model.architecture.action_prediction_type=$ACTION_PREDICTION_TYPE" \
  "model.architecture.action_scale=$ACTION_SCALE" \
  "model.architecture.attention_mask_mode=$ATTENTION_MASK_MODE" \
  "model.architecture.video_attention_mask_mode=$VIDEO_ATTENTION_MASK_MODE" \
  "model.architecture.use_proprioception=$USE_PROPRIOCEPTION" \
  "model.action_backbone.shift_action=$SHIFT_ACTION" \
  dataloader.dataset_dir=/workspace/cache/huggingface_cache/hub/datasets--OpenWAM--LIBERO/snapshots/bcb2eaf1121ae4cbd324f8862807abce282500e8 \
  model.video_backbone.model_path=/workspace/model/wan2_2_openwam_native_from_diffusers \
  model.video_backbone.text_encoder_type="$TEXT_ENCODER_TYPE" \
  model.video_backbone.vlm_text_embedding_method="$VLM_TEXT_EMBEDDING_METHOD" \
  model.video_backbone.text_embedding_dir="${TEXT_EMBEDDING_DIR:-null}" \
  model/video_backbone/encoder=vjepa21 \
  "model.video_backbone.encoder.model_path=$VJEPA_ENCODER_MODEL_PATH" \
  "model.video_backbone.encoder.vjepa21_forward=$VJEPA21_FORWARD" \
  "model.video_backbone.encoder.add_input_bottleneck=$ADD_INPUT_BOTTLENECK" \
  "model.video_backbone.encoder.bottleneck_dim=$BOTTLENECK_DIM" \
  "model.video_backbone.encoder.use_avg_t_pooling=$USE_AVG_T_POOLING" \
  "model.video_backbone.encoder.use_wan_style_patch_grid=$USE_WAN_STYLE_PATCH_GRID" \
  "model.video_backbone.encoder.history_num_tubelets=$HISTORY_NUM_TUBELETS" \
  "model.video_backbone.encoder.seperate_encode_view=$SEPERATE_ENCODE_VIEW" \
  "model.video_backbone.encoder.svae_path=$SVAE_PATH" \
  "model.video_backbone.encoder.svae_target_dim=$SVAE_TARGET_DIM" \
  model.video_backbone.from_scratch=true \
  "model.video_backbone.shift_video=$SHIFT_VIDEO" \
  training.num_epochs=10 training.max_steps=null training.batch_size=64 training.zero_stage=1 \
  training.gradient_accumulation_steps=1 \
  "training.noise_schedule_distribution=$NOISE_SCHEDULE_DISTRIBUTION" \
  "training.output_path=$TRAIN_OUTPUT_PATH" \
  project.wandb.project=null \
  "$@"
