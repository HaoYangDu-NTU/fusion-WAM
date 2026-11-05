#!/usr/bin/env bash
set -euo pipefail

TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-umt5-xxl}"
VLM_TEXT_EMBEDDING_METHOD="${VLM_TEXT_EMBEDDING_METHOD:-avg_all_layers}"
ulimit -n 65536
ulimit -c 0
cd /workspace/code/OpenWAM
USE_AVG_T_POOLING="${USE_AVG_T_POOLING:-true}"
case "$USE_AVG_T_POOLING" in
  true|false) ;;
  *) echo "USE_AVG_T_POOLING must be true or false" >&2; exit 2 ;;
esac
if [[ "$USE_AVG_T_POOLING" == "true" ]]; then
  DEFAULT_SVAE_PATH="/workspace/code/OpenWAM/outputs/libero_vjepa21_cached_text_svae/pretrain_50epochs_20260919_111657/reducer/svae_20260919_112006/svae.pt"
  DEFAULT_OUTPUT_PATH="/workspace/code/OpenWAM/outputs/libero_vjepa21_cached_text_wan22_style_1_3b_svae"
else
  DEFAULT_SVAE_PATH="/workspace/code/OpenWAM/outputs/svae_pooling_false_restart_20260927_194017/reducer/svae_20260927_194136/svae.pt"
  DEFAULT_OUTPUT_PATH="/workspace/code/OpenWAM/outputs/libero_vjepa21_cached_text_wan22_style_1_3b_svae_no_pooling"
fi
SVAE_PATH="${SVAE_PATH:-$DEFAULT_SVAE_PATH}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}"
[[ -f "$SVAE_PATH" ]]
export PATH="/workspace/basics/miniconda3/envs/openwam/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
# All four H100 GPUs on this server attach to NUMA node 1.
exec taskset -c 48-95,144-191 /workspace/basics/miniconda3/envs/openwam/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/train.py \
  dataloader=libero \
  model/video_backbone=wan22_style_1_3b \
  dataloader.dataset_dir=/workspace/cache/huggingface_cache/hub/datasets--OpenWAM--LIBERO/snapshots/bcb2eaf1121ae4cbd324f8862807abce282500e8 \
  model.video_backbone.model_path=/workspace/model/wan2_2_openwam_native_from_diffusers \
  model.video_backbone.text_encoder_type="$TEXT_ENCODER_TYPE" \
  model.video_backbone.vlm_text_embedding_method="$VLM_TEXT_EMBEDDING_METHOD" \
  model.video_backbone.text_embedding_dir="${TEXT_EMBEDDING_DIR:-null}" \
  model/video_backbone/encoder=vjepa21 \
  model.video_backbone.encoder.model_path=/workspace/model/vjepa \
  "model.video_backbone.encoder.use_avg_t_pooling=$USE_AVG_T_POOLING" \
  model.video_backbone.from_scratch=true \
  model.video_backbone.shift_video=12.0 \
  "model.video_backbone.encoder.svae_path=$SVAE_PATH" \
  model.video_backbone.encoder.svae_target_dim=48 \
  training.num_epochs=10 training.batch_size=64 training.zero_stage=2 \
  "training.output_path=$TRAIN_OUTPUT_PATH" \
  project.wandb.project=null \
  "$@"
