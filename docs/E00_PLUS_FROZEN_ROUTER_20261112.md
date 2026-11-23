# E00+ Frozen M29 Dense Router Experiment

## Objective

Compare a matched Action-Decoder-only fine-tune with a dense Action Visual Router fine-tune. Both start from the original E00 checkpoint, freeze the complete E00 backbone, use the same LIBERO training data and seed, and receive 1,000 optimizer updates. The Router arm uses an FP32 residual gate and FP32 residual multiply/add; its Top-k sparse loss is disabled. After training, evaluate both models on the same 600-trial LIBERO-Plus pilot used for the earlier B0/B1/B2 comparison.

## Checkpoints and configuration identity

- Original E00 source: `/data/fcy/code/OpenWAM_v2/outputs/libero_vjepa_1_3b_dev_vx_ax_s42.0/2026-10-26_20-36-02/checkpoint_step_10690.safetensors`
- Source SHA-256: `9279797d40dfe433ecc4298fd70a0c93ea93ec780dd478359e36944c949f9440`
- The Dense Router tensors come from the seed-matched B1 initialization checkpoint at `outputs/e00_plus_finetune/initialization/B1/checkpoint_step_0.safetensors` (SHA-256 `5e4fb6b70b0a8cb938e74ab55b6dbc32a7db81035aaf64be6244884ab31eae5d`). Representative original-E00 tensors copied into this initialization match exactly: M29 block 0 attention weights, M29 block 29 attention weights, and Action Decoder weight/bias all have max absolute difference 0.
- Fresh, audit-preserving initialization directories: `outputs/e00_plus_frozen_router/initialization/decoder_only_e00_v2` and `outputs/e00_plus_frozen_router/initialization/fp32_dense_e00_v2`.
- The only source-config path adjustments are host-local V-JEPA and text-cache paths. Fusion layers, REPA depth/coefficient/target fusion method/layers match the original E00 config exactly. Training now opts into `training.model_config_from_finetune_checkpoint=true`, preventing current experiment-profile defaults from silently changing the source architecture.

## Matched training setup

Both arms use seed 42, 4 GPUs, per-GPU batch 64 (global batch 256), BF16 mixed precision, learning rate `1e-5`, and 1,000 updates. The Decoder-only arm trains `action_backbone.action_decoder.*`. The Dense Router arm trains that same decoder plus `action_router.*`. All other parameters, including the E00 M29 backbone, are excluded by the explicit trainable-parameter allowlist.

The Dense Router has 512 hidden dimensions, 8 heads, Top-32 diagnostic support, initial effective gate 0.01, FP32 gate/residual arithmetic, and sparse-loss weight 0.0.

## Smoke verification

The corrected `smoke_e00` run completed 4 steps for both arms. Dense Router had no BF16/FP32 crash after making the final action projection input/output dtype handling explicit.

At Dense Router smoke step 4, the logged values were: total loss 0.1256, action loss 0.00473, Top-32 mass 0.413, attention entropy 4.555 (near the uniform maximum `ln(98) ≈ 4.585`), effective tokens 92.6/98, Top-32 union coverage 99.98%, gate 0.0100134, and Router residual ratio 0.363%. Across four updates, gate changed slightly but nonzero. These are wiring/initialization diagnostics only; four updates do not establish control performance or meaningful attention sparsity.

An earlier formal attempt ran only two updates using live E00 profile overrides that changed Fusion/REPA configuration. It was stopped and excluded. Its logs and command are preserved under `reports/E00_plus/frozen_router_20261111/train/decoder_only/config_override_aborted/`.

## Evaluation protocol

Reuse the exact previous 600-trial LIBERO-Plus pilot manifest: 200 trials each for Camera Viewpoints, Light Conditions, and Background Textures, corrected instruction suffix enabled, same tasks/seeds/initial states, same worker/batch/render settings. The previous B0/B1/B2 pilot summary is at `reports/E00_plus/diagnosis_20261111/pilot_summary.json`. This experiment does not launch the full LIBERO-Plus benchmark.

The evaluation launcher resolves the unique timestamped `checkpoint_step_1000.safetensors` under each training output before starting, and writes a separate report directory for each arm.

## Current status (2026-11-12)

- Corrected smoke: both arms complete.
- Decoder-only 1,000-step training: complete in 22m43s; checkpoints at steps 500 and 1,000 are saved under `outputs/e00_plus_frozen_router/train/decoder_only/2026-11-12_10-27-38/`.
- Dense Router 1,000-step training: complete in 24m30s; checkpoints at steps 500 and 1,000 are saved under `outputs/e00_plus_frozen_router/train/fp32_dense/2026-11-12_10-51-57/`.
- At Router training step 1,000, gate was 0.01000415, residual ratio 0.362%, Top-32 mass 0.415, entropy 4.553, effective tokens 92.3/98, and Top-32 union coverage 99.96%. Attention stayed broadly distributed despite the added readout; this is a mechanism diagnostic, not a success-rate result.
- M29 block 0 and block 29 representative weights are bit-identical between each final checkpoint and its initialization (max difference 0), confirming the trunk remained frozen. The Action Decoder changed in both arms. Dense checkpoint stores Action Decoder and Router tensors as FP32; the preserved E00 M29 trunk remains BF16.
- LIBERO-Plus pilot is queued in `reports/E00_plus/frozen_router_20261111/evaluation_queue/`; the queue launches both 600-trial evaluations after all four GPUs remain below 3,000 MiB for three consecutive 60-second checks. Other GPU workloads currently occupy the cards.
- No success-rate conclusions are available yet.

## 2026-11-21 continuation

The November 12 queue exited after 14 LIBERO-Plus environment timeouts and zero valid trials. The November 21 run uses the same 600 selected variants, corrected instructions and step-1,000 checkpoints, with eight workers rendering across four GPUs. The three-stage controller adopts existing valid rows, runs the frozen FP32 checkpoint ON/OFF, then trains a matched scratch baseline and FP32 Dense Router. The historical baseline lacks a sampler identity and initialization digest, so both scratch arms use the same current training recipe and seed 42. The primary score uses the final checkpoint. A gain of at least 12 net successes in 600 trials triggers confirmation on a disjoint 600-variant selection and a second matched training seed (43). This threshold is a compute allocation decision.

[Live status](../reports/E00_plus/frozen_router_20261111/sequence_20261121/status.json) · [Stage controller](../scripts/e00_router_sequence.py) · [Report directory](../reports/E00_plus/frozen_router_20261111/sequence_20261121/)
