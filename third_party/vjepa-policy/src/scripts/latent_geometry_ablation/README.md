# Latent geometry ablations

These are isolated follow-up runs for the completed two-view V-JEPA2-L policy
ablation. The frozen semantic encoder is the only representation being
changed; the world-prediction and Action Expert losses remain enabled together.

`dinov2_native16` keeps the 224px DINOv2 ViT-L/14 crop and returns the native
`5 x 16 x 16` patch grid. It uses the same sampled policy clip
`[-4, 0, 4, ..., 32]` and endpoint images `[0, 8, 16, 24, 32]`. Its logical
Predictor geometry is `(T,H,W)=(5,16,16)`, with `t0 -> t1..t4` and 2,048
future tokens per two-view sample.

`wan22_stride2_t5` sets `past_frames=0` and `video_frame_stride=2`, so the VAE
receives exactly `[0, 2, 4, ..., 30, 32]` (17 RGB frames). Wan's temporal
compression ratio of four produces five native latent frames, mapped directly
to logical `t0..t4`; the world mask predicts `t1..t4` from `t0`. The logical
Predictor geometry remains `(5,14,14)`, with 1,568 future tokens per sample.

The pair launcher reserves GPUs `0-3` for DINOv2 native16 and `4-7` for Wan:

```bash
bash scripts/latent_geometry_ablation/launch_pair.sh
```

The geometry checkpoints require their dedicated serving adapter; the previous
latent-space adapter would restore DINOv2's interpolated 14x14 grid and Wan's
three-step temporal geometry. Queue full four-suite LIBERO evaluation after an
already-running training process with:

```bash
EVAL_GPUS=0,1,2,3 BASE_PORT=19100 \
  bash scripts/latent_geometry_ablation/wait_and_eval.sh \
    dinov2_native16 RUN_DIR TRAIN_PID
EVAL_GPUS=4,5,6,7 BASE_PORT=19200 \
  bash scripts/latent_geometry_ablation/wait_and_eval.sh \
    wan22_stride2_t5 RUN_DIR TRAIN_PID
```

Each waiter verifies the final finite training record, waits for its four GPUs
to become idle, and retries the resumable four-suite evaluation up to three
times. DINO and Wan evaluations can therefore start independently as soon as
their respective training group finishes.

Each launcher refuses to reuse a non-empty output directory. Use the individual
runner with `--check` for checkpoint validation or `--smoke` for a short run:

```bash
bash scripts/latent_geometry_ablation/run.sh dinov2_native16 --check
OUTDIR=/tmp/vjepa_native16_smoke bash scripts/latent_geometry_ablation/run.sh dinov2_native16 --smoke
```

Four-rank batch-32 preflights on the actual 80GB H100 groups use 70.0 GiB for
DINOv2 with 13 checkpointed Predictor blocks and image microbatch 64, and 70.5
GiB for Wan with 8 checkpointed blocks and video microbatch 32. These settings affect only
recomputation and frozen-encoder chunking, not the model, losses, or optimizer.
