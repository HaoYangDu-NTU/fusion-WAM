# Latent-space ablations

These scripts replace only the frozen latent encoder in the two-view 224x224
V-JEPA2-L policy baseline. They hard-lock global batch 128, 21,360 optimizer
steps, seed 7, Predictor 24x1024x16, Action Expert 24x512, and corrected
endpoint-aligned Predictor RoPE.

- `dinov2_vitl`: five frames at physical offsets `[0,8,16,24,32]`; native
  16x16 DINOv2 patches are endpoint-aligned interpolated to 14x14.
- `dinov3_vitl`: the same five physical frames and native 14x14 patches.
- `wan22_vae38`: physical offsets `[0,16,32]` are placed at logical Predictor
  times `[0,2,4]`; context is `t0`, and future loss uses only `t2,t4`.
- `internvideo3_8b`: the frozen InternVideo3-8B visual tower at pinned Hub
  revision `c4602918`; final normalized pre-merger tokens preserve the native
  `5x14x14x1152` grid. This run uses the untruncated T5 length 128 cache.

`launch_pair.sh` refuses to start unless both exact local checkpoints exist and
all eight GPUs are healthy and free. DINOv3 remains unavailable until the
official gated checkpoint has been provisioned locally. On a degraded machine,
`launch_one.sh` runs one experiment on an explicitly selected healthy four-GPU
group.

The full-BS single-rank memory preflight uses ACB 7 and an image microbatch of
320 for DINOv2 (77.8 GiB allocated), and ACB 0 with a video microbatch of 32 for
Wan2.2 (64.4 GiB allocated). These settings change only recomputation and frozen
encoder chunking; model initialization, data, loss, and optimizer are unchanged.

```bash
bash scripts/latent_space_ablation/launch_pair.sh dinov2_vitl wan22_vae38

bash scripts/latent_space_ablation/launch_one.sh \
  dinov2_vitl 0,1,2,3
```
