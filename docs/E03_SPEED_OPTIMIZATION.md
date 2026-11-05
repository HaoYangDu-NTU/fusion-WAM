# E03 Cache / Patch implementation and verification

2026-10-25. Implementation and CPU smoke/interface checks complete (41 passed).
The user subsequently authorized stopping/discarding the old run and starting fresh Patch training
after cache/smoke validation. Original E03 stopped gracefully at step1568 and is marked ABANDONED;
its checkpoint files are retained but will not initialize the new run.
The persistent pipeline is running under tmux `openwam-exp-e03-cache-fast`
(original smoke/numerical controllers have finished):
smoke cache -> baseline/cache/patch smoke -> full cache -> fresh formal E03-Patch.
Each stage must succeed before the next starts. Live state/logs are under
`reports/E03_speed_optimization/pipeline_status.json` and `pipeline.log`.
Four-GPU measurements, raw-latent checks and numerical-repeat control have passed.
Full cache generation is running; formal launch remains gated on complete cache coverage.

## Actual baseline data flow

Per rank batch B=64, two cameras, consecutive frames 0..32:

| Stage | E03 / E03-Cache | E03-Patch |
|---|---|---|
| Reader RGB | 64 clips, 33 PIL images, each 448x224 | same pixels/context |
| RGB tensor | `[64,3,33,224,448]`, BF16 in [-1,1] | skipped at cache hit |
| Independent views | `[128,3,33,224,224]` | unchanged in cache builder |
| Native VAE output | `[128,16,9,28,28]` | same cache |
| Width-stitched raw latents | `[64,16,9,28,56]` | identical |
| Existing/replacement patch | `(1,2,2)` | `(1,4,4)` |
| Clean/future input projection | `[64,1536,9,14,28]` | `[64,1536,9,7,14]` |
| M29 visual sequence | `[64,3528,1536]` | `[64,882,1536]` |
| Video output/loss target | `[64,16,9,28,56]` | unchanged |

Shapes were exercised with CPU test encoders and the real M29 implementation at reduced width/depth;
the GPU smoke script records actual full-size tensors per measured step. Earlier real-VAE geometry
checks are in `reports/wan_vae_encoder_check.json`.

3528 = 9 * 14 * (14 * 2). Baseline already uses a 2x2 spatial patch, with no duplicate token
expansion. 28x28 is the native Wan2.1 /8 grid for each 224x224 view. Temporal /4 maps 33
frames to 9 latents. There is no added high-resolution input or accidental view duplication.

Relevant code:

- `openwam/dataloader/bases/lerobot_v3_reader.py`: window selection, frame decode and cache bypass.
- `openwam/model/video_backbone/encoder/wan_vae.py`: input normalization, independent views and frozen VAE.
- `openwam/model/video_backbone/wan/models/vae.py:VideoVAE_.encode`: native causal chunk loop and state reset.
- `openwam/model/video_backbone/encoder/bottleneck.py`: direct clean projection, low-rank future projection.
- `openwam/model/video_backbone/wan_backbone.py:prepare`: patchification, per-view RoPE and embeddings.
- `openwam/model/architectures/dual_system/joint_self_attn.py`: joint video/action block loop.

The VAE is frozen, forced to eval and evaluated under no_grad. A complete training batch invokes
its public encoder once, split internally into 128 single-view clips at microbatch=1. Each native
33-frame clip has 9 temporal encoder chunks: 1152 native encoder chunk invocations per rank/step.
TI2V reuses the first latent and does not independently re-encode the clean frame during training.
Adjacent windows overlap, and all ten epochs repeat frozen encoding, but causal state is reset at
each window start. Whole-episode encoding sliced into windows is therefore not equivalent.

The 30-layer trainable M29 also receives 3.6x E01's visual sequence length. Instrumented steady
baseline step is 20.91 s: VAE 10.10 s, RGB preprocessing 1.14 s, M29 forward 2.33 s, backward and
DeepSpeed update 7.27 s, DataLoader wait about 7 ms. The online encoder path accounts for about
54% of the step. Projection is about 2.7 ms. Attention alone was not the principal established cause.

## E03-Cache: computation-only change

Profile: `configs/experiments/E03-Cache.yaml`.

- Cache contains only original, normalized posterior-mean BF16 VAE outputs, before any trainable projection.
- Each entry is a full 33-frame window with independent view encoding and unchanged padding.
- 256-index shards, atomic file replacement under per-shard flock; readers mmap only the requested tensor.
- No in-memory global feature table and no per-epoch cache copy. Each DataLoader worker opens its own handle.
- Cache identity locks data configuration, metadata content, media size/mtime, preprocessing source hashes,
  official VAE SHA256, normalization and temporal/view contracts. Model spatial patch is deliberately
  excluded from the raw-latent identity, so Cache and Patch share the same files.
- Stochastic color jitter, history and non-dense frame sampling are rejected. The current reader has no
  random RGB augmentations on this profile. Any future augmentation requires separate validation.
- Missing/corrupt entries raise; no silent reader fallback to the next sample. Full training preflight
  checks that every window is present; partial smoke caches cannot start a formal run.
- Deployment still uses the frozen RGB encoder, with no training cache dependency.

Exact size: 273465 * 16 * 9 * 28 * 56 * 2 = **123,492,418,560 bytes**, about **115.01 GiB**, plus
small safetensors headers. Do not create an episode cache plus duplicate window cache. The builder
defaults to estimate-only, and can populate just the epoch-0 sampler prefix needed by smoke tests.

```bash
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
# Estimate only, no encoding or cache payload generated:
$PY scripts/precompute_wan_latents.py
# Populate smoke windows and prefetch only, four GPUs (after they are free):
$PY -m torch.distributed.run --standalone --nproc_per_node=4 scripts/precompute_wan_latents.py --smoke-steps 8
# Full offline build, only when intentionally requested; resumable by shard/index:
$PY -m torch.distributed.run --standalone --nproc_per_node=4 scripts/precompute_wan_latents.py --build
```

CPU tests verify exact input/conditioning, loss and projection-gradient equality at rtol=atol=0.
Real LIBERO reader routing is checked at indices 0,136732,273464 with a small CPU encoder fixture,
including the padded episode tail. Real pretrained-Wan verification passed for 16 builder samples
and the complete first training batch: 4 ranks x 64 windows, every latent element exactly equal,
max absolute error zero. See `gpu/cache_parity_rank*.json` under the report directory.
First-step training losses are bitwise equal; subsequent BF16 trajectories have small differences.
The original 1e-5 relative trajectory threshold rejected them, so an additional repeated-online
control measured the training process's own numerical variation. It passed; see
`reports/E03_speed_optimization/gpu/numerical_control.json`. Maximum total-loss relative deviation
was 0.0112% online/cache versus 0.0094% online/repeated-online; gradient-norm deviation was
0.0162% versus 0.0229%. The acceptance envelope is three times the measured online-repeat
relative deviation per metric with a 1e-4 rounding floor and a hard 0.5% ceiling. This is numerical
trajectory consistency, not bitwise training reproducibility. Raw-cache identity remains strict
rtol=atol=0 and is not relaxed by the trajectory control.

## E03-Patch: explicitly different model structure

Profile: `configs/experiments/E03-Patch.yaml`.

The existing 2x2 patch is replaced by 4x4, rather than stacking another lossy pooling stage. A learned
Conv3d with kernel=stride=(1,4,4) packs local spatial regions; clean and future projections remain
separate, with the same 256-dimensional future bottleneck. All 9 temporal positions remain.

The output head predicts 16*4*4 channels per token and unpatchifies to the original latent grid.
Thus raw VAE features and video loss targets are not pooled. Per-view boundaries are divisible by
4; local RoPE and view embeddings use 7 columns per camera and do not cross the seam in patchification.
Attention and time masks derive their spatial count from the patch contract. CPU tests exercise
clean-prefix causal masks, forward/backward and nonzero, finite gradients updating patch weights.

| Component | Baseline / Cache | Patch4 | Delta |
|---|---:|---:|---:|
| Input projections | 510,976 | 855,040 | +344,064 |
| Output head | 98,368 | 393,472 | +295,104 |
| Total | 609,344 | 1,248,512 | **+639,168** |

These are scratch-initialized trainable policy parameters, with no pretrained projection weights.
The VAE remains frozen. No action/text/loss/batch/epoch/optimizer setting is changed by either profile.
Patch4 changes the representation-to-policy interface and is not computation-equivalent to E03.
It can be compared under a fixed M29 Transformer/action/text backbone, provided encoder-specific
tokenization/adapters are part of the representation definition and their parameter counts are reported.
This is also spatially aligned with the ViT profiles: VAE stride8 * patch4 = ViT stride16 * patch2 =
32 RGB pixels per policy token, giving each camera a 7x7 grid. It is not an encoder-weights-only ablation.

Remaining comparison caveats: E01/E02 sample sparse RGB frames at stride4 and retain historical/current
observation information; E03 uses dense33-frame future supervision and current-only deployment observation.
Temporal token counts, clean-prefix organization, feature normalization and latent loss spaces differ.
Thus current results compare native representation pipelines under the same M29 family, not encoder
weights under identical observations/adapters. A strict single-factor study would need a separate common
RGB/history/adapter protocol and corresponding reruns of all encoders. Training loss values across native
feature spaces are not directly comparable; closed-loop LIBERO/Plus success is the primary outcome.

## Two-view A vs B investigation

The [Fast-WAM paper](https://arxiv.org/html/2603.16666v1) and
[official repository](https://github.com/yuantianyuan01/FastWAM) provide the reference context.
The following comparison uses this experiment's Wan2.1 and unchanged 224x224 camera resolution;
it does not assert that every FastWAM version uses the same exact geometry.

| Item | A: independent VAE then width concat | B: RGB width concat then VAE |
|---|---|---|
| Per-sample VAE inputs | 2 x `[1,3,33,224,224]` | `[1,3,33,224,448]` |
| Raw latents after packing | `[1,16,9,28,56]` | `[1,16,9,28,56]` |
| Baseline M29 tokens | 3528 | 3528 |
| Convolution spatial workload | two small canvases | one double-width canvas, roughly same area |
| VAE spatial attention | isolated within each camera | includes both cameras; more pairwise work |
| Cross-camera mixing before policy | none | convolution mixes at seam; attention mixes globally |

B reduces invocation count but does not reduce tokens or automatically halve compute. Native VAE
middle AttentionBlock can make the wide-canvas attention more expensive. It changes the representation,
including pixels away from the seam through attention. No B profile was activated. An isolated
`scripts/probe_wan_view_layout.py` measures both without changing any training config; it has not yet
run because all GPUs are occupied.

## Tests, timings and run readiness

`reports/E03_speed_optimization/cpu_tests.log` covers cache round-trip, multiprocess reads, missing
and incomplete cache handling, input/loss/gradient equality, patch gradients, old checkpoint loading,
and Patch4 ordinary/batched LIBERO inference plus LIBERO-Plus checkpoint relocation. These use a
small encoder and reduced-size M29 to check semantics, not to benchmark throughput.

`reports/E03_speed_optimization/real_reader_cpu.json` records the actual reader checks and clearly
labels the encoder fixture. Both new configurations also pass Hydra dry-run resolution.

The opt-in StepTimer now records RGB preprocessing, VAE (including native microbatches), projection,
M29 forward, backward/DeepSpeed step, optimizer and host DataLoader wait, plus actual tensor shapes.
Nested CUDA-event phases overlap and must not be summed. Backward includes checkpoint recomputation
and distributed optimizer work; a separate DeepSpeed step phase identifies the latter.

```bash
# Refuses to run when GPUs are occupied; never stops another process or launches full training.
$PY scripts/smoke_wan_optimization.py --steps 8 --warmup 3
# Optional isolated view-layout probe after GPUs are free:
$PY scripts/probe_wan_view_layout.py
```

The suite runs all three profiles sequentially at 4 x batch64, uses the same seeded sampler/noise,
and writes per-rank timing JSONL and a consolidated `reports/E03_speed_optimization/gpu/summary.json`.
Only the smoke horizon is truncated equally for these probes. Checkpoint output stays under each
profile's smoke path. No existing formal checkpoint is overwritten.

| Variant | Tokens | Measured steady step | VAE | M29 forward/backward | Peak memory |
|---|---:|---|---|---|---|
| Original E03 | 3528 | 20.911 s | 10.101 s | 2.335 / 7.271 s | 58.858 GiB |
| E03-Cache | 3528 | 9.470 s | no online forward | 2.338 / 7.076 s | 58.754 GiB |
| E03-Patch | 882 | 2.435 s | no online forward | 0.578 / 1.819 s | 22.551 GiB |

All three: four GPUs, batch64/rank, same 8-step smoke schedule, first3 warmup, final5 measured.
Step is average of the maximum rank time at each step; phase averages above are rank0. Startup,
cache construction and final full-state checkpoint I/O are excluded and must be reported separately.
Cache speedup is 2.21x; Cache+Patch speedup is 8.59x relative to the online baseline. The latter
includes a model-interface change and must not be described as computational equivalence.

Real-weight GPU acceptance passed; full offline cache coverage remains outstanding. Cache isolates
computation from model changes relative to E03. The user
authorized fresh E03-Patch training after acceptance; this compares native encoder/adapter pipelines
with a fixed M29 Transformer and approximately aligned spatial token geometry. Report it explicitly as
E03-Patch rather than overwriting E03, and retain the observation/temporal caveats above.

Original E03 profile SHA256: `6e637287da63459865ad43675efdaf1715c0e753320277b1d4fa977bbf0c80a4`.
Original code/config snapshot before these edits: `reports/E03_speed_optimization/baseline_source.tar.gz`.
Current full-state identity intentionally changes with source edits; restoring an old formal run
must use its archived source rather than bypassing identity validation.

## Cache-generation optimization, 2026-10-25 11:14

The original one-process/GPU builder was CPU-serial: decode, preprocess, launch microbatch1 VAE,
copy latent to CPU, then decode the next window. GPU utilization oscillated despite low VRAM.
Low memory itself is expected for a frozen encoder without training activations/optimizer.

Microbatch1/2/4/8/16/32 were measured on 16 real windows. Encoder-only throughput rose from
6.86 to 9.16 windows/s per GPU, but microbatch>1 changed BF16 rounding (max absolute latent
difference 0.03125). This alternative was rejected to preserve exact parity with the existing cache.
Results: `reports/E03_speed_optimization/cache_batch_probe.json`.

Independent processes retain microbatch1 and native per-view/window state, allowing CPU decoding
and GPU execution to overlap across workers. Synchronized four-GPU end-to-end measurements:

| Processes/GPU | Total processes | Windows checked | Total windows/s | Bitwise cache parity |
|---|---:|---:|---:|---|
| 1 | 4 | 128 | 15.50 | exact |
| 2 | 8 | 256 | 18.68 | exact |
| 4 | 16 | 512 | 21.87 | exact |

The selected 4-process/GPU configuration is 1.41x faster in this probe; GPU residency was about
5.6 GiB/card versus 1.4 GiB/card originally. Probe includes RGB decoding, preprocessing, encoding,
CPU transfer and cache parity reads; it excludes startup and atomic shard serialization. Actual
generation throughput/ETA should be checked from ongoing shard progress.

`scripts/precompute_wan_latents.py` now accepts `--workers-per-gpu`; local process ranks map to
GPU `LOCAL_RANK // workers_per_gpu`, while shard ownership remains `shard_index % WORLD_SIZE`.
Complete shards are reused without rewriting; incomplete shards retain all existing entries and
are atomically extended. Changing worker count does not change the cache identity or representation.

The old four-process build was deliberately stopped; completed atomic shards were retained.
The validated controller `reports/E03_speed_optimization/resume_optimized_cache.py` selects the
fastest exact-equivalent probe, restarts the full build at 16 processes, then continues the prior
fresh E03-Patch training authorization after complete-cache preflight. Failure blocks training.

```bash
$PY -m torch.distributed.run --standalone --nproc_per_node=16 \
  scripts/precompute_wan_latents.py --build --workers-per-gpu 4
```

Selection and latest control state: `cache_worker_selection.json` and `pipeline_status.json`
under `reports/E03_speed_optimization/`. No encoder/model/patch/training hyperparameter changed.

## CPU prefetch and asynchronous shard writes, 2026-10-25

Reused the pipeline pattern from `OpenWAM/openwam/cache/training.py`: a CPU DataLoader
prefetches strict-index samples, while one writer thread publishes bounded pending shards.
Each of the existing 16 encoder processes now has two spawn-based CPU reader workers,
prefetch factor two, and at most two pending shard writes. GPU microbatch remains one;
the older V-JEPA batch128 setting is deliberately not copied because it changes Wan BF16 numerics.

Only missing windows and a few parity-check windows are decoded. Complete shards are retained.
The writer acquires the original per-shard process lock, merges any existing entries, rejects
conflicting duplicates, and atomically replaces the file. All writer futures must succeed before
generation reports success or the training controller proceeds. The cache identity, original
latent format, preprocessing, camera layout, and training configuration are unchanged.

Focused validation: `tests/test_wan_cache_pipeline.py` checks concurrent merge safety and
preservation of a published shard after a writer error (two tests passed). A real eight-window
generation with two CPU workers checks its first four windows against independently decoded,
synchronous original VAE encoding: bitwise equal, maximum error zero. A second invocation checks
persisted entries and must generate zero additional windows. Logs live in
`reports/E03_speed_optimization/prefetch{,_resume}_validation.log`.
This small correctness test includes worker startup and is not a throughput benchmark.
Production progress reports now include loader wait, encoding and writer backpressure time.
