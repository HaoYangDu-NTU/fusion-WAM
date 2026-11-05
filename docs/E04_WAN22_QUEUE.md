# E04: M29 + RynnBrain + Wan2.2 VAE encoder

Authorized background sequence: finish E03 LIBERO-Plus, record complete results in
`result.md`, release evaluation GPUs/ports, validate the real Wan2.2 VAE encoder,
generate its full latent cache, validate cached training, then train E04 from scratch.

Controller: `reports/E04_wan22/pipeline.py`. Live state and all stage logs are under
`reports/E04_wan22/`; the tmux session is `openwam-exp-e04-watcher`.
No GPU work begins before the E03 evaluation `DONE` marker and result section exist.
A stage failure blocks all later stages. `reports/E04_wan22/STOP` prevents the next
stage from starting; it does not interrupt an already running child stage.

The encoder uses only `encoder.*` and `conv1.*` from the verified
`encoder/wan22-vae/Wan2.2_VAE.pth`; it is frozen, BF16, microbatch one. Preprocessing,
33 dense frames, native temporal resets, and independent 224x224 camera views match
E03. Raw latents are `[48,9,14,28]` after the two view grids are concatenated in width.
The native spatial stride is 16; spatial patch two gives 882 tokens, matching
E03-Patch's stride eight plus spatial patch four. Channels/representation and the
necessary input/output projections differ; the M29 Transformer, RynnBrain text
encoder, action backbone, loss, batch256, ten epochs and 10,690-step budget stay fixed.

Cache: `cache/wan22_libero_windows`, 273,465 windows, 86.26 GiB raw BF16 tensors.
The generator reuses CPU prefetch, bounded asynchronous writes, process locks,
atomic shard publication, and restart-safe merging. Existing Wan2.1 cache identity
remains byte-for-byte compatible. No E03 shards are reused as Wan2.2 features.

After evaluation, eight real windows first validate exact synchronous versus
prefetched encoding and the expected shape. Full generation then uses 16 encoder
processes across four GPUs, two CPU workers per process, prefetch factor two, and
at most two pending writes per process. Every process checks exact numerical parity.
Complete cache coverage/identity and pretrained asset SHA256 are required. A four-step,
four-GPU smoke at the formal batch size must have finite losses, nonzero gradients,
and a valid native encoder checkpoint before fresh training starts. Formal training
does not load the smoke weights, old E03 weights, or optimizer state.

CPU checks: 13 tests passed, including separate version/weight identities, Wan2.2
geometry, frozen encoder, asynchronous shard safety, and unchanged comparison settings.
The existing complete Wan2.1 cache manifest was also recomputed and matched exactly.
GPU acceptance remains pending until the currently running E03 evaluation finishes.

Formal output root: `outputs/m29_rynn9b_wan22_vae/train/s42`. The controller refuses
an existing formal run or a second formal launch. This queue schedules E04 training;
it does not automatically schedule E04 benchmark evaluation.

2026-10-26 recovery: full cache generation and parity checks succeeded, but the first
training smoke exposed a Wan2.1-only version/shape guard in
`WanVideoBackbone.preprocess_cached_input_for_train`. The guard now accepts both
native versions and checks their separate raw shapes. A regression test exercises
Wan2.2 cached input preparation, first-latent conditioning and projection gradients;
12 related tests passed. The failed smoke log is retained with suffix
`.failed_20261026_0353`. Restart with `--reuse-complete-cache` requires the previous
cache acceptance, identical identity and complete coverage; it skips re-encoding
and reruns smoke before fresh formal training.
