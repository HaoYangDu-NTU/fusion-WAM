# E01/E02 frozen image feature cache

`E01-Cache` and `E02-Cache` are separate performance configurations. Original E01/E02,
their checkpoints and results remain available. Both cache configurations preserve
the model, seed, batch, training budget, losses and deployment protocol.

The cache stores BF16 raw frozen-backbone grids at actual episode timestamps:

| Experiment | Stored stereo grid | Assembled input | Online adaptation | M29 tokens |
|---|---|---|---|---:|
| E01 | `[1280,14,28]` | `[B,1280,10,14,28]` | original frozen output normalization | 980 |
| E02 | `[1408,14,28]` | `[B,1408,10,14,28]` | original learned history delta, future pair mean, normalization | 490 |

Views are encoded independently and their grids concatenated along width. Temporal
offsets are `[-4,0,4,8,12,16,20,24,28,32]`; history and future padding clamp inside the
same episode. No projected features, learned temporal adapters or diffusion targets
are stored. E02's existing adapter executes in `compute_loss`, preserving its gradients.

Dataset workers read actions/proprio/prompt/masks through the original reader hooks,
without RGB decoding. The collator batches timestamp lookups, deduplicates repeated
frames within the batch and reuses a bounded LRU of mmap shard handles. This differs
from loading whole shards into Python RAM. Workers prefetch batches while the GPU
trains; moving reads into collate alone is not an additional source of asynchronous I/O.

At 273,465 timestamps, cache payload estimates are 255.58 GiB (E01) and 281.14 GiB
(E02), excluding small headers/metadata. Storing all ten grids independently per
window would require about ten times as much space.

## Validation status

Real pretrained encoders and real LIBERO samples passed CPU smoke tests with a
reduced one-layer M29: loss/backward, frozen-backbone bypass, identical serialized
features/loss/projection gradients, updated projection, E02 adapter gradients, and
unchanged action/proprio/text/padding at episode boundaries. Reports:
`reports/image_frame_cache/E01/smoke.json` and `E02/smoke.json`.

These CPU checks preserve the encoding batch used to produce the compared features.
Independent single-timestamp CPU BF16 encoding differs numerically from window-batched
encoding (mean cosine approximately 0.9992 / 0.9991). They therefore do **not** certify
offline-batch versus production-GPU-batch equivalence. No full cache or formal cached
training has been launched. Production cached training is gated by an identity-bound
CUDA/batch64 validation file; if it fails, use the original online configuration.

## Commands

Estimate only (does not load the encoder or create a full cache):

```bash
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
"$PY" scripts/precompute_image_features.py --experiment E01
"$PY" scripts/precompute_image_features.py --experiment E02
```

On idle GPUs, explicitly build, validate, then use the separate cache profile:

```bash
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/precompute_image_features.py --experiment E01 --build
"$PY" scripts/validate_image_features.py --experiment E01
"$PY" scripts/experiment.py train E01-Cache --gpus 0,1,2,3 --smoke-steps 4
# Only after acceptance:
"$PY" scripts/experiment.py train E01-Cache --gpus 0,1,2,3
```

Substitute E02 for the corresponding second experiment. `--limit-frames` creates a
bounded diagnostic cache, which cannot satisfy formal training coverage validation.
Generation supports shard-level resume, locks, atomic writes, CPU prefetch and bounded
asynchronous writes. Deployment still encodes available RGB online; parameter names
and checkpoint state layout are unchanged.

## E01-P1 cache preparation, 2026-10-30

The online E01-P1 queue was stopped at the user's request before formal training.
Its four-step online smoke completed; the inference smoke was interrupted. The
original queue retains its STOP marker and must not be resumed unchanged.

`E01-P1-Cache` inherits E01-Cache and applies P1's video=x/action=v and shift42.
Only the feature source and output/run name differ from online P1. DINO inputs,
normalization, projection, clean/future packing, masks and training budget remain.

The bounded GPU probe (`scripts/probe_image_frame_cache.py`) passed on H100 with
Torch 2.7.1+cu128: windows 0, 64 and 273464 have identical pixels, raw features and
normalized features for offline frame batches versus online batch64 (max error 0).
This is not full-cache coverage acceptance. Report:
`reports/image_frame_cache/E01/gpu_batch64_probe_20261030.json`.

`scripts/prepare_e01_p1_cache.py` runs a finite preparation sequence: four-GPU cache
generation, full-cache coverage/numerical validation, four-step P1 cached smoke,
and one standard-LIBERO inference smoke. It **never starts formal training**.
Status/logs: `reports/E01_P1_visual_cache_20261030/`; its own STOP file stops only
these preparation stages. The cache manifest also pins local model JSON/Python
metadata and the Torch version. Later training requires a fresh explicit launch.
