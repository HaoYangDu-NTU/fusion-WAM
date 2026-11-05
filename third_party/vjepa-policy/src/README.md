# VJEPA-Policy

VJEPA-Policy combines a frozen V-JEPA2 video encoder, a trainable latent world-model
Predictor, and a flow-matching Action Expert. Language instructions and robot
proprioception condition both trainable components through token-level cross-attention.

## Architecture

`WorldModelPredictor` receives context latents from the frozen encoder and mask tokens
for future positions. Each Transformer block applies 3D-RoPE self-attention, cross-attends
to T5 and proprioception tokens, and exposes its context K/V tensors.

`ActionExpert` is a DiT-style flow model. At every layer, action queries attend to the
corresponding Predictor context K/V and the action chunk's own K/V, then cross-attend to
the same language and proprioception tokens. Diffusion time modulates the attention and
feed-forward paths with adaptive RMS normalization.

The training interpolation is

```text
a_t = (1 - t) * epsilon + t * action
velocity_target = action - epsilon
```

Inference integrates the predicted velocity from noise to action with Euler steps.

## Repository Layout

```text
src/vjepa_policy/
  datasets/                 # LeRobot video/action dataset and normalization
  hub/                      # torch.hub V-JEPA2 builders
  models/
    vision_transformer.py   # official V-JEPA2 encoder
    predictor.py            # official V-JEPA2 predictor
    ac_predictor.py         # official V-JEPA2-AC predictor
    vjepa2_1/               # official V-JEPA2.1 implementation
    lang_predictor.py       # earlier future-latent similarity model
    wm_predictor.py         # language/state-conditioned latent Predictor
    action_expert.py        # flow-matching Action Expert
    vjepa_policy.py         # joint training policy
  policy_serving/           # checkpoint loading and LIBERO inference
  text_embeddings.py        # T5 instruction-cache CLI
  train_common.py           # data/model construction shared by training
  trainer.py                # Accelerate training loop
scripts/
  train_vjepa_policy.py     # canonical Python training entrypoint
  train_vjepa_policy.sh     # four-GPU default launcher
tests/                      # public architecture and data-contract tests
examples/libero/            # optional LIBERO serving, client, and latent evaluation
```

## Installation

```bash
pip install -e .
```

The official V-JEPA2 ViT-G configuration uses 22 encoder attention heads. Training
rejects the legacy 16-head compatibility model.

## Data

Training expects local LeRobot datasets and precomputed T5-XXL instruction embeddings.
Generate the text cache with:

```bash
python -m vjepa_policy.text_embeddings \
  --dataset-dirs /path/to/libero_*_lerobot \
  --cache-dir /path/to/text_cache \
  --model-name google/t5-v1_1-xxl
```

The default policy samples native frame offsets
`[-4, 0, 4, ..., 28, 32]` and actions `[0, ..., 31]`. All LeRobot image/video features
are discovered from metadata, sorted by feature key, and independently resized to
`224 x 224`. The encoder processes each view with its own local `(t, h, w)` coordinates.
Predictor tokens are ordered by `(t, view, h, w)` and receive a learned view embedding
after the encoder-to-Predictor projection. Explicit `--views` overrides discovery and
fixes the camera order when loading a checkpoint trained with a particular ordering.
Action and state features use quantile normalization.

Datasets with a larger robot state can select an ordered subset through
`--action-indices` and `--state-indices`. The same projection is applied to metadata
statistics before normalization and is recorded in checkpoint topology. Datasets whose
natural-language instruction is stored per episode can use `--instruction-field remarks`
for both text-cache generation and training.

## Training

The shell launcher defaults to four GPUs, per-device batch size 32, global batch size
128, 21,360 optimizer steps, 10 epochs, seed 7, and BF16:

```bash
export LIBERO_DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA
export VJEPA2_ENCODER_CHECKPOINT=/path/to/vitg.pt
export TEXT_EMBEDDING_CACHE=/path/to/text_cache

bash scripts/train_vjepa_policy.sh
```

`DATASET_STATS=/path/to/dataset_stats.json` is optional. When omitted, statistics are
aggregated from LeRobot metadata and saved with the run.

The canonical Python entrypoint exposes all architecture and optimization settings:

```bash
python scripts/train_vjepa_policy.py --help
```

Its default trainable architecture is a 24-layer, 1024-dimensional, 16-head Predictor
and a 24-layer, 512-dimensional Action Expert. Predictor RoPE uses corrected frequency
pairing and endpoint-aligned spatial interpolation. Each frozen encoder view also uses
endpoint-aligned spatial interpolation from the `224 x 224` input grid to the pretrained
RoPE grid.

The reproducible frozen-encoder comparisons live under `scripts/encoder_ablation/`.
They support V-JEPA2 ViT-L, V-JEPA2.1 ViT-L, and V-JEPA2.1 ViT-G while keeping the
policy, data, global batch, schedule, and seed fixed. V-JEPA2.1 is always called with
`training=False`, so only the final normalized layer output conditions the Predictor.

## Serving

The package-level LIBERO adapter loads the frozen encoder and trained state dictionaries:

```python
from vjepa_policy.policy_serving import PolicyServingConfig, VJEPAPolicyServing

policy = VJEPAPolicyServing(
    ckpt_path="/path/to/checkpoint_step021360.pt",
    pretrained_encoder="/path/to/vitg.pt",
    text_cache_dir="/path/to/text_cache",
    config=PolicyServingConfig(dataset_stats="/path/to/dataset_stats.json"),
)
actions = policy.infer(observation)["actions"]
```

The adapter accepts checkpoints created before the public API rename. Legacy class names
stored in checkpoint topology metadata are canonicalized before validation; parameter
loading remains strict.

The default LIBERO serving contract matches the independent-view policy: both main and
wrist cameras are resized independently to `224 x 224`, encoder RoPE interpolation is
enabled, and the resulting view-token sequences are distinguished by learned view
embeddings. The simulator may still render observations at its native `256 x 256`;
that resolution is not the policy input layout.

An optional openpi websocket entrypoint is provided for LIBERO evaluation:

```bash
python examples/libero/serve_policy.py \
  --ckpt /path/to/checkpoint_step021360.pt \
  --pretrained-encoder /path/to/vitg.pt \
  --text-cache-dir /path/to/text_cache \
  --dataset-stats /path/to/dataset_stats.json
```

To evaluate all four LIBERO suites with the same defaults:

```bash
export VJEPA2_ENCODER_CHECKPOINT=/path/to/vitg.pt
export TEXT_EMBEDDING_CACHE=/path/to/text_cache
export POLICY_PYTHON=/path/to/policy/python
export LIBERO_PYTHON=/path/to/libero/python

bash examples/libero/evaluate_policy.sh /path/to/run_dir
```

## Checkpoints

The encoder is frozen and is not duplicated in policy checkpoints. A checkpoint contains
Predictor and Action Expert state dictionaries, optimizer-step metadata, precision, seed,
and model topology. Normalization statistics are stored separately as
`dataset_stats.json` and must accompany the checkpoint. Class and file renaming does not
change parameter keys.

## Publishing

Do not upload local training artifacts or rollback snapshots. A minimal source release
contains `src/`, `scripts/`, `tests/`, optional `examples/libero/`, `hubconf.py`,
`pyproject.toml`, `README.md`, `LICENSE`, and `.gitignore`.
