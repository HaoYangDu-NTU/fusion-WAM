# FM-VJEPA-Policy evaluation

The existing scripts default to `--policy-backend openwam`; old checkpoint layouts,
OpenWAM image/state processing, action conversion, text caches and resume identities
remain unchanged. Select `--policy-backend fm-vjepa` explicitly for the native FM
checkpoint format (`checkpoint_step010680.pt`, `resolved_args.json`, `dataset_stats.json`).
This adds an evaluation adapter, not a model or training change.

FM uses the saved native serving configuration, ViT-L encoder, independent camera
encoding, raw position3/axis-angle3/gripper-qpos2 state, and T5-v1.1-XXL length128 text.
The server receives rotated, unresized RGB images. Native serving performs the
training resize/normalization. History is recorded on every control step, with t-4
clamped to the first policy observation. Native `infer_batch` produces 32 environment
actions, with 16 executed before replanning. No extra EEF10 or gripper conversion is
applied. Each active episode has independent history and an action queue.

The harness retains its existing benchmark protocol (settling, task RNG, trials,
termination, original Plus instructions by default). LIBERO has 40 tasks × 50 trials;
Plus has 10030 tasks × 1 trial. A 20-step `--smoke` run is not a success-rate result.
The wrapper's `--correct-instruction-suffix` still selects Plus text behavior;
its matching **native FM** text cache must already cover all selected prompts.
OpenWAM's UMT5 cache cannot be used.

## haoyang environment and seeded validation

The private `/data/fcy/code/OpenWAM/.venvs/fm-eval` environment inherits Torch 2.11
from the existing lerobot environment and shadows required pure Python packages
using private links. Existing OpenWAM and simulator environments are unchanged.
The declarative spec, offline build notes, native action-equivalence witness and
logs live in `diagnose_model/fm_harness_20261001/`.

Run this documented validation from the local machine (loads one FM model on GPU 1):

```bash
ssh haoyang-pretrain 'cd /data/fcy/code/OpenWAM && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONPATH=. .venvs/fm-eval/bin/python -u diagnose_model/fm_harness_20261001/witness.py'
```

Expected sentinel: `FM_NATIVE_WITNESS`, shape `[32, 7]`, maximum absolute error
`0.0`, `batch2: 2`, finite output, NVIDIA H100. This compares the adapter with native
serving at the same random seed, using independently constructed native inputs.

Run the regression suite on haoyang:

```bash
cd /data/fcy/code/OpenWAM
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /data/fcy/basics/miniconda3/envs/openwam/bin/python -m pytest -q tests/test_vla_harness_fm.py tests/test_vla_harness_openwam.py tests/test_vla_harness_libero_plus.py tests/test_vla_harness_resume.py
```

## Invocation

On haoyang, from `/data/fcy/code/OpenWAM`:

```bash
bash scripts/eval_libero_vjepa.sh \
  diagnose_model/future_prediction_analysis/fm_vjepa_policy/runs/fm_vjepa_4gpu_bs64_10epochs_20260929_023453 \
  checkpoint_step010680.pt \
  --policy-backend fm-vjepa \
  --fm-python /data/fcy/code/OpenWAM/.venvs/fm-eval/bin/python \
  --fm-encoder /data/fcy/code/OpenWAM/assets/vjepa/vjepa2_1_vitl_dist_vitG_384.pt \
  --fm-text-cache /data/fcy/code/OpenWAM/data/fm-vjepa-text128/original_text \
  --benchmark libero --gpus 0,1,2,3 --workers 32 --max-batch-size 8 \
  --base-port 8950 --output-dir outputs/fm_libero
```

For a smoke test add `--smoke --suites libero_spatial --gpus 1 --workers 2
--max-batch-size 2 --base-port 14940 --render cpu` and use a fresh output directory.
For Plus use `--benchmark libero-plus` (its existing dedicated interpreter remains
responsible for simulator execution).

For resumable LIBERO followed by Plus use the same three positional arguments
(checkpoint directory, checkpoint name, output directory) with
`scripts/eval_libero_then_plus.sh`, passing the same FM flags. The cache must include
`manifest.json`; its hash and the model assets are bound to `job.json` on first run.

`python -m benchmarks.vla_harness.run_after --after-output-dir PREDECESSOR
--output-dir NEXT --ready-file TEXT_CACHE/manifest.json -- COMMAND ...` waits for
predecessor `DONE`, both complete non-smoke summaries, and a completed text manifest.
It does not interpret a crashed/stopped process as completion. `NEXT/STOP` cancels
waiting; duplicate queue/controller instances are rejected by file locks. The
same `NEXT/STOP` stops the resumable controller after launch.
