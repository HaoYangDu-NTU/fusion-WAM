# Route 2 experiments

These launchers reproduce the corrected-RoPE Route 2 control and its three
single-variable follow-ups. All default to BF16, seed 7, global batch 128,
21,360 optimizer steps, and 10 epochs.

| Directory | Controlled change |
| --- | --- |
| `baseline` | Route 2 Predictor with the historical FiLM Action Expert |
| `action_cross` | Replaces Action Expert FiLM with language/state cross-attention |
| `clean_context` | Clean-context Predictor attention and full Predictor K/V for actions |
| `horizontal224` | Two 224x224 views concatenated horizontally with encoder RoPE interpolation |
| `independent224` | Two independently encoded 224x224 views with local RoPE and learned view embeddings |

Run from the repository root. Environment variables may override any path or
runtime setting shown in the launcher.

```bash
bash scripts/route2/action_cross/run.sh
```

Evaluate the original Route 2 + Action Expert cross-attention checkpoint on
LIBERO Seeds 17 and 29 concurrently:

```bash
bash scripts/route2/action_cross/eval_seed_stability_parallel.sh
```

The independent-view experiment includes a matching evaluation wrapper. It
uses two local 224x224 camera grids, re-plans every 16 actions, and evaluates
50 initial states per task across all four LIBERO suites:

```bash
bash scripts/route2/independent224/eval.sh RUN_DIR
```
