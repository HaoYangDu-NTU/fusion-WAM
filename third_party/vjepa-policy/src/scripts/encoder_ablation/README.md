# Encoder Ablation

These launchers keep the policy, data, seed, global batch size, and 21,360-step
schedule fixed while replacing only the frozen video encoder.

```bash
bash scripts/encoder_ablation/run.sh vjepa2_vitl
bash scripts/encoder_ablation/run.sh vjepa2_1_vitl
bash scripts/encoder_ablation/run.sh vjepa2_1_vitg
```

`run_queue.sh` performs a one-step memory preflight at ACB 4, 8, then 12,
uses the fastest passing configuration, trains, and runs all four LIBERO suites.
