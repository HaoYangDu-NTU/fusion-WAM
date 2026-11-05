# Action-only LIBERO ablation

This isolated variant keeps the frozen context encoder, endpoint-aligned Predictor
RoPE, Predictor K/V path, and Action Expert unchanged. It removes the unoptimized
future-target encoder pass and latent prediction head from training, reporting
`loss_world=0` and `cos_sim=0`.

The two comparison runs use four GPUs each, per-device batch 32, global batch 128,
21,360 optimizer steps (10 epochs), BF16, and seed 7. `launch_pair.sh` assigns
V-JEPA2 ViT-L to GPUs 0-3 and V-JEPA2.1 ViT-L to GPUs 4-7, then automatically
evaluates each checkpoint with replan 16, 50 trials per task, and all four LIBERO
suites.

The launcher uses seven activation-checkpointed Predictor blocks. Both encoders
completed a three-step four-GPU preflight at this setting, peaking near 80,401
MiB of 81,559 MiB per H100 after optimizer state allocation; six blocks OOM on
the second forward.

The policy server uses `/data/zhangyang/miniconda3/envs/prts-ft/bin/python`; the
simulation client uses `/data/zhangyang/miniconda3/envs/libero/bin/python` and
the existing `/data/zhangyang/workspace/LIBERO` installation. The launcher does
not modify either environment.

```bash
bash scripts/action_only_ablation/launch_pair.sh
```

If one four-GPU group is unavailable, `queue_vjepa21_healthy.sh` can wait for a
completed V-JEPA2 training plus four-suite evaluation, verify GPUs 0-3 are
healthy and idle, and then launch V-JEPA2.1 from scratch on that same group.
