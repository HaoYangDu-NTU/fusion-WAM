# E01 prediction and video-shift search

## 当前追加队列（2026-10-31）

用户新增 P3：video/action = v/v。当前顺序为 P2 完整训练 → 标准 LIBERO 2000 trials → 记录结果 → P3 smoke 与推理检查 → P3 从头完整训练 → 标准 LIBERO 2000 trials → 记录结果 → 停止。S24/S60 暂未排队。下文两阶段设计保留为最初方案。

P3 配置：`configs/experiments/E01-P3-Cache-W2.yaml`。与 P2-Cache-W2 相比仅修改 action prediction 为 v，以及独立输出名称；沿用 DINO 原始特征 cache、每卡 4 workers/每 worker 2 线程、global batch256、10690 steps、seed42、video shift42、action shift5。P3 不从 P2 checkpoint 初始化。

队列：`reports/E01_P3_cache_w2_20261031`，通过 `--after-queue reports/E01_P2_cache_w2_20261031` 等待前序 DONE、complete 状态及完整评测验收。前序失败/停止则不启动；不会仅因 GPU 空闲就提前启动。P3 完成后不继续其他实验，不启动 LIBERO-Plus。

Authorized on 2026-10-30. Run sequentially after existing GPU workloads finish.
Standard LIBERO only; no LIBERO-Plus tasks or services are scheduled.

## Audited reference

Saved E01: `outputs/m29_rynn9b_dinov3/train/s42/2026-10-24_02-30-16`.
Final checkpoint is step10690. Prediction x/x, video shift42, action shift5,
logit_normal, action scale1, seed42, global batch256, 10 epochs. Current baseline
training/data settings match its saved config. Full standard LIBERO is complete:
1887/2000, 94.35%; suites 97.00 / 99.40 / 91.60 / 89.40 percent. Base is reused.

First stage: P1=x/v, P2=v/x, both video shift42. Each uses fresh seed42 initialization,
full original budget, then 40 tasks × 50 trials. Base is not retrained.

Second stage starts only after both complete results are accepted. It uses selected
prediction with video shift24 and shift60; shift42 aliases the selected first-stage
checkpoint and result. Action shift5 and all unrelated settings remain fixed.

Selection is fixed before results: pick highest Overall only if improvement over
Base exceeds both 1 percentage point and an approximate 95% unpaired Bernoulli
uncertainty threshold, and neither Goal nor Long drops more than 1pp. Otherwise
retain x/x. This is conservative single-seed screening, not a formal significance
claim (task/trial correlations and training-seed uncertainty remain).

## Existing mathematics

`z_sigma=(1-sigma)*x+sigma*epsilon`; velocity target is `epsilon-x`.
For x-prediction, prediction and target are both mapped to velocity space through
`(z_sigma-x)/max(sigma,0.05)`, then the original masked, timestep-weighted loss is
used. Inference uses the same conversion before the existing Euler step.

Video shift transforms `sigma` to `s*sigma/(1+(s-1)*sigma)` in both train and
inference schedules. Action shift remains5. No v/v or shift12 experiment is scheduled.

Optional future diagnostics reconstruct clean prediction from raw output, mask out
the clean prefix and padded future positions, and record MSE and channel cosine
similarity. They are detached and do not change losses/gradients. Logs are written
to each formal run's `ablation_metrics.jsonl`; the report averages the last100 unique
steps. These are single-noisy-input training diagnostics, not multi-step rollout metrics.
Base did not record those diagnostics, so missing Base values stay unavailable.

## Queue and recovery

```bash
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
"$PY" scripts/e01_ablation_queue.py --prepare
"$PY" scripts/e01_ablation_queue.py --run
```

Queue state: `reports/E01_ablation_20261030/status.json`.
Results: `reports/E01_ablation_20261030/RESULTS.md` and the table in `result.md`.
Each run performs four-GPU four-step training smoke with checkpoint saving,
finite-loss/gradient checks, and one standard-LIBERO smoke inference before full
training. Smoke results never enter the 2000-trial tables.

The controller waits for empty GPUs and free ports and uses the shared project GPU
lock. Independent stage workers preserve exit status across controller disconnects.
Restart with the same `--run` command to attach to live stages, reuse accepted
experiments, resume formal training from complete Accelerate state, and resume
evaluation using the existing deduplicating harness. Failures are retained and stop
the sequence; missing state or ambiguous runs require inspection. Code/config
identity is pinned once execution begins. `STOP` in the queue directory stops only
queue-owned subprocesses; remove it only when intentionally resuming.

Online E01 encoding is used initially. Current feature-cache CPU tests alone do not
certify production GPU numerical equivalence, so the search does not silently adopt
unverified cached features. No full cache generation is inserted ahead of P1.

## S24 启动（2026-10-31 晚）

P2、P3 均已完成标准 LIBERO，Overall 分别 92.25%、90.65%；baseline x/x 为 94.35%。用户要求固定 x/x 测试 video shift24。新配置 `E01-S24-xx-Cache-W2`，队列 `reports/E01_S24_xx_cache_w2_20261031`。从头训练，复用冻结 DINO cache 和 worker 2 线程，其他模型/训练设置沿用 baseline。先 smoke，再正式 10690 步和标准 LIBERO 2000 trials，结果写入 result.md。S42 复用 baseline；S60 未排队。此指令更新上方此前的等待安排。

## S60 追加队列（2026-10-31 晚）

用户追加授权：S24 训练与标准 LIBERO 完成，并写入 result.md 的 E01-S24 行后，自动启动 E01-S60。新配置 `E01-S60-xx-Cache-W2` 与 S24-W2 仅 video shift（24→60）及输出身份不同；全新初始化，非 S24 finetune。依赖队列 `reports/E01_S24_xx_cache_w2_20261031` 的 DONE/complete 与完整结果验收，再进行 smoke、正式训练、标准 LIBERO 2000 trials 和结果记录。S60 自身队列为 `reports/E01_S60_xx_cache_w2_20261031`，完成后停止。上文 S60 未排队的记录为此前安排。

## S12 追加队列（2026-11-01）

用户追加 S12，更新最初暂不测试 shift12 的范围。S60 训练、标准 LIBERO 和结果记录完成后，执行 `E01-S12-xx-Cache-W2` 的 smoke、从头训练、标准 LIBERO 2000 trials 和结果记录。与 S60 唯一模型设置差异为 video shift60→12；prediction 固定 x/x、action shift5，训练预算和缓存保持一致。队列 `reports/E01_S12_xx_cache_w2_20261101` 依赖 `reports/E01_S60_xx_cache_w2_20261031` 的 DONE/complete 与完整结果验收。失败或停止不会提前放行。S12 完成后停止，不运行 LIBERO-Plus。

## Video Loss 两组对照追加队列（2026-11-01 晚）

用户授权在当前 S12 训练和标准 LIBERO 评测结束后，顺序执行两组完整训练与标准 LIBERO。此安排更新上文“S12 完成后停止”：S12 自身控制器仍停止，两个独立依赖队列接续执行。

| 顺序 | Profile / 结果标签 | λ_video | λ_action | 前置队列 |
|---|---|---:|---:|---|
| 1 | E01-LV05-Cache-W2 / E01-LV05 | 0.5 | 1 | reports/E01_S12_xx_cache_w2_20261101 |
| 2 | E01-LV2-Cache-W2 / E01-LV2 | 2 | 1 | reports/E01_LV05_cache_w2_20261101 |

第二组队列为 `reports/E01_LV2_cache_w2_20261101`。两组均固定 x/x、video shift42、action shift5、logit_normal、action scale1；不是继承 S12 的 shift12，也不因 S12 结果自动更换参考配置。冻结 DINO/RynnBrain，复用 `cache/dinov3_libero_frames`，4 workers/rank、2 CPU threads/worker。其余视觉接口、980 tokens、bottleneck256、模型、优化器、学习率、global batch256、seed42、10690步预算不变。均从原始 scratch 初始化完整训练，不继承前一组训练权重；仅故障恢复时使用该组自己的完整状态。

复用现有队列的四卡4步训练 smoke、checkpoint reload + 实际 LIBERO rollout 检查、有限 loss/有效梯度门控，通过后正式训练。每组完整评测标准 LIBERO 40 tasks × 50 trials，验收四suite各500条有效记录、零基础设施错误，再写 `result.md` 的独立 Video Loss 表和队列 `VIDEO_LOSS_RESULTS.md`。λ_video=1复用 E01-Base，不重复训练。日志 loss_video 已乘权重，跨组比较未加权损失需除以对应 λ_video；future MSE/cosine 是独立诊断。只执行标准 LIBERO，不运行 Plus。第二组完成后停止。

依赖队列失败/STOP不会放行下一组；独立输出、GPU共享锁、结果去重和完整状态断点续训沿用原机制。排队期间不改动当前 S12 的配置、进程或 checkpoint。新增配置及报告代码会改变项目 source fingerprint；已运行 S12 控制器继续使用其已加载逻辑，若以后需要重启旧队列，应先审计源码变化，不能覆盖其历史 source pin。
