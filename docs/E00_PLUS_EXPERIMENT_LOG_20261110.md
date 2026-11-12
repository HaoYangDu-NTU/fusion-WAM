# E00+：匹配微调与评测速度实验记录

记录日期：2026-11-10，时区 Asia/Shanghai。结果快照截至 14:50 左右。

本记录汇总本轮实际执行的设计、结果和待完成项目。架构完整说明见 [E00+ 实现说明](E00_PLUS.md)，研究讨论见 [CausalWAM × M29 方案](CAUSALWAM_M29_ROUTING_RESEARCH_PLAN.md)。实时结果以报告目录中的 `status.json`、`RESULTS.md` 和原始 SQLite 为准；本文中的“进行中”是记录时状态。

## 1. 研究问题与基线

研究问题：在已有完整世界模型交互的基础上，为动作读出增加一条受控、动作条件化的视觉读取路径，能否改善闭环成功率；Top-k 稀疏正则能否进一步带来收益？

基线 E00 为 M29 + 冻结 V-JEPA2.1 + 离线 RynnBrain1.1-9B 文本特征（avg_all_layers），不是 RAM 5B 模型。M29 包含 30 层 Joint Attention：视觉侧 hidden 1536，动作侧 hidden 1024。视觉序列为 490 tokens，包括 98 个 clean tokens（两视角各 7×7）及 392 个 future tokens；动作 hidden 为 `[B,32,1024]`。

本轮先从已有 E00 checkpoint 做短程匹配微调，筛选改进方向；不启动三组 10,690-step 从头训练，也不把微调结果混入原固定预算 Encoder 对比。

## 2. Action Head 设计与消融

Router 位于 joint loop 之后、原 Action Decoder 之前：

```text
最终 Action Hidden ────────────────┐
        │ Q                       │
        └── Cross Attention ── g × residual ── 相加 ── 原 Action Decoder
最终 Clean Visual Hidden ── K/V ──┘
```

数学形式：

\[
H_a'=H_a+\sigma(\gamma)\,W_o\operatorname{Attention}(W_qH_a,W_kH_c,W_vH_c).
\]

- Q 来自 `[B,32,1024]` action hidden，K/V 来自 `[B,98,1536]` 最后一层 clean hidden。
- Router width 512，8 heads，head dimension 64；输入使用逐 token LayerNorm，logits/softmax 使用 FP32。
- 初始有效 gate 为 0.01，参数初始化 `gate_logit=logit(0.01)≈-4.595`。BF16 下实际约 0.01001。
- 保留原 30 层 dense Joint Attention 和全部 future prediction，不做硬 token pruning。
- 正则为对有效 batch/head/action-query 平均的 Top-k attention mass loss：

\[
\mathcal L=\mathcal L_{video}+\mathcal L_{action}
+\lambda(s)\operatorname{mean}_{b,h,i}\left(1-\sum_{j\in\operatorname{TopK}(W_{bhi,:},32)}W_{bhij}\right).
\]

| 组别 | 修改 | 稀疏系数 | 用途 |
|---|---|---|---|
| Initial B0 | 原 E00 checkpoint，不额外训练 | 0 | 当前代码与协议下重新测基线 |
| FT-B0 | 原模型继续微调 1000 updates | 0 | 分离额外训练本身的收益 |
| FT-B1 / Dense | 增加 residual visual router，微调 1000 updates | 0 | 验证额外视觉读取是否有效 |
| FT-B2 / Sparse | 同 B1 的结构和初值，微调 1000 updates | 最大 0.01 | 验证稀疏正则的增量贡献 |

关键比较是 `FT-B1 − FT-B0` 与 `FT-B2 − FT-B1`。FT-B0 的提升本身不能作为 Router 有效的证据。

这是 soft sparse readout：全部 98 个 K/V 仍参与计算；Top-k 正则只鼓励注意力集中。它不保证选中局部因果因素、不提供全局 token 预算，也不减少原主干 FLOPs。Final clean hidden 已经过深层混合，注意力热图不能直接当作局部像素归因。原 dense 路径也可能绕过 Router。

## 3. 微调配置、权重与复现入口

训练入口：[scripts/e00_plus_finetune.py](../scripts/e00_plus_finetune.py)。配置：[configs/e00_plus_finetune.yaml](../configs/e00_plus_finetune.yaml)。

| 配置 | 本轮设置 |
|---|---|
| 硬件 | 4 × H100 |
| 三组共同来源 | E00 `checkpoint_step_10690.safetensors` |
| 更新数 | 每组独立 1000 optimizer updates |
| Optimizer / Scheduler | 新建，不延续源 checkpoint 的训练状态 |
| Learning rate | 1e-5 |
| Batch | 每卡 64，global 256，gradient accumulation 1 |
| Seed | 42 |
| 精度 / ZeRO | BF16 / stage 1 |
| 训练范围 | 冻结视觉 Encoder，使用离线文本；主干及新增 Router 正常更新 |
| B2 调度 | 前 100 updates 系数为 0，随后 400 updates 线性升至 0.01 |
| 保存 | 每 500 updates，保留最后两份 checkpoint 与恢复状态 |

100/400 是短程微调调度；原从头训练 E00+ 的 1000/4000 调度保持不变。源历史配置每卡 batch32，但未由配置独立确认源 world size；不宣称两轮 global batch 相同。

来源完整路径：

```text
/data/fcy/code/OpenWAM_v2/outputs/libero_vjepa_1_3b_dev_vx_ax_s42.0/
2026-10-26_20-36-02/checkpoint_step_10690.safetensors
SHA256: 9279797d40dfe433ecc4298fd70a0c93ea93ec780dd478359e36944c949f9440
```

三组初始化位于 `outputs/e00_plus_finetune/initialization/B0|B1|B2/`。B1/B2 共享相同 Router 初值；只追加 9 个 tensors、2,624,001 个参数，2151 个共有 tensors 已验证与源权重精确一致。数据归一化与文本 cache 指纹已校验。无需下载新权重，原来源权重保留。

B0 微调最终权重：

```text
outputs/e00_plus_finetune/train/B0/2026-11-10_13-06-52/checkpoint_step_1000.safetensors
SHA256: 7b99a3b1ce3fd8d2c7d99192fce652486cfaf3e3301534bae72cf66692bc53a3
```

以下用于复现独立运行；当前后台队列已启动，不要重复执行 `--run`：

```bash
cd /data/fcy/code/openwam-exp
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
"$PY" scripts/e00_plus_finetune.py                 # 预览
"$PY" scripts/e00_plus_finetune.py --prepare       # 初始化与身份校验
CUDA_VISIBLE_DEVICES='' "$PY" scripts/e00_plus_finetune.py --validate-cpu
"$PY" scripts/e00_plus_finetune.py --run           # 有限队列
```

正式从头训练注册配置为 [E00+-Dense.yaml](../configs/experiments/E00+-Dense.yaml) 与 [E00+.yaml](../configs/experiments/E00+.yaml)，与本轮 warm-start pilot 分开。

## 4. 正式评测协议与已完成结果

本轮统一使用标准 LIBERO 2000 trials：Spatial/Object/Goal/Long 各 500。四卡模型副本、32 workers、batch 上限 8、GPU3 渲染、raw 图像；10 步去噪、执行 10 步。Initial B0、FT-B0、FT-B1、FT-B2 均沿用原每步传输路径，`client-action-queue=false`，不在不同模型之间混用新旧评测路径。当前队列不自动运行 LIBERO-Plus。

正式评测入口为 `python -m benchmarks.vla_harness.run_eval`。实际完整命令保存在各阶段 `command.json`，示例：

```bash
"$PY" -m benchmarks.vla_harness.run_eval --benchmark libero \
  --ckpt-dir outputs/e00_plus_finetune/train/B0/2026-11-10_13-06-52 \
  --ckpt-name checkpoint_step_1000.safetensors \
  --gpus 0,1,2,3 --workers 32 --max-batch-size 8 \
  --render gpu --render-gpus 3 --image-format raw \
  --output-dir /path/to/a/new/evaluation_directory
```

### 完整结果

| Suite | Initial B0 | FT-B0 | 变化（百分点） |
|---|---:|---:|---:|
| Spatial | 97.40%（487/500） | 97.00%（485/500） | −0.40 |
| Object | 100.00%（500/500） | 100.00%（500/500） | 0.00 |
| Goal | 96.20%（481/500） | 96.00%（480/500） | −0.20 |
| Long / libero_10 | 94.20%（471/500） | 96.20%（481/500） | +2.00 |
| **合计** | **96.95%（1939/2000）** | **97.30%（1946/2000）** | **+0.35** |

两次均为 2000 个有效 trials，运行错误为 0。FT-B0 评测于 13:50:45 开始、14:48:13 完成，用时约 **57 分 28 秒**，包括启动与收尾。B0 训练目录时间为 13:06:52，训练阶段于 13:50:42 完成，约 **43 分 50 秒**，包含该运行的初始化/保存开销。

结果解释：额外 1000 updates 后净增加 7 次成功，主要来自 Long（+10），Spatial 和 Goal 分别减少 2 和 1 次成功。只能说明本次单训练 seed 的短程继续训练有小幅收益；尚不能判断 Router 或稀疏正则是否有效，也未完成配对显著性检验或多训练 seed 复验。

| 项目 | 记录时状态 |
|---|---|
| 三组真实模型 GPU smoke，各 6 steps、每卡 batch64 | 已完成 |
| Initial B0 标准 LIBERO | 已完成，96.95% |
| FT-B0 1000 updates + 标准 LIBERO | 已完成，97.30% |
| FT-B1 训练与评测 | 排队，速度试验后恢复 |
| FT-B2 训练与评测 | 排在 FT-B1 后 |

原始证据：

- [队列结果汇总](../reports/E00_plus/finetune_20261110/RESULTS.md)
- [Initial B0 评测](../reports/E00_plus/finetune_20261110/03_initial_eval_B0/evaluation/RESULTS.md)
- [FT-B0 评测](../reports/E00_plus/finetune_20261110/05_eval_B0/evaluation/RESULTS.md)
- [队列实时状态](../reports/E00_plus/finetune_20261110/status.json)

## 5. RoboScope 借鉴改进：真实 checkpoint 速度收益实验

目标是测量本框架公共评测路径的实际收益，尤其是 client-action-queue 减少逐步通信的作用，以及 batch 上限与真实吞吐的关系。本次先使用真实 FT-B0 checkpoint；结论只覆盖该模型、history 接口、H100 与测得的并发配置，不能直接外推 VPP2 或其他 Encoder。

试验插在 FT-B0 评测完成后、FT-B1 训练之前。14:48 已开始 `00_oracle_legacy`，截至本文记录时尚无完整正确性或加速结果。不会将推荐参数自动混入原 B1/B2 对照。

| 阶段 | 设置与对照 | 判定目的 |
|---|---|---|
| 串行正确性，两次 | 单卡、1 worker、batch1；Spatial/Long 各 2 任务，最多 137 步；旧路径 vs queue | 逐次核对输入图像/历史/state/prompt 哈希、预测动作和实际执行动作；要求精确一致 |
| 四卡 ABBA，四次 | 相同 40 任务 trial0，最多 240 步；32 workers，batch 上限 8；旧→新→新→旧 | 只切换 queue，重复测量降低运行顺序影响 |
| 单卡 batch 初筛，六次 | 32 workers；queue 开启；上限 1/2/4/8/16/32 | 测单卡吞吐、实际 batch 和显存 |
| 四卡 batch 确认，三次 | 固定 40 workers；上限 4/8/16 | 测四卡并发瓶颈；每卡最多 10 个环境，上限16不代表实际16 |
| 最快两项反序复测，两次 | 四卡候选各再测一次 | 结合重复结果推荐参数 |

共 17 个有界 profiles；不额外启动完整 LIBERO/Plus。137 步正确性 profile 覆盖非整动作块截断。240 步 profiles 的成功情况不能当作正式 2000-trial 成功率。

记录的指标包括：

- 模型启动时间、含启动总时间、模型启动后端到端环境 steps/s。
- 实际请求数、实际 batch 分布、原生 `_predict_batch` 吞吐（每卡去除前两次调用）。
- GPU 利用率、设备显存与 PyTorch 峰值 allocated/reserved、CPU 状态。
- 同任务 episode 状态/长度差异；同时比较旧自身和新自身重复，以识别动态 batch 数值差异。

串行 oracle 通过后才做性能对照；发现 OOM、环境错误、外来 GPU 计算干扰或显存余量不足的候选不参与推荐。吞吐在最好值 5% 以内优先较小 batch。原始证据保留，不将模型推理提速、通信请求下降与完整评测加速混为一谈。

控制器只暂停原队列的下一阶段，不中断 B0。完成或失败后，在归属、源码指纹和资源检查允许时移除自己创建的 STOP，恢复 B1/B2。不会终止其他项目任务。运行中的生产源码保持不变，计时 wrapper 和控制脚本均放在报告目录。

入口与产物：

- [详细计划](../reports/runtime_benefit_20261110/PLAN.md)
- [控制器](../reports/runtime_benefit_20261110/controller.py)：等待、调度、分析与恢复原队列。
- [单项运行器](../reports/runtime_benefit_20261110/run_case.py)：读取 case 配置，支持 `--dry-run`。
- [实时状态](../reports/runtime_benefit_20261110/status.json)
- 报告目录：`reports/runtime_benefit_20261110/`，最终查看 `oracle.json`、`summary.json`、`RESULTS.md`。
- `cases/*/` 保存 case 配置、命令、日志、模型 batch 计时、telemetry 和原始评测 SQLite。

本轮运行脚本依赖固定报告目录与已有调度状态，不应在当前试验运行时重复启动。复现应使用独立报告目录、端口和输出目录。

## 6. 验证、后续更新与结论边界

初始化与微调定向测试：20 项通过，证据 `reports/E00_plus/finetune_tests_20261110.xml`。速度试验控制与分析检查：6 项通过，证据 `reports/runtime_benefit_20261110/tests.xml`。这些检查验证配置、身份、动作审计逻辑与调度保护；不能代替尚未完成的真实速度试验。

后续按以下顺序补充结果：

1. 真实 oracle 是否通过；ABBA 新旧路径端到端速度及重复波动。
2. 单卡/四卡 batch 探测表、显存与推荐值。
3. FT-B1 和 FT-B2 完整 LIBERO，与 FT-B0 的差异。
4. 若有可重复收益，再考虑 LIBERO-Plus、router-disable、选中 vs 随机 token 干预，以及多训练 seed 复验；这些不在当前有限队列中。

Router 训练还应结合 gate、残差贡献比例、Top-k mass、视角权重和注意力覆盖率判断是否被实际利用。稀疏损失可正常反传到共享主干；即使不直接改 future tokens，也不能声称预测分支在优化上完全独立。HiRAE 风格层级残差预算、多层 routing、单独阻断稀疏项输入梯度均保留为后续消融，本轮未实现。

当前已有结论仅为：**FT-B0 从 96.95% 提高到 97.30%；Router 增益、稀疏正则增益与 client-action-queue 的真实加速幅度仍待完成对照。**


## 7. 追加：B2 前的 LIBERO-Long 速度专项（16:13 安排）

用户要求在 B2 正式评测前验证 Long 提速。现已设置 B1 评测结束后暂停队列，运行 [Long 专项](../reports/runtime_long_20261110/PLAN.md)，结束或失败后按归属检查恢复 B2。使用 FT-B0 权重、Long 10任务各2trials、原生700步上限；四卡新旧 ABBA，单卡batch2/4/8/16，再四卡新旧各确认一次，另有两项短程真实正确性oracle，共12项。

上一轮速度试验在第二次旧路径收尾触发“外部GPU进程”标记，未记录触发PID，不能确定存在真实外部任务；保留无效状态。新试验改为session归属、消失PID忽略、连续两次外部观测确认，并保存PID/PPID/session/命令。9项定向测试已通过。

Long按整任务分片，10任务最多10workers，四卡每卡仅2–3路请求；batch上限增大不保证实际batch增大。专项profile不替代正式500-trial Long分数，不自动改变B2正式协议。实时进度见 [Long状态](../reports/runtime_long_20261110/status.json)。本节记录安排，不代表已获得速度收益。


## 8. 用户修订：随机子集验证 RoboScope 借鉴机制，B2 正式评测暂停

16:26左右更新，替代第7节安排；旧Long专项尚未启动case。B1评测后直接训练B2，再暂停其正式评测，使用新FT-B2权重运行随机Long子集。随机种子20261110，抽取task2/4/7；固定GPU0模型、GPU3渲染、3workers、batch上限4，3trials/task、700步上限。先在task2上做旧/新串行oracle（137步），再对整个子集做旧→新→新→旧ABBA，共6项，不再扫描batch。

目标是验证已接入的RoboScope式F1环境侧动作队列/历史管理是否带来实际加速，分解请求、模型调用、环境执行与初始化成本。F2完整并发搜索、F3 step_chunk、F4常驻环境池尚未接入，不当成已有收益。完成或失败均不自动恢复B2正式评测。最新 [计划](../reports/runtime_long_20261110/PLAN.md) 与 [状态](../reports/runtime_long_20261110/status.json) 为准。


## 9. FT-B1 最终结果（16:48 更新）

B1完整LIBERO评测已结束，2000个有效trials，0运行错误；B2于16:47:52开始训练，正式评测按用户要求保持训练后暂停，先执行随机子集RoboScope式F1速度试验。

| Suite | FT-B0 | FT-B1（Dense Router） | B1−B0（百分点） |
|---|---:|---:|---:|
| Spatial | 97.00% | 96.40% | −0.60 |
| Object | 100.00% | 99.80% | −0.20 |
| Goal | 96.00% | 95.00% | −1.00 |
| Long | 96.20% | 94.40% | −1.80 |
| **合计** | **97.30%（1946/2000）** | **96.40%（1928/2000）** | **−0.90** |

B1也比未微调的Initial B0（96.95%）低0.55个百分点。当前单训练seed短程微调结果未支持新增Dense Router带来收益；不能由此确定退化机制，稀疏正则增量效果仍待B2后续正式评测。证据：[B1完整结果](../reports/E00_plus/finetune_20261110/07_eval_B1/evaluation/RESULTS.md)。


## 10. 条件授权：加速有效则自动评测 B2

用户进一步授权在随机任务加速有效时，用加速方法评测B2。已新增后台条件启动器：真实oracle通过、重复无错误、两组新旧对照都更快，平均含启动总时间和启动后吞吐均改善至少2%才继续；该门槛是工程防噪声条件，不是统计显著性检验。

随机子集通过后先做四卡小规模ABBA部署确认（4suite各抽相同3个task IDs、每任务2trials），仍固定batch8，不做batch搜索。确认同样有收益后，自动评测完整LIBERO2000trials，4卡/32workers/batch8/raw/GPU3渲染与B0/B1相同，只开启client-action-queue；未通过则继续暂停。不会将单模型3workers直接用于全量。B2加速结果独立标记协议差异，完整评测仍用原生suite horizon，profile不作为正式分数。

入口与状态：[条件启动器](../reports/runtime_long_20261110/conditional_b2_eval.py)、[专项计划](../reports/runtime_long_20261110/PLAN.md)，conditional_status.json与acceleration_decision.json记录判断，conditional_B2_eval/evaluation保存正式结果。原队列STOP保留避免重复评测。进程审计新增嵌套session/父进程退出覆盖，14项定向检查通过。


## 11. 用户明确启动 B2 加速正式评测

在查看随机子集未达到重复提速门槛的结果后，用户明确要求使用加速路径评测B2。已启动独立完整LIBERO2000-trial任务：4卡、32workers、batch8、raw、GPU3渲染，开启client-action-queue。原随机试验与自动门槛判断保留，不据此宣称稳定提速；本次授权取代此前条件启动要求。配置/进程/状态与结果保存在reports/runtime_long_20261110/conditional_B2_eval/，无需额外下载权重。


## B2 accelerated 最终结果（2026-11-10T20:10:37）

完整LIBERO：1935/2000，96.75%，运行错误0。使用client-action-queue=true，按用户明确指示运行；与B0/B1的传输路径不同。原始结果：reports/runtime_long_20261110/conditional_B2_eval/evaluation/RESULTS.md。
