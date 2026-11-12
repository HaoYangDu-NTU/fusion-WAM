# E00+：动作相关的 Clean Visual Residual Router

更新时间：2026-11-10。研究依据见 [方案与讨论记录](CAUSALWAM_M29_ROUTING_RESEARCH_PLAN.md)。

本轮匹配微调结果与真实 checkpoint 速度对照设计见 [2026-11-10 实验记录](E00_PLUS_EXPERIMENT_LOG_20261110.md)。FT-B0 标准 LIBERO 已完成：97.30%（1946/2000）；Initial B0 为96.95%（1939/2000）。

## 2026-11-10：已有 E00 checkpoint 的匹配微调试验

新增独立入口 [e00_plus_finetune.py](../scripts/e00_plus_finetune.py)，参数见
[e00_plus_finetune.yaml](../configs/e00_plus_finetune.yaml)。下面是 warm-start pilot，
不改变本页原 E00/E00+ 从头训练配置，也不将微调结果混入原固定预算的 Encoder 对比。

- 来源：`OpenWAM_v2/outputs/libero_vjepa_1_3b_dev_vx_ax_s42.0/2026-10-26_20-36-02/checkpoint_step_10690.safetensors`。
  已确认是 M29 + RynnBrain1.1-9B avg_all_layers + V-JEPA2.1，包含原 30 层主干、490 visual tokens。
  历史配置每卡 batch32；源训练 world size 未由此配置确认。本次三组都统一为四卡、每卡64。
- B0：原模型继续训练；B1：新增 Dense Residual Router；B2：同一个 Router + Top-k mass loss。
  三组均从同一 source 出发，使用新的 optimizer/scheduler，seed42、LR1e-5、1000 updates、global batch256。
  冻结原视觉 Encoder、保留原视频/动作损失；主干与 Router 正常更新。
- B1/B2 共享完全相同的 Router 初值。初始 gate0.01、k32、width512、8 heads。
  B2 的系数前100 updates 为0，随后400 updates线性增长至0.01；B1 始终为0。
  这样短程微调会实际经历稀疏正则阶段，不能直接沿用从头训练的1000/4000调度。
- 初始化脚本只追加9个 `action_router.*` tensors、2,624,001个参数；2151个共有 tensors 已逐项验证与源权重精确一致。
  B0 链接源权重；B1/B2 使用同一份只读初始化权重（硬链接），配置中的正则系数不同。
  不使用宽松 checkpoint 加载，不改变原 checkpoint/config，不需要下载新预训练权重。
- 初始化检查同时验证数据归一化和文本 cache 指纹；三个模型通过严格训练加载器和 CPU 有限值检查。
  原 B0 的旧环境评测仅供历史参考；队列会在当前代码/协议下重新评测初始 B0。
- 自动队列顺序：B0/B1/B2 各6步 GPU smoke（正式 batch64）→ 初始B0标准LIBERO →
  B0微调1000步+LIBERO → B1微调1000步+LIBERO → B2微调1000步+LIBERO。
  每次标准LIBERO为2000 trials，沿用10步去噪/执行10步/原每步传输路径；本轮不叠加客户端动作队列提速开关。
  LIBERO-Plus、router-disable和多训练种子复验在首轮结果后安排，不由此队列自动启动。

```bash
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
# 配置/命令预览，不训练。
"$PY" scripts/e00_plus_finetune.py
# 准备独立初始权重；再次执行会校验来源身份。
"$PY" scripts/e00_plus_finetune.py --prepare
CUDA_VISIBLE_DEVICES='' "$PY" scripts/e00_plus_finetune.py --validate-cpu
# 等待四张GPU空闲，顺序运行有限队列；不终止现有任务。
"$PY" scripts/e00_plus_finetune.py --run
```

本地初始化证据：`outputs/e00_plus_finetune/initialization/READY.json`、`CPU_VALIDATED.json`。
训练分别写入 `outputs/e00_plus_finetune/train/B0|B1|B2/<timestamp>/`；smoke独立保存，不用于正式续训。
队列状态、逐阶段命令/日志、完整标准评测和结果汇总位于
`reports/E00_plus/finetune_20261110/`（`status.json`、`RESULTS.md`、`results.json`）。
创建该目录下的 `STOP` 可阻止后续阶段；停止正在运行的训练使用该训练目录的 `STOP_TRAINING`。
已完成阶段不会重复运行；失败/中断阶段要求先检查，不会自动重训覆盖已有结果。
代码或配置在等待期间变化时队列会停止，避免不同源码混入同一对照。

后续判断依次看 `SR(FT-B0)-SR(initial B0)`、`SR(FT-B1)-SR(FT-B0)`、
`SR(FT-B2)-SR(FT-B1)`。1000步、单训练seed的首轮试验仅作筛选，不能代替原10690步从头训练或多seed结论。

## 实验与范围

| 实验 ID | 对应消融 | 内容 |
|---|---|---|
| E00 | B0 | 原 M29 + RynnBrain 9B + V-JEPA 2.1 |
| E00+-Dense | B1 | 同 baseline + residual visual readout，无稀疏正则 |
| E00+ | B2 | B1 + top-k attention mass loss |

`E00+` 及其对照继承 E00 的原生 tubelet、1 clean + 4 future、490 tokens、冻结 encoder、离线文本、30 层 M29、动作预测与训练预算。所有正式训练均从头初始化，共有模块的构造及 RNG 不被新 router 改变。B1/B2 的初始化与采样协议相同，只改变正则权重。

当前实现是 soft sparse readout，不裁剪主干 tokens，不宣称完成因果表征解耦或减少主干 FLOPs。Vision/text 联合路由、多层 HiRAE 融合及稀疏项单独 stop-gradient 不在本轮实现范围。

## 固定设置

- 位置：joint loop 之后、原线性 action decoder 之前。
- Q：最终 action hidden `[B,32,1024]`。
- K/V：最终 clean visual hidden `[B,98,1536]`，长度从实际 grid/clean metadata 推导。
- Router width 512，8 heads，每 head 64 维；k=32。
- 输入使用逐 token、无 affine 参数的 LayerNorm；QK 不另做单位范数归一化。
- 复用主干已有 view embedding、位置与任务条件；不额外添加关系偏置，不使用 attention dropout。
- 门控 `sigmoid(gate_logit)`，logit 初始化为 `logit(0.01)`；BF16 参数下实际值约 0.01001。
- FP32 logits 和 softmax；任务读出使用全部 softmax 权重，不做 hard top-k。
- E00+ 的最大稀疏系数 0.01；optimizer step <1000 时为 0，1000→5000 线性升满；E00+-Dense 始终为 0。
- 正则按有效 batch/head/action-query 平均；支持 `[B,T]` 和 `[B,T,D]` padding mask，后者至少一个有效动作维度才计入 query。
- 正则正常反传至 router Q/K 与 M29 主干；冻结 V-JEPA/离线 RynnBrain 不更新。
- 不为 router 设置独立 LR 倍率，采用标准 optimizer base LR。

损失为 `L_video + L_action + lambda(step) * mean(1 - topk_mass)`。1000 是已完成 optimizer updates 的数目；该步系数为 0，之后开始增长，5000 时为 0.01。恢复时从已恢复的 optimizer step 继续计算，不依赖易丢失的模块级 loss 缓存。

## 实现文件

- [action_router.py](../openwam/model/architectures/utils/action_router.py)：路由、质量损失、调度和诊断。
- [joint_self_attn.py](../openwam/model/architectures/dual_system/joint_self_attn.py)：末端接入，训练辅助量显式返回；默认推理仍返回 video/action 两项。
- [base.py](../openwam/model/architectures/base.py)：稀疏损失合并及 router device/dtype 分发。
- [openwam_trainer.py](../openwam/train/openwam_trainer.py)：optimizer step 传递、分布式指标、JSONL/W&B 日志。
- [E00+.yaml](../configs/experiments/E00+.yaml)、[E00+-Dense.yaml](../configs/experiments/E00+-Dense.yaml)：正式实验注册配置。
- [test_action_router.py](../tests/test_action_router.py)：数学、mask、梯度、初始化、重载与恢复验证。
- [smoke_e00_plus.py](../scripts/smoke_e00_plus.py)：真实资产与数据的端到端验收，可选择完整 M29。

Router 注册为可训练 top-level module，自动进入既有 optimizer groups，并包含在统一 checkpoint 中。E00 默认不创建 router 参数。旧 checkpoint 没有此配置时仍使用原路径；不能将旧 E00 权重直接严格加载为 E00+ 并当成同一实验恢复。

## 使用方式

以下命令在仓库根目录执行，Python 使用项目环境：

```bash
cd /data/fcy/code/openwam-exp
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python

# 仅解析并保存配置，不训练。
"$PY" scripts/experiment.py train 'E00+' --dry-run
"$PY" scripts/experiment.py train 'E00+-Dense' --dry-run

# 真实 encoder、真实 LIBERO 数据、缩小为两层 M29 的 CPU 验收。
CUDA_VISIBLE_DEVICES='' "$PY" scripts/smoke_e00_plus.py

# 完整 30 层 M29 的 CPU 验收；消耗较多内存和时间。
CUDA_VISIBLE_DEVICES='' "$PY" scripts/smoke_e00_plus.py --full-policy \
  --output reports/E00_plus/full_policy_cpu_acceptance.json

# GPU 空闲时，先做完整模型小 batch smoke；不是正式预算。
"$PY" scripts/experiment.py train 'E00+' --gpus 0 --batch-size 1 --smoke-steps 5

# 四卡正式训练，按 baseline 协议从头训练。
"$PY" scripts/experiment.py train 'E00+-Dense' --gpus 0,1,2,3 --seed 42
"$PY" scripts/experiment.py train 'E00+' --gpus 0,1,2,3 --seed 42

# 恢复时传入同一实验运行目录，沿用原身份校验和完整状态恢复。
"$PY" scripts/experiment.py train 'E00+' --gpus 0,1,2,3 --seed 42 \
  --resume /absolute/path/to/the/same/run
```

正式任务顺序运行，沿用项目 GPU 调度约束。原 E00+ run identity、配置和源码不匹配时不应强行恢复。当前其他项目正在使用四张 H100，未启动本实验的正式 GPU 训练。

训练输出分别位于 `outputs/m29_rynn9b_vjepa21_e00plus_sparse/` 与 `outputs/m29_rynn9b_vjepa21_e00plus_dense/`。完成后通过现有 `scripts/experiment.py eval` 执行标准 LIBERO 和完整 LIBERO-Plus，保持 baseline 的评测种子及动作执行协议。

## 日志与解释

训练目录新增 `router_metrics.jsonl`，包含 step、opt_step、loss 和 router 标量；对应指标也进入 W&B：

- `loss_router_sparse`：实际加权正则；`router_sparse_raw`：未加权正则；`router_sparse_weight`：本次 forward 实际系数。
- `router_topk_mass`、`router_entropy`、`router_effective_tokens`、`router_tokens90`。
- `router_topk_union_coverage`：所有有效 heads/queries 的固定 top-k 并集比例；近零权重也会被纳入，不能视为有效依赖范围。
- `router_view0_mass` / `router_view1_mass`：front/wrist，按真实 `t-h-view-w` 布局统计。
- `router_q_norm` / `router_k_norm`、`router_logits_std`：排查单纯放大 logits。
- `router_gate`、`router_residual_ratio`、`router_action_delta_rms`：hidden 与动作空间的直接贡献。

`router_action_delta_rms` 使用同一次 forward 的 `updated-before` 经 decoder 的线性权重映射，排除无效动作槽位，位于归一化动作预测空间；它不是反归一化后的物理动作差，也不是完整多步去噪关闭 router 的闭环影响。

默认不保存完整 attention tensor、不额外运行 autograd 梯度分解；噪声分桶、token 干预、完整采样关闭分支、梯度比探针可作为后续评估。直接 forward 支持 `use_action_router=False` 供分支消融使用；应用到完整采样时需要确保每个去噪步一致地设置。

## 验证记录与边界

验收产物保存在 [reports/E00_plus](../reports/E00_plus)。定向回归为 **135 passed、4 deselected**，JUnit 记录为 `tests.xml`。`acceptance.json` 是真实 V-JEPA、真实文本 cache、真实 LIBERO reader 下的缩小 M29 验收，检查 warmup/active sparsity 两次反传、冻结参数无梯度和生产部署加载器严格重载，重载预测差为 0；还通过 benchmark 的两步去噪采样，生成有限的 `[32,10]` 动作。它不提供闭环成功率，也不代表完成了正式 GPU 训练。

`full_policy_cpu_acceptance.json` 已通过：真实 encoder 与原 30 层策略完成两次反传、更新、生产加载器严格重载和单步预测一致性检查。仍不能代替四卡 ZeRO、每卡 batch64 的 GPU 吞吐/显存/恢复检查。缩小模型测试使用 mock encoder 的部分在 pytest 中单独覆盖，与真实 encoder 验收分开报告。

扩展回归中，旧 dev-launcher 测试依赖本机不存在的 `/workspace/code/OpenWAM`。这类环境路径失败不作为 E00+ 数值回归结论；本次定向回归使用 `-k 'not launcher'`，保留该限制记录。
