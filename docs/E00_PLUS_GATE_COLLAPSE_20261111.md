# E00+ Gate 数值精度与 Attention Collapse 专项诊断

更新时间：2026-11-11T15:06:26。原始训练/checkpoint/生产实现未修改。该文补充 [总诊断报告](E00_PLUS_DIAGNOSIS_20261111.md)。

## 结论

1. **已确认 Gate 的前向有效值受 BF16 量化限制。** Gate 在优化器中，有非零梯度；ZeRO1 保存的 FP32 主参数在训练中发生变化，但变化不足以改变 BF16 模型参数。不能再把现象描述为“优化器完全没有更新 Gate”。
2. **B2 有明显的 Query 选择同质化，部分 Head 偏好固定位置。** 不是所有 Head 在所有任务上都选择相同位置；没有证据将全部 Router 判为完全塌缩或将高权重位置判为因果相关。
3. 目前没有启动新的训练、没有改成 FP32 Gate、没有扩大到全量 Plus。下面的真实数据梯度测量及标量 optimizer 数值回放全部在 CPU 完成，不新增 GPU 任务。原先获授权的有限 Plus pilot 继续执行。

## 1. Gate 检查链路

| 检查 | 证据与结果 |
|---|---|
| 原参数 dtype | B1/B2 `action_router.gate_logit` 均为 BF16 |
| Optimizer 收录 | 实际模型按原 freeze 配置构造 `build_trainable_parameters`，Gate 均出现恰好一次，requires_grad=True；历史 ZeRO param_slice_mappings 同时确认 |
| 历史更新证据 | step500/1000 optimizer 的 FP32 主参数变化、exp_avg/exp_avg_sq 非零 |
| 当前真实梯度 | 两个最终 checkpoint 各 8 个真实 LIBERO 任务样本×3组固定噪声，24/24 次 Gate 梯度非零 |
| 每步数值变化 | 保存每次标量 optimizer.step 前后 master/BF16 值；峰值LR下 master 12/12步变化，BF16 0/12步变化；末尾LR下 master/BF16均0/12步变化 |
| 证据边界 | 未记录原训练1000步的每步master/gradient，不能恢复逐步历史；新逐步回放是隔离数值实验，不是原global batch256训练的重演 |

### 历史 optimizer 的直接证据

通过 checkpoint 内 `param_slice_mappings` 定位 Gate，均在 rank3/group0、offset477072398、numel1。读取对应 FP32 主参数及 AdamW moments，而不是根据参数顺序猜位置。

| 模型 | step | FP32 master logit | BF16 模型 logit | Adam exp_avg | Adam exp_avg_sq |
|---|---:|---:|---:|---:|---:|
| B1 | 500 | -4.594245433807 | -4.59375 | -7.792264e-08 | 1.046119e-11 |
| B1 | 1000 | -4.594278335571 | -4.59375 | -6.66608e-07 | 4.387855e-12 |
| B2 | 500 | -4.594311714172 | -4.59375 | 1.018127e-06 | 1.703882e-11 |
| B2 | 1000 | -4.594357013702 | -4.59375 | -4.830071e-07 | 6.652189e-12 |

初始化的模型 Gate 为−4.59375；ZeRO1从低精度参数构建FP32 master。最终 B1/B2 master相对初值分别为−0.000528335571/−0.000607013702，变化方向也不是放大Gate。BF16在该数值附近ULP=0.03125，半ULP=0.015625，累计变化仍远小于模型舍入边界。原1000条训练日志的Gate指标均不变，与此一致。

实际使用的是 **FP32 master累积更新、再复制到BF16模型副本**。因此“每次BF16更新直接被丢弃，完全不能累积”不是这次ZeRO1的准确机制。

![Gate master 与模型副本](../reports/E00_plus/diagnosis_20261111/gate_precision/gate_master_vs_model.png)

### 当前真实梯度和逐步回放

从固定等间隔候选训练索引中保留前8个不同prompt，清单在两模型forward前落盘。每样本3组噪声种子一致。调用原 `prepare_inputs` 和 `compute_loss`，以真实监督动作和未来视觉数据计算loss；仅请求Gate的导数，其他权重在该诊断进程暂时冻结以缩小计算图。这不改变Gate的偏导数路径；CPU BF16数值不应被宣称逐位等价于CUDA训练。所有真实模型参数只读，没有 optimizer.step。

| 模型 | 非零梯度 | 最小梯度 | 最大梯度 |
|---|---:|---:|---:|
| B1 | 24/24 | -2.348423e-05 | 5.364418e-05 |
| B2 | 24/24 | -4.9829483e-05 | 0.00012493134 |

另建立一次性FP32标量副本，恢复最终Gate的AdamW moments与step=1000，用前6次真实单样本梯度回放。分别使用原末尾LR=1e−7和原峰值LR=1e−5，betas=(0.9,0.95)、weight_decay=0.01、eps=1e−8。原生产模型不接收这些更新。未重演global batch256、四卡梯度平均、全模型gradient clipping和scheduler，所以仅验证存储/舍入行为，不估计真实训练收益。

峰值LR下，两模型的master每步变化约1.4e−6至5.7e−6，BF16副本一直不变。末尾LR下，回放中的FP32 master也不变：−4.59附近FP32的ULP约4.77e−7，足够小的更新也会低于其舍入分辨率。不能据此断言原最后每一步都完全不动。逐步记录见 `scalar_replay.jsonl`。

### 还有一次 Gate 乘法前量化

原实现 `gate_logit.float().sigmoid()` 之后执行 `gate.to(actions.dtype)`。初始值与两组最终FP32 master分别计算sigmoid再转BF16，得到相同乘数 **0.010009765625**。因此即使保留FP32 Gate参数，在当前变化量下，只改参数dtype仍无法改变前向乘数。

未来短程验证应分别观测：FP32原参数、sigmoid输出、乘法实际使用的gate、残差及最终动作。候选最小修复是FP32保存Gate，并让标量乘法/残差加法在FP32完成后再按接口转回隐藏dtype；这会改变舍入路径，必须单独控制初始输出差异。**本次没有实施该修复或重新训练。** 也没有证据证明修复精度后成功率一定提升。

## 2. B2 Attention Collapse 检查

仍是一个末端Router、8heads、32queries、98tokens，soft attention。此处使用上述8个真实训练样本×3组固定噪声（每模型24次forward），B1/B2同样本同seed。它补充原4条闭环轨迹的460次forward统计；两批采样分布不同，数字不能直接混作一个评测成绩。

Query重叠在每次forward、每个head内部计算全部496对不同query；Jaccard=交集/并集，Overlap=交集/32。跨任务统计先对每任务3组噪声及32queries平均成token概率，再比较各任务的Top8集合及Jensen–Shannon divergence（JSD，nats）。同时记录JSD，避免把高重叠集合误当成完全相同的概率分布。

| 指标 | B1 | B2 |
|---|---:|---:|
| Top32 mass | 0.411704 | 0.999914 |
| Entropy（nats） | 4.555288 | 2.236377 |
| Query Top32交集/32 | 0.561873 | 0.921829 |
| Query Top32 Jaccard | 0.425752 | 0.901310 |
| Query Top1一致率 | 0.158172 | 0.773679 |
| Query JSD均值 | 0.006814 | 0.043470 |
| 跨任务Top8 Jaccard | 0.065813 | 0.421924 |
| 跨任务JSD均值 | 0.006430 | 0.282802 |

B2 Query JSD均值大于B1，但中位数仅0.000827（B1为0.004441），少数跨视角切换使均值变大。因此不能只看均值下结论，也不能说所有query分布比B1更相似。B1本来接近均匀，低Top-k重叠也不自动代表有效动作分工。随机独立Top32集合的期望交集/32为32/98≈0.3265，只是参照，不是实际模型应达到的目标。

### 每Head结果（编号0–7）

| Head | B1熵 | B2熵 | B1 Query Jaccard | B2 Query Jaccard | B2 Head Camera | B2 Wrist Camera | B2底行质量 | B2跨任务Top8 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.5607 | 1.9236 | 0.4006 | 0.9482 | 0.000000 | 1.000000 | 0.9982 | 0.8175 |
| 1 | 4.5607 | 2.1048 | 0.3840 | 0.7955 | 0.327982 | 0.672018 | 0.8948 | 0.4381 |
| 2 | 4.5563 | 2.5547 | 0.4570 | 0.9806 | 0.999999 | 0.000001 | 0.6466 | 0.5171 |
| 3 | 4.5611 | 1.9990 | 0.3583 | 0.9882 | 0.000000 | 1.000000 | 0.0000 | 0.1422 |
| 4 | 4.5550 | 2.4061 | 0.4357 | 0.9907 | 0.999999 | 0.000001 | 0.6355 | 0.5138 |
| 5 | 4.5489 | 2.4392 | 0.4726 | 0.8524 | 0.674341 | 0.325659 | 0.4124 | 0.2297 |
| 6 | 4.5391 | 2.3596 | 0.4809 | 0.8313 | 0.671626 | 0.328374 | 0.7717 | 0.2977 |
| 7 | 4.5605 | 2.1039 | 0.4169 | 0.8236 | 0.326926 | 0.673074 | 0.9198 | 0.4192 |

视角索引按真实t-h-view-w布局解析：index=(h×2+view)×7+w，98个tokens，末14个tokens对应两个视角最底行。Head0的高权重tokens主要为91–97（腕部视角底行），底行attention mass约99.8%，跨8个任务Top8 Jaccard约0.817。Head2/4偏Head Camera，Head0/3偏Wrist Camera；多个head反复使用底行，但Head3跨任务位置变化更明显，不能称整个模块都是固定常量路由。

![Head指标](../reports/E00_plus/diagnosis_20261111/gate_precision/head_collapse_metrics.png)

![B2跨任务Token概率](../reports/E00_plus/diagnosis_20261111/gate_precision/task_token_profiles_B2.png)

![B1跨任务Token概率](../reports/E00_plus/diagnosis_20261111/gate_precision/task_token_profiles_B1.png)

**机制判断：** 有证据支持“B2部分head形成固定空间偏好、同一head不同动作query的选集高度重复”。“为了满足稀疏loss而形成快捷选择/attention sink”是待检验解释；底行可能包含夹爪或已经在30层Transformer中聚合的信息，未做对象标注、空间干预或局部token删除，不能认定这些都是无关背景。跨任务仍有变化，且原ON/OFF有非零动作差异，暂不符合整个分支几乎完全未参与的停止条件。

## 3. 后续决策

- 优先级1已完成数值诊断；下一步若安排短程实验，应先验证FP32 Gate和实际乘法路径的可更新性，并与现有BF16路径同起点对照。不要直接延长原配置训练。
- 优先级2已有定量证据。后续更有区分力的是固定观测改变action/noise/text，以及底行/选中token与随机token干预，观察动作敏感性。不要直接增加“多样性loss”来追求好看的热图。
- 继续已授权的每模型600条Plus pilot可以检验有限扰动下的表现；不扩大到10030条，不据此宣称统计显著。Gate量化和Attention同质化均不能单独解释B0/B1/B2正式分数差异。

## 4. 复现与证据

所有新增脚本、原始数组、日志均在 `reports/E00_plus/diagnosis_20261111/gate_precision/`。`optimizer_evidence.json` 保存历史checkpoint路径和精确映射，`runtime_membership.jsonl`记录optimizer收录，`gradients.jsonl`记录48次真实梯度，`scalar_replay.jsonl`记录24次标量step，`attention_B1.npz`/`attention_B2.npz`保留完整注意力概率，`sample_selection.json`保存8个索引及文本，`summary.json`保存统计。

```bash
cd /data/fcy/code/openwam-exp
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
D=reports/E00_plus/diagnosis_20261111/gate_precision
# 只读重聚合现有数组与报告，无需加载大模型
OPENBLAS_NUM_THREADS=1 "$PY" "$D/analyze_precision.py"
"$PY" "$D/write_precision_report.py"
"$PY" -m pytest "$D/test_evidence.py" -q
# 原始采集命令（已运行，不要向同一目录重复追加日志）
# OPENBLAS_NUM_THREADS=1 "$PY" -u "$D/inspect_states.py" > "$D/inspect.log" 2>&1
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$PY" -u "$D/probe.py" > "$D/probe.log" 2>&1
```

3项检查通过：统计边界与布尔分位数、原始概率归一化/任务种子/集合重叠独立复核、历史master量化及非零moments。采集时无权重文件写入；标量回放只产生JSON日志，没有可部署的新checkpoint。
