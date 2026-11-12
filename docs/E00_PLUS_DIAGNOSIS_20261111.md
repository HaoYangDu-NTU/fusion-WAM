# E00+ 公平性、Router 机制与 LIBERO-Plus pilot 诊断

日期：2026-11-11；本报告由只读 checkpoint 诊断、已有审计证据与有限评测生成。更新时间：2026-11-11T15:40:24。

## 1. 已验证的主要结论

- B1/B2 的 Gate 初值和最终值逐位相同，均为 `sigmoid(-4.59375)=0.0100135701`；1000条训练日志亦全部恒定。没有学会增大 Gate，但这不能推断 Gate 没有梯度。
- B2 的稀疏正则确实生效：Top-32 mass 从 B1 约0.412提高至约0.999896，entropy显著下降；这是soft attention集中，不是删除tokens或因果证据。
- 分支有可测但较小的贡献，不能称为完全失效：隐藏残差约0.36%/0.50%，完整采样后可执行动作ON/OFF相对L2差异约0.42%/0.52%。
- 4任务有界闭环中，B1/B2 的ON/OFF均为3/4成功，没有显示关闭分支导致成功率下降；Long均在137步人为上限停止，不能归因模型失效。
- 原始标准LIBERO中B0仍最高，B2与B1的0.35pp差异不能当成Router优于基线或统计显著性证据。

## 2. Legacy / client-action-queue 公平性

复用当前B2真实权重的已完成审计，未重复运行相同正确性轨迹。来源：`reports/runtime_long_20261110/oracle.json`，Long task2，137个控制步、14次完整10步去噪预测，动作最大绝对差为0。对照源码四个文件execution/client/chunk_server/worker哈希与当前文件全部一致；审计使用的B2 checkpoint路径与正式B2完全相同。

| 检查对象 | 证据与边界 |
|---|---|
| 输入图像与历史帧 | 每次预测的images/history_images/clean_condition_images按shape、dtype与像素内容计算SHA；两路径精确相同 |
| 机器人状态与文本 | state、prompt参与同一哈希；同时检查prompt字符串 |
| 预测动作 | 比较模型服务返回的可执行LIBERO7D动作块，逐元素精确相等；不声称审计覆盖未消费的后续22个native动作槽 |
| 实际消费动作 | SQLite逐控制步action按task/episode/step匹配，137步最大差0；137不是10的整数倍，覆盖块中截断 |
| chunk与history逻辑 | queue在worker逐步更新独立图像历史，只在队列耗尽时请求；legacy在服务端逐步维护历史；均执行原10步后再规划，reset/end清空缓存 |
| 补充边界测试 | 既有运行时158项针对性测试覆盖early done、历史可变buffer、episode token、重置与多会话；另有FT-B0串行525步/54次预测零误差 |

**结论与限制：** 在该真实B2串行轨迹和已检查接口语义上等价，未发现输入/动作消费语义不一致；满足复用现有严格小规模检查的条件，因此没有重复该测试，也未无证据重跑2000条B2。它不是所有观测和并行动态batch下逐位一致的数学证明。既有ABBA中同路径重复也会出现episode步数/结果变化，跨路径变化不能单独诊断为传输不等价。原始三组正式结果仍明确保留路径差异，不能宣称已在同一传输路径重测全量三组。

本轮新增ON/OFF和Plus pilot全部统一legacy、同一批任务与相同推理配置，形成协议对齐的小规模对照。若未来发现同一固定观测/固定batch下不一致，应在新目录重测B2 legacy 2000条；本次没有触发该条件。证据索引：[fairness_evidence.json](../reports/E00_plus/diagnosis_20261111/fairness_evidence.json)。

## 3. 原标准LIBERO结果与训练公平性

| 模型 | 成功/2000 | 成功率 | 路径 | 相对FT-B0 |
|---|---:|---:|---|---:|
| Initial E00 | 1939 | 96.95% | legacy | −0.35pp |
| FT-B0 | 1946 | 97.30% | legacy | 0 |
| FT-B1 Dense Router | 1928 | 96.40% | legacy | −0.90pp |
| FT-B2 Sparse Router | 1935 | 96.75% | queue | −0.55pp |

三组都从同一个E00 step10690继续1000 updates、新建optimizer/scheduler、seed42、LR1e-5、global batch256、BF16/ZeRO1，冻结V-JEPA，RynnBrain9B使用相同离线cache。B1/B2 Router初始化相同；B1稀疏系数0，B2前100步0、随后400步升至0.01。完整配置提取在[models.json](../reports/E00_plus/diagnosis_20261111/models.json)。所有原始报告、checkpoint和训练配置均未覆盖或改写。

## 4. Gate 与 Residual Ratio

| 模型 | 初始logit | 最终logit | 初始g | 最终g | 存储dtype |
|---|---:|---:|---:|---:|---|
| B1 | -4.59375 | -4.59375 | 0.0100135701 | 0.0100135701 | BF16 |
| B2 | -4.59375 | -4.59375 | 0.0100135701 | 0.0100135701 | BF16 |

在logit≈−4.594附近，BF16相邻可表示值间距为0.03125。小LR、短预算与低精度存储可能使更新未达到可表示变化；`.float().sigmoid()`只提高计算精度，不改变参数存储精度。后续专项诊断已直接读取ZeRO1 master：B1/B2最终分别为−4.594278335571/−4.594357013702，有非零Adam moments；真实样本Gate梯度各24/24非零，master更新未跨BF16舍入边界。生产forward还会将sigmoid结果转BF16，因此仅改参数dtype不够。详见[Gate精度与Collapse专项报告](E00_PLUS_GATE_COLLAPSE_20261111.md)。训练并未遗失整个Router：B1/B2的Q/K/V/out权重都有变化。

下面主表使用用户公式对每个sample的整个`[32,1024]`隐藏张量求L2比值（与Frobenius范数等价），每次去噪forward一个值；每模型46个预测chunk×10次forward，共460个值。`raw`为g×RouterOutput，`effective`为实际BF16加法后的H′−H。

| 模型/残差口径 | Mean | Median | P90 | P99 |
|---|---:|---:|---:|---:|
| B1 raw | 0.003576 | 0.003554 | 0.004089 | 0.004519 |
| B1 effective | 0.003642 | 0.003626 | 0.004150 | 0.004575 |
| B2 raw | 0.005030 | 0.004984 | 0.005724 | 0.006330 |
| B2 effective | 0.005115 | 0.005084 | 0.005820 | 0.006405 |

另保存按query求比值的14720条分布：B1 raw mean0.003598/P990.004757；B2 raw mean0.005062/P990.006694。每个样本内query、head与去噪步高度相关，这些分位数是描述性统计，不能当独立样本置信区间。

B1约29.95%、B2约23.48%的hidden坐标具有非零拟议残差，但BF16加法后该坐标仍保持原值；整体effective残差仍非零，且舍入也可放大部分坐标变化，不能用这个比例直接推算动作丢失比例。

![Gate](../reports/E00_plus/diagnosis_20261111/figures/gate.png)

![Residual/Attention distributions](../reports/E00_plus/diagnosis_20261111/figures/distributions.png)

## 5. Top-32、Entropy、各Head/Query与视角

实际只有**1个**末端Router，8 heads、32 action queries、98 clean K/V；没有30层Router。每模型117760条head-query-forward注意力行。

| 模型 | Top32 Mean | Median | P90 | P99 | Entropy均值(nats) | Head/Front mass | Wrist mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.412134 | 0.403305 | 0.461657 | 0.513269 | 4.555555 | 0.516548 | 0.483452 |
| B2 | 0.999896 | 0.999992 | 1.000000 | 1.000000 | 2.185959 | 0.500162 | 0.499838 |

视角ID复用模型forward实际传给Router的view_ids，遵循t-h-view-w交错布局，不误用前49/后49。B2全局两视角约各50%，但单head/query常接近单一视角；全局均衡不能排除局部极端集中。

| Head | B1 Top32 | B2 Top32 | B1 Front | B2 Front |
|---|---:|---:|---:|---:|
| 0 | 0.417918 | 0.999988 | 0.4691 | 0.0000 |
| 1 | 0.401382 | 0.999912 | 0.4847 | 0.4924 |
| 2 | 0.411630 | 0.999959 | 0.5386 | 1.0000 |
| 3 | 0.415041 | 0.999746 | 0.4669 | 0.0001 |
| 4 | 0.412758 | 0.999979 | 0.5669 | 1.0000 |
| 5 | 0.417178 | 0.999757 | 0.5648 | 0.5084 |
| 6 | 0.421263 | 0.999916 | 0.5366 | 0.5084 |
| 7 | 0.399905 | 0.999920 | 0.5047 | 0.4919 |

每个Head×Query单元的mean/median/P90/P99保存在`attention_cells_B1.npz`、`attention_cells_B2.npz`；每次预测的原始数组保存在`cases/B1_on|B2_on/router_samples/*.npz`。完整逐head、逐query分布在mechanism_summary.json。FP32求和会出现约1e−7的越界误差，不解释为概率超过1的机制问题。

B2训练末Q/K norm约8.41/9.19，B1约4.59/4.58，且B2 Q/K权重相对初值的RMS变化约0.00193，显著高于B1约0.000074–0.000078。**推测：** 一部分集中可以通过增大logits实现；仅凭mass不能判定学到了更有价值的空间选择。

![Training curves](../reports/E00_plus/diagnosis_20261111/figures/training_curves.png)

![Head/query and cameras](../reports/E00_plus/diagnosis_20261111/figures/head_query_and_views.png)

## 6. Router ON/OFF：动作输出与有界闭环

只通过报告目录中的architecture.forward包装传递`use_action_router=False`，在每一次去噪forward关闭分支，直接令ActionHidden′=ActionHidden。没有改权重、没有detach或重训；eval模式、固定每请求seed42、相同观测与history、同一32位置预测/前10步执行。

每模型在4个ON轨迹的前2次预测上各做一次OFF完整采样，共8对完全相同输入的动作块。以下MSE是**实际可执行LIBERO7D、10步chunk**的数值差，维度混合，不是归一化80维loss、物理位置误差或全轨迹MSE。

| 模型 | 配对chunks | Mean Action MSE | Mean相对L2差 | Mean范数差(ON−OFF) | Mean max-abs差 |
|---|---:|---:|---:|---:|---:|
| B1 | 8 | 3.57267286e-06 | 0.004239 | -0.002281 | 0.006291 |
| B2 | 8 | 5.26204232e-06 | 0.005224 | -0.000526 | 0.007735 |

分量相对L2差：B1 translation约0.535%、rotation约3.55%；B2 translation约0.665%、rotation约4.53%。Rotation原动作范数较小，百分比会被放大，不等价于角度误差。数值贡献较小但可测，不能判为零贡献，因此允许进入固定200/category pilot；该允许判断不是认定Router有性能收益。

| 模型 | Router ON | Router OFF | Long结果 |
|---|---:|---:|---|
| B1 | 3/4 | 3/4 | 都在137步上限未完成 |
| B2 | 3/4 | 3/4 | 都在137步上限未完成 |

固定每个suite task2、trial0、原生seed与初始状态，单卡batch1/worker1，GPU0同时渲染，137步上限；所有4组0运行错误。它是有界smoke消融，不是正式LIBERO分数。成功的3个任务完成步数有1–5步变化，未见成功状态改变；Long任务正常完成可能超过137步，这个timeout不能证明Router引起失败。

关闭Router不会回到B0：最终Action Decoder中B1/B2分别有1537/4043个权重元素与FT-B0不同（共81920），说明至少动作主路径也已变化；共享M29还可能受到action与sparse梯度影响。该参数差是事实，不能仅凭差异确定哪个更新导致成功率变化。

## 7. LIBERO-Plus 固定 pilot

只测试Camera Viewpoints、Light Conditions、Background Textures。每类200条、每suite每类50条；每模型共600条，三模型1800条。随机种子20261111，按原基础任务分组轮转抽样；112个category×suite×base组各5条，另8组为4或6条，以满足suite精确50。清单在任何pilot结果产生前落盘，三组共享同一600条。

统一legacy、4卡/32workers/batch8/raw/GPU3渲染、corrected text=true、官方seed10000、官方初始状态与settle30、原生600/700步上限。离线RynnBrain cache已预检查，40条唯一corrected prompts全部命中，不下载新权重。不使用smoke，不把模型失败提前截短；完整性只针对选定600条，不是10030条Plus完整成绩。

| 扰动类别 | B0 | B1 | B1−B0 | B2 | B2−B0 |
|---|---:|---:|---:|---:|---:|
| Camera Viewpoints | 132/200 (66.00%) | 139/200 (69.50%) | +3.50pp | 132/200 (66.00%) | +0.00pp |
| Light Conditions | 193/200 (96.50%) | 193/200 (96.50%) | +0.00pp | 195/200 (97.50%) | +1.00pp |
| Background Textures | 186/200 (93.00%) | 181/200 (90.50%) | -2.50pp | 184/200 (92.00%) | -1.00pp |

三组固定pilot均已完成；这是单checkpoint、有限扰动样本的描述性比较，不作统计显著性声明。错误、失败明细与任务分组均保留。

![Plus pilot](../reports/E00_plus/diagnosis_20261111/figures/plus_pilot.png)

**B0 pilot失败案例（按预先清单顺序取前3条，无事后挑选）：**
- Camera Viewpoints / libero_spatial / task739，fail，600步：`pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_145_0_0_initstate_0`。
- Camera Viewpoints / libero_spatial / task849，fail，600步：`pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_303_15_100_0_0_initstate_0`。
- Camera Viewpoints / libero_spatial / task862，fail，600步：`pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_310_15_100_0_0_initstate_0`。

**B1 pilot失败案例（按预先清单顺序取前3条，无事后挑选）：**
- Camera Viewpoints / libero_spatial / task680，fail，600步：`pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_42_0_100_0_0_initstate_0`。
- Camera Viewpoints / libero_spatial / task739，fail，600步：`pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_145_0_0_initstate_0`。
- Camera Viewpoints / libero_spatial / task849，fail，600步：`pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_303_15_100_0_0_initstate_0`。

**B2 pilot失败案例（按预先清单顺序取前3条，无事后挑选）：**
- Camera Viewpoints / libero_spatial / task849，fail，600步：`pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_303_15_100_0_0_initstate_0`。
- Camera Viewpoints / libero_spatial / task862，fail，600步：`pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_310_15_100_0_0_initstate_0`。
- Camera Viewpoints / libero_spatial / task830，fail，600步：`pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_162_0_0_initstate_0`。

## 8. 判断与后续建议

**目前不自动启动完整Plus，也不重新训练。** 原标准LIBERO不支持Router优于FT-B0；小样本ON/OFF没有成功率增益，Gate恒定且残差较小。先结合本次固定pilot判断是否有特定扰动收益，不能仅凭接近1的Top32 mass扩大评测。

后续结构/优化候选（仅建议，本次未改）：验证FP32 Gate存储及残差乘法路径（历史master更新已查明）、为新Router单独设置可验证的LR/warmup、比较稀疏项与action项对Q/K的梯度、避免只靠logit放大降低稀疏损失、比较更早层或投影后的clean K/V。当前证据不足以优先延长同一配置训练；应先确定Gate可更新、残差预算和动作贡献再安排新的训练消融。

**失败原因边界：** 原B1/B2标准LIBERO退化可能来自新增读出或稀疏项对共享主干的影响，但尚未因果定位；Gate前向量化限制已由历史master与独立数值回放确认，但尚不能归因为分数退化的原因。这里没有目标mask、token干预或背景反事实配对，因此不把attention解释为因果重要性。

## 9. 可复现入口与原始证据

本轮只新增`reports/E00_plus/diagnosis_20261111/`与本报告，不修改生产架构/原checkpoint/已有报告。

```bash
cd /data/fcy/code/openwam-exp
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
# 只读重新聚合本轮原始数组与结果，不重复跑环境
"$PY" reports/E00_plus/diagnosis_20261111/analyze.py
"$PY" reports/E00_plus/diagnosis_20261111/write_report.py
# 单项机制/Plus评测的真实命令分别保存在cases/*/command.json和plus_pilot/*/command.json
# 不要在现有目录重复启动controller；新复现必须使用独立输出目录与端口。
```

- `models.json`：三组真实训练配置、9个Router参数的初末差、训练指标首末及Gate唯一值。
- `checkpoint_identity.json`：只读权重与配置哈希；B0/B2与原记录核验，生产源码指纹保持不变。
- `training_milestones.json`：100/150/200/300/500等optimizer步的真实训练指标，配套training_curves图。
- `fairness_evidence.json`：既有等价性审计与当前源码哈希适用性。
- `mechanism_summary.json`、`attention_cells_*.npz`：分布与各head/query统计。
- `cases/*/plan.json`、`console.log`、`paired_actions.jsonl`、`router_samples/*.npz`、SQLite：真实有界诊断证据。
- `plus_selection.json`、`plus_excluded_tasks.json`、`plus_text_preflight.json`、`plus_dry_run.json`：固定pilot任务与协议。
- `plus_pilot/*/evaluation/`：每个模型的原始配置、manifest、SQLite、完整600条结果。
- `tests.xml`：3项针对性检查通过（hook不改变输出、每个去噪forward开关生效、固定有界任务与协议）。


## 10. Gate精度与Attention Collapse追加诊断

详见[专项报告](E00_PLUS_GATE_COLLAPSE_20261111.md)。真实checkpoint optimizer已确认Gate收录、非零梯度矩和FP32 master更新；BF16模型副本不变。CPU真实8任务×3噪声诊断中，B2 Query Top32 Jaccard约0.901（B1约0.426），Head0约99.8%注意力在腕部底行；存在部分head固定位置偏好，不能称整个Router完全塌缩。专项采集不占GPU、不重新训练，原始数据及3项通过的检查单独保存。
