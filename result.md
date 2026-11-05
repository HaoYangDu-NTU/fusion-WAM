# OpenWAM-Exp 评测结果

## E01–E08 汇总

各实验均使用 M29 + RynnBrain 9B。LIBERO 为 2,000 trials，LIBERO-Plus 为 corrected text、10,030 trials；表中仅填写完整评测成绩。

| 实验 | Visual encoder / representation | LIBERO | LIBERO-Plus | 状态 |
|---|---|---:|---:|---|
| E01 | DINOv3 H+ | **94.35%**（1887/2000） | **68.08%**（6828/10030） | 两项评测完成 |
| E02 | I-JEPA | **92.75%**（1855/2000） | **69.87%**（7008/10030） | 两项评测完成 |
| E03（E03-Patch） | Wan2.1 VAE encoder；原始 latent Cache + 空间 4×4 Patch | **95.35%**（1907/2000） | **66.93%**（6713/10030） | 两项评测完成 |
| E04 | Wan2.2 VAE encoder；原始 latent Cache + 空间 2×2 Patch | **94.50%**（1890/2000） | **76.02%**（7625/10030） | 两项评测完成 |
| E05 | VGGT-Ω 512 Encoder；冻结、双视角联合编码、空间 Patch；2 clean＋8 future，980 tokens | **73.55%**（1471/2000） | **47.63%**（4777/10030） | 两项评测完成 |
| E06 | FLUX.2 VAE（官方 Klein 仓库 BF16）；冻结、逐帧独立双视角、原生 packing/BN；2 clean＋8 future，980 tokens | **97.15%**（1943/2000） | **76.17%**（7640/10030） | 两项评测完成 |
| E07 | DINOv3-L/16；冻结、独立双视角、2 clean＋8 future，980 tokens | **95.00%**（1900/2000） | **70.54%**（7075/10030） | LIBERO / LIBERO-Plus均完成 |
| E08 | SigLIP-L/16；冻结、独立双视角、2 clean＋8 future，1280 tokens | **94.40%**（1888/2000） | **67.16%**（6736/10030） | LIBERO / LIBERO-Plus均完成 |

E03 成绩来自 E03-Patch，非已弃用的原始 E03 训练；E03-Patch 与 E04 均为 882 visual tokens。各 encoder 的输入时序及必要的输入/输出投影存在差异，完整训练配置与结果来源见各实验明细。

汇总更新：2026-10-29（北京时间）。


<!-- ENCODER_DETAIL_COMPARISON_BEGIN -->
## LIBERO 各套件成功率对比

单元格为成功率（成功数 / 有效 trial）；仅填写完整评测结果。

| 套件 | [E01](outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero/summary.json) | [E02](outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero/summary.json) | [E03](reports/E03_speed_optimization/automatic_evaluation/evaluation/libero/summary.json) | [E04](reports/E04_wan22/automatic_evaluation/evaluation/libero/summary.json) | [E05](reports/E05_training/automatic_evaluation/evaluation/libero/summary.json) | [E06](reports/E06_training/automatic_evaluation/evaluation/libero/summary.json) |
|---|---:|---:|---:|---:|---:|---:|
| libero_spatial | 97.00%（485/500） | 95.00%（475/500） | 97.40%（487/500） | 96.80%（484/500） | 89.40%（447/500） | 98.00%（490/500） |
| libero_object | 99.40%（497/500） | 98.60%（493/500） | 99.40%（497/500） | 98.60%（493/500） | 72.40%（362/500） | 98.40%（492/500） |
| libero_goal | 91.60%（458/500） | 97.00%（485/500） | 95.00%（475/500） | 95.40%（477/500） | 84.80%（424/500） | 96.60%（483/500） |
| libero_10 | 89.40%（447/500） | 80.40%（402/500） | 89.60%（448/500） | 87.20%（436/500） | 47.60%（238/500） | 95.60%（478/500） |
| 总计 | 94.35%（1887/2000） | 92.75%（1855/2000） | 95.35%（1907/2000） | 94.50%（1890/2000） | 73.55%（1471/2000） | 97.15%（1943/2000） |

## LIBERO-Plus 各类别成功率对比

统一使用 corrected text；总成功率按全部 trial 加权计算，不对七个类别取简单平均。

| 扰动类别 | [E01](outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero-plus/summary.json) | [E02](outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero-plus/evaluation/summary.json) | [E03](reports/E03_speed_optimization/automatic_evaluation/evaluation/libero-plus/summary.json) | [E04](reports/E04_wan22/automatic_evaluation/evaluation/libero-plus/summary.json) | [E05](reports/E05_training/automatic_evaluation/evaluation/libero-plus/summary.json) | [E06](reports/E06_training/automatic_evaluation/evaluation/libero-plus/summary.json) |
|---|---:|---:|---:|---:|---:|---:|
| Objects Layout | 69.44%（1059/1525） | 71.74%（1094/1525） | 76.59%（1168/1525） | 78.89%（1203/1525） | 56.20%（857/1525） | 77.44%（1181/1525） |
| Camera Viewpoints | 38.71%（619/1599） | 52.41%（838/1599） | 61.04%（976/1599） | 88.43%（1414/1599） | 45.59%（729/1599） | 39.84%（637/1599） |
| Robot Initial States | 75.35%（1168/1550） | 73.42%（1138/1550） | 67.87%（1052/1550） | 73.68%（1142/1550） | 45.23%（701/1550） | 81.81%（1268/1550） |
| Language Instructions | 67.86%（1043/1537） | 66.69%（1025/1537） | 66.95%（1029/1537） | 66.88%（1028/1537） | 27.98%（430/1537） | 79.70%（1225/1537） |
| Light Conditions | 89.14%（1018/1142） | 86.16%（984/1142） | 73.73%（842/1142） | 83.98%（959/1142） | 56.13%（641/1142） | 92.73%（1059/1142） |
| Background Textures | 80.20%（863/1076） | 75.93%（817/1076） | 41.17%（443/1076） | 34.57%（372/1076） | 70.26%（756/1076） | 82.43%（887/1076） |
| Sensor Noise | 66.08%（1058/1601） | 69.46%（1112/1601） | 75.14%（1203/1601） | 94.13%（1507/1601） | 41.41%（663/1601） | 86.38%（1383/1601） |
| 总计 | 68.08%（6828/10030） | 69.87%（7008/10030） | 66.93%（6713/10030） | 76.02%（7625/10030） | 47.63%（4777/10030） | 76.17%（7640/10030） |

<!-- ENCODER_DETAIL_COMPARISON_END -->

## E02：M29 + RynnBrain 9B + I-JEPA

标准 LIBERO 和 LIBERO-Plus 评测均已完整结束，使用同一 checkpoint：

```text
outputs/m29_rynn9b_ijepa/train/s42/2026-10-22_23-15-46/checkpoint_step_10690.safetensors
```

### LIBERO

评测目录：

```text
outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero
```

评测状态：

- 完成度：`2000 / 2000 episodes`
- 有效 trial：`2000`
- 错误：`0`
- 评测完成：是
- 总平均成功率：**92.75%**

| Benchmark | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| `libero_spatial` | 475 | 500 | **95.00%** |
| `libero_object` | 493 | 500 | **98.60%** |
| `libero_goal` | 485 | 500 | **97.00%** |
| `libero_10` | 402 | 500 | **80.40%** |
| **总计** | **1855** | **2000** | **92.75%** |

原始结果文件：

- 汇总：[summary.json](outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero/summary.json)
- SQLite：[recording-a37086a0-2398-4bed-ba01-8daef8ff46f6.sqlite](outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero/recording-a37086a0-2398-4bed-ba01-8daef8ff46f6.sqlite)

### LIBERO-Plus

使用 corrected text，评测目录：

```text
outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero-plus/evaluation
```

评测状态：

- 完成度：`10030 / 10030 trials`
- 有效 trial：`10030`
- 无效记录：`0`
- 评测完成：是（完整 benchmark，非 smoke）
- 总平均成功率：**69.87%**（`7008 / 10030`）

| 扰动类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| Objects Layout | 1094 | 1525 | **71.74%** |
| Camera Viewpoints | 838 | 1599 | **52.41%** |
| Robot Initial States | 1138 | 1550 | **73.42%** |
| Language Instructions | 1025 | 1537 | **66.69%** |
| Light Conditions | 984 | 1142 | **86.16%** |
| Background Textures | 817 | 1076 | **75.93%** |
| Sensor Noise | 1112 | 1601 | **69.46%** |
| **总计** | **7008** | **10030** | **69.87%** |

总成功率按总成功数 / 总有效 trial 计算，不对样本数不同的七个类别取简单平均。

原始结果文件：

- 汇总：[summary.json](outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero-plus/evaluation/summary.json)
- SQLite：[recording-feeda3fd-2798-4e83-ab5e-d9fe589b1bd4.sqlite](outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46/libero-plus/evaluation/recording-feeda3fd-2798-4e83-ab5e-d9fe589b1bd4.sqlite)

结果更新：2026-10-24。LIBERO 与 LIBERO-Plus 为独立 benchmark，分别报告成绩。

<!-- E01_AUTO_RESULT_BEGIN -->
## E01：M29 + RynnBrain 9B + DINOv3 H+

最终固定预算 checkpoint（seed 42，10,690 steps）：

```text
outputs/m29_rynn9b_dinov3/train/s42/2026-10-24_02-30-16/checkpoint_step_10690.safetensors
```

### LIBERO

- 完成度：`2000 / 2000`；评测完整，非 smoke。
- 错误：`0`
- 总成功率：**94.35%**（`1887 / 2000`）

| 类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| libero_spatial | 485 | 500 | **97.00%** |
| libero_object | 497 | 500 | **99.40%** |
| libero_goal | 458 | 500 | **91.60%** |
| libero_10 | 447 | 500 | **89.40%** |
| **总计** | **1887** | **2000** | **94.35%** |

原始结果：[summary.json](outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero/summary.json)，[评测目录](outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero/)。

各次尝试的 SQLite、完整计数和退出状态保存在该目录的 `attempts/` 下。

### LIBERO-Plus

- 完成度：`10030 / 10030`；评测完整，非 smoke。
- 无效记录：`0`
- 总成功率：**68.08%**（`6828 / 10030`）

使用 corrected text；总成功率按总成功数 / 总有效 trial 计算。

| 类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| Objects Layout | 1059 | 1525 | **69.44%** |
| Camera Viewpoints | 619 | 1599 | **38.71%** |
| Robot Initial States | 1168 | 1550 | **75.35%** |
| Language Instructions | 1043 | 1537 | **67.86%** |
| Light Conditions | 1018 | 1142 | **89.14%** |
| Background Textures | 863 | 1076 | **80.20%** |
| Sensor Noise | 1058 | 1601 | **66.08%** |
| **总计** | **6828** | **10030** | **68.08%** |

原始结果：[summary.json](outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero-plus/summary.json)，[评测目录](outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero-plus/)。

各次尝试的 SQLite、完整计数和退出状态保存在该目录的 `attempts/` 下。

结果更新：2026-10-24 21:02:45（北京时间）。
<!-- E01_AUTO_RESULT_END -->

<!-- E03_PATCH_AUTO_RESULT_BEGIN -->
## E03-Patch：M29 + RynnBrain 9B + Wan2.1 VAE encoder

原始 VAE latent Cache + 空间 4×4 Patch，882 visual tokens；从头训练完成 10,690 steps。

最终 checkpoint：`outputs/m29_rynn9b_wan21_vae_patch4_cache/train/s42/2026-10-25_13-38-35/checkpoint_step_10690.safetensors`。

### LIBERO

完整评测，非 smoke；总成功率：**95.35%（1907/2000）**。错误/无效记录：0。

| 套件 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| libero_spatial | 487 | 500 | **97.40%** |
| libero_object | 497 | 500 | **99.40%** |
| libero_goal | 475 | 500 | **95.00%** |
| libero_10 | 448 | 500 | **89.60%** |
| 总计 | 1907 | 2000 | **95.35%** |

原始结果：[summary.json](reports/E03_speed_optimization/automatic_evaluation/evaluation/libero/summary.json)。

### LIBERO-Plus

完整评测，非 smoke；总成功率：**66.93%（6713/10030）**。错误/无效记录：0。

使用 corrected text；总成功率按总成功数 / 总有效 trial 计算。

| 扰动类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| Objects Layout | 1168 | 1525 | **76.59%** |
| Camera Viewpoints | 976 | 1599 | **61.04%** |
| Robot Initial States | 1052 | 1550 | **67.87%** |
| Language Instructions | 1029 | 1537 | **66.95%** |
| Light Conditions | 842 | 1142 | **73.73%** |
| Background Textures | 443 | 1076 | **41.17%** |
| Sensor Noise | 1203 | 1601 | **75.14%** |
| 总计 | 6713 | 10030 | **66.93%** |

原始结果：[summary.json](reports/E03_speed_optimization/automatic_evaluation/evaluation/libero-plus/summary.json)。

<!-- E03_PATCH_AUTO_RESULT_END -->

<!-- E04_AUTO_RESULT_BEGIN -->
## E04：M29 + RynnBrain 9B + Wan2.2 VAE encoder

原始 VAE latent Cache + 空间 2×2 Patch，882 visual tokens；从头训练完成 10,690 steps。

最终 checkpoint：`outputs/m29_rynn9b_wan22_vae/train/s42/2026-10-26_10-03-52/checkpoint_step_10690.safetensors`。

### LIBERO

完整评测，非 smoke；总成功率：**94.50%（1890/2000）**。错误/无效记录：0。

| 套件 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| libero_spatial | 484 | 500 | **96.80%** |
| libero_object | 493 | 500 | **98.60%** |
| libero_goal | 477 | 500 | **95.40%** |
| libero_10 | 436 | 500 | **87.20%** |
| 总计 | 1890 | 2000 | **94.50%** |

原始结果：[summary.json](reports/E04_wan22/automatic_evaluation/evaluation/libero/summary.json)。

### LIBERO-Plus

完整评测，非 smoke；总成功率：**76.02%（7625/10030）**。错误/无效记录：0。

使用 corrected text；总成功率按总成功数 / 总有效 trial 计算。

| 扰动类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| Objects Layout | 1203 | 1525 | **78.89%** |
| Camera Viewpoints | 1414 | 1599 | **88.43%** |
| Robot Initial States | 1142 | 1550 | **73.68%** |
| Language Instructions | 1028 | 1537 | **66.88%** |
| Light Conditions | 959 | 1142 | **83.98%** |
| Background Textures | 372 | 1076 | **34.57%** |
| Sensor Noise | 1507 | 1601 | **94.13%** |
| 总计 | 7625 | 10030 | **76.02%** |

原始结果：[summary.json](reports/E04_wan22/automatic_evaluation/evaluation/libero-plus/summary.json)。

<!-- E04_AUTO_RESULT_END -->


<!-- E05_AUTO_RESULT_BEGIN -->
## E05：M29 + RynnBrain 9B + VGGT-Ω

冻结 VGGT-Ω 512 权重，224×224/视角；逐时间位置双视角联合编码，沿宽度拼接空间特征。
2 clean＋8 future，980 visual tokens；从头训练完成 10,690 steps，global batch 256，seed 42。

最终 checkpoint：`/data/fcy/code/openwam-exp/outputs/m29_rynn9b_vggt_omega_spatial/train/s42/2026-10-26_23-52-12/checkpoint_step_10690.safetensors`。

两项完整评测已于 2026-10-28 03:04（北京时间）结束。

### LIBERO

完整评测，非 smoke；总成功率：**73.55%（1471/2000）**。错误/无效记录：0。

| 套件 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| libero_spatial | 447 | 500 | **89.40%** |
| libero_object | 362 | 500 | **72.40%** |
| libero_goal | 424 | 500 | **84.80%** |
| libero_10 | 238 | 500 | **47.60%** |
| 总计 | 1471 | 2000 | **73.55%** |

原始结果：[summary.json](reports/E05_training/automatic_evaluation/evaluation/libero/summary.json)。

### LIBERO-Plus

完整评测，非 smoke；总成功率：**47.63%（4777/10030）**。错误/无效记录：0。

使用 corrected text；总成功率按总成功数 / 总有效 trial 计算。

| 扰动类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| Objects Layout | 857 | 1525 | **56.20%** |
| Camera Viewpoints | 729 | 1599 | **45.59%** |
| Robot Initial States | 701 | 1550 | **45.23%** |
| Language Instructions | 430 | 1537 | **27.98%** |
| Light Conditions | 641 | 1142 | **56.13%** |
| Background Textures | 756 | 1076 | **70.26%** |
| Sensor Noise | 663 | 1601 | **41.41%** |
| 总计 | 4777 | 10030 | **47.63%** |

原始结果：[summary.json](reports/E05_training/automatic_evaluation/evaluation/libero-plus/summary.json)。

<!-- E05_AUTO_RESULT_END -->


<!-- E06_AUTO_RESULT_BEGIN -->
## E06：M29 + RynnBrain 9B + FLUX.2 VAE

冻结 FLUX.2 VAE（官方 Klein 仓库 BF16），224×224/视角；逐帧、双视角独立编码，沿宽度拼接原生 packing＋固定 BN latent。
2 clean＋8 future，980 visual tokens；从头训练完成 10,690 steps，global batch 256，seed 42。

最终 checkpoint：`/data/fcy/code/openwam-exp/outputs/m29_rynn9b_flux2_vae/train/s42/2026-10-28_10-09-17/checkpoint_step_10690.safetensors`。

训练完成 10,690 步；两项完整评测于 **2026-10-29 04:26（北京时间）** 全部结束。

### LIBERO

完整评测，非 smoke；总成功率：**97.15%（1943/2000）**。错误/无效记录：0。

| 套件 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| libero_spatial | 490 | 500 | **98.00%** |
| libero_object | 492 | 500 | **98.40%** |
| libero_goal | 483 | 500 | **96.60%** |
| libero_10 | 478 | 500 | **95.60%** |
| 总计 | 1943 | 2000 | **97.15%** |

原始结果：[summary.json](reports/E06_training/automatic_evaluation/evaluation/libero/summary.json)。

### LIBERO-Plus

完整评测，非 smoke；总成功率：**76.17%（7640/10030）**。错误/无效记录：0。

使用 corrected text；按全部 trial 加权计算总成功率。

| 扰动类别 | 成功数 | 有效 trial | 成功率 |
|---|---:|---:|---:|
| Objects Layout | 1181 | 1525 | **77.44%** |
| Camera Viewpoints | 637 | 1599 | **39.84%** |
| Robot Initial States | 1268 | 1550 | **81.81%** |
| Language Instructions | 1225 | 1537 | **79.70%** |
| Light Conditions | 1059 | 1142 | **92.73%** |
| Background Textures | 887 | 1076 | **82.43%** |
| Sensor Noise | 1383 | 1601 | **86.38%** |
| 总计 | 7640 | 10030 | **76.17%** |

原始结果：[summary.json](reports/E06_training/automatic_evaluation/evaluation/libero-plus/summary.json)。

<!-- E06_AUTO_RESULT_END -->

<!-- E01_ABLATION_QUEUE_BEGIN -->

## E01 Prediction / Shift 消融

| Experiment | Video Pred | Action Pred | Video Shift | Spatial | Object | Goal | Long | LIBERO Overall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E01-Base | x | x | 42 | 97.00% | 99.40% | 91.60% | 89.40% | 94.35% |
| E01-P1 | x | v | 42 | 95.00% | 98.60% | 92.20% | 80.80% | 91.65% |
| E01-P2 | v | x | 42 | 93.20% | 99.60% | 96.00% | 80.20% | 92.25% |
| E01-P3 | v | v | 42 | 97.00% | 97.80% | 91.00% | 76.80% | 90.65% |
| E01-S12 | x | x | 12 | 93.60% | 98.20% | 78.00% | 66.00% | 83.95% |
| E01-S24 | x | x | 24 | 96.80% | 97.80% | 86.80% | 83.40% | 91.20% |
| E01-S42 | x | x | 42 | 97.00% | 99.40% | 91.60% | 89.40% | 94.35% |
| E01-S60 | x | x | 60 | 95.80% | 99.40% | 90.20% | 85.60% | 92.75% |

完整记录：[RESULTS.md](reports/E01_LV05_cache_w2_20261101/RESULTS.md)。

<!-- E01_ABLATION_QUEUE_END -->

2026-10-31：按用户要求，P2 于 1,957 步停止并删除正式训练 checkpoint；后续自动训练暂停，优先执行 Cache 四卡对照测试。

<!-- E01_P2_W2_RESTART_20261031 -->
2026-10-31：按用户新指令，E01-P2 使用独立 `E01-P2-Cache-W2` 配置从头重启。每卡 4 个 DataLoader workers、每 worker 2 线程，复用 DINO 原始特征缓存；模型、训练预算与 prediction(v/x) 保持原 P2。旧 step1957 运行已取消且 checkpoint 已删除。新任务完成 10690 步后自动运行标准 LIBERO 2000 trials 并更新上表；仅执行 P2，不继续 Shift 搜索或 LIBERO-Plus。队列：[E01_P2_cache_w2_20261031](reports/E01_P2_cache_w2_20261031/RESULTS.md)。

<!-- E01_P3_QUEUE_BEGIN -->

E01-P3（v/v）已完成训练与标准 LIBERO：1813/2000，Overall **90.65%**。套件结果见上方消融表；[评测记录](/data/fcy/code/openwam-exp/reports/E01_P3_cache_w2_20261031/E01-P3/evaluation/libero/summary.json)。

<!-- E01_P3_QUEUE_END -->

<!-- E01_S24_XX_REQUEST_20261031 -->
2026-10-31 晚：用户固定 baseline x/x，启动 video shift24 对照（action shift5、logit_normal、action scale1）。采用 `E01-S24-xx-Cache-W2`，从头训练 10690 步并自动评测标准 LIBERO 2000 trials。shift42 直接复用 E01-Base；本次只运行 S24。配置与请求：[REQUEST.json](reports/E01_S24_xx_cache_w2_20261031/REQUEST.json)。

<!-- E01_S60_XX_REQUEST_20261031 -->
2026-10-31 晚追加安排：S24 完成训练和标准 LIBERO 2000 trials、结果写入本文件 E01-S24 行后，再自动启动 S60。S60 固定 x/x、video shift60、action shift5，沿用 DINO cache、worker 2 线程、batch256、seed42，从头训练 10690 步；之后自动完成标准 LIBERO 并记录 E01-S60。该安排替代此前“本次仅 S24”的后续队列限制，不运行 LIBERO-Plus。队列：[REQUEST.json](reports/E01_S60_xx_cache_w2_20261031/REQUEST.json)。

<!-- E01_S12_XX_REQUEST_20261101 -->
2026-11-01 追加安排：S60 完成训练、标准 LIBERO 2000 trials 并记录 E01-S60 结果后，自动启动 E01-S12。S12 固定 x/x、video shift12、action shift5，DINO cache、worker 2 线程、batch256、10690 步、seed42；从头训练并完成标准 LIBERO，随后记录 E01-S12。GPU smoke 在前序结束后执行；不运行 LIBERO-Plus。队列：[REQUEST.json](reports/E01_S12_xx_cache_w2_20261101/REQUEST.json)。

<!-- E01_VIDEO_LOSS_BEGIN -->

## E01 Video Loss 权重消融

固定 x/x、video/action shift42/5、logit_normal、λ_action=1；仅改变 λ_video。
新实验从头训练 10690 步，global batch256、seed42，复用 DINO 原始特征 Cache、worker 2 线程。
每组先 smoke，再正式训练和标准 LIBERO 40×50=2000 trials；不运行 LIBERO-Plus。

| Experiment | λ_video | λ_action | Spatial | Object | Goal | Long | LIBERO Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| E01-Base | 1 | 1 | 97.00% | 99.40% | 91.60% | 89.40% | 94.35% |
| E01-LV05 | 0.5 | 1 | 92.40% | 99.20% | 89.00% | 86.80% | 91.85% |
| E01-LV2 | 2 | 1 | 已取消 | 已取消 | 已取消 | 已取消 | 用户停止调参，未训练 |

λ_video=1 复用原 E01-Base，不新增基线训练。历史 baseline 在线编码；新实验采用已验证的冻结特征缓存。
日志 Video Loss 已乘 λ_video；跨权重比较原始视频损失时须除以对应权重。Future MSE/cosine 不乘该权重。
单训练种子结果，不代表统计显著性；总体成功率为主，同时检查 Goal/Long。

- E01-LV05: [summary.json](/data/fcy/code/openwam-exp/reports/E01_LV05_cache_w2_20261101/E01-LV05/evaluation/libero/summary.json)

完整记录：[VIDEO_LOSS_RESULTS.md](reports/E01_LV05_cache_w2_20261101/VIDEO_LOSS_RESULTS.md)。

<!-- E01_VIDEO_LOSS_END -->

<!-- E07_AUTO_RESULT_BEGIN -->

## E07：M29 + RynnBrain 9B + DINOv3-L/16

冻结 DINOv3-L，224×224 独立双视角；2 clean＋8 future，980 visual tokens。
最后层 1024D patch features，沿宽度拼接；clean 直接投影，future bottleneck256→M29 1536D。
x/x、shift42/5、logit_normal、λ_video=λ_action=1；seed42、global batch256、10690 steps。
原始冻结特征独立 Cache；4 workers/rank、每 worker 2 线程。标准 LIBERO 40×50=2000 trials。

| 套件 | 成功数 / Trials | 成功率 |
|---|---:|---:|
| libero_spatial | 495 / 500 | 99.00% |
| libero_object | 491 / 500 | 98.20% |
| libero_goal | 474 / 500 | 94.80% |
| libero_10 | 440 / 500 | 88.00% |
| Overall | 1900 / 2000 | **95.00%** |

原始结果：[summary.json](/data/fcy/code/openwam-exp/reports/E07_training_20261102/E07/evaluation/libero/summary.json)。

<!-- E07_AUTO_RESULT_END -->

<!-- E08_AUTO_RESULT_BEGIN -->

## E08：M29 + RynnBrain 9B + SigLIP-L/16

冻结 SigLIP-L/16，256×256 独立双视角；2 clean＋8 future，1280 visual tokens。
最后层 1024D patch features，沿宽度拼接；clean 直接投影，future bottleneck256→M29 1536D。
x/x、shift42/5、logit_normal、λ_video=λ_action=1；seed42、global batch256、10690 steps。
原始冻结特征独立 Cache；4 workers/rank、每 worker 2 线程。标准 LIBERO 40×50=2000 trials。

| 套件 | 成功数 / Trials | 成功率 |
|---|---:|---:|
| libero_spatial | 485 / 500 | 97.00% |
| libero_object | 490 / 500 | 98.00% |
| libero_goal | 480 / 500 | 96.00% |
| libero_10 | 433 / 500 | 86.60% |
| Overall | 1888 / 2000 | **94.40%** |

原始结果：[summary.json](/data/fcy/code/openwam-exp/reports/E08_training_20261102/E08/evaluation/libero/summary.json)。

<!-- E08_AUTO_RESULT_END -->

<!-- E07_E08_PLUS_BEGIN -->

## E07 / E08：LIBERO-Plus

完整七类10030条；corrected instructions，冻结原final step10690权重；10步sync、每chunk执行10步动作、关闭velocity cache。

| 类别 | E07 DINOv3-L | E08 SigLIP-L |
|---|---:|---:|
| Objects Layout | 74.56%（1137/1525） | 72.72%（1109/1525） |
| Camera Viewpoints | 33.77%（540/1599） | 40.90%（654/1599） |
| Robot Initial States | 74.13%（1149/1550） | 77.81%（1206/1550） |
| Language Instructions | 74.17%（1140/1537） | 68.64%（1055/1537） |
| Light Conditions | 95.10%（1086/1142） | 91.77%（1048/1142） |
| Background Textures | 81.04%（872/1076） | 86.43%（930/1076） |
| Sensor Noise | 71.89%（1151/1601） | 45.85%（734/1601） |
| Overall | **70.54%**（7075/10030） | **67.16%**（6736/10030） |

E07 原始结果：[summary.json](/data/fcy/code/openwam-exp/reports/E07_E08_plus_20261103/E07/evaluation/libero-plus/summary.json)。

E08 原始结果：[summary.json](/data/fcy/code/openwam-exp/reports/E07_E08_plus_20261103/E08/evaluation/libero-plus/summary.json)。

<!-- E07_E08_PLUS_END -->
