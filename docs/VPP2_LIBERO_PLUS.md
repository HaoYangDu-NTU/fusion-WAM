# VPP2 的 LIBERO-Plus 评测

权重：`weights/vpp2`，官方 Video step 10,000 + Action step 30,000。
这是在现有 OpenWAM Plus 环境协议下评测 VPP2，官方 VPP2 发布没有提供 Plus 分数。
LIBERO-OOD、LIBERO-PRO 与这里的 LIBERO-Plus 是不同 benchmark。

## 模型与环境协议

| 项目 | 配置 |
|---|---|
| VPP2 源码 | https://github.com/roboterax/video-prediction-policy-2 |
| 固定版本 | `a103a1e1784d3e45014549a82238894d3ed85959` |
| 模型配置 | 从 Action checkpoint 的 portable `resolved_config` 读取 |
| 加载 | 官方 `LiberoActionModel.load_checkpoint`，严格加载配对 Video/Action |
| 图像 | 两相机各 256×256 原始 RGB，旋转 180 度一次；官方 `training_exact` 张量双线性缩放，各 224×224，水平拼接 224×448 |
| 本体状态 | 位置3 + 轴角3 + 两个夹爪关节位置2；使用发布的 min/max 统计 |
| 文本 | UMT5，最大长度128，原 prompt 模板；新指令编码时才将编码器移到GPU，完成后移回CPU，文本特征CPU缓存复用，释放约11GB显存 |
| 动作 | 32×7；10 步 Euler，shift5；每次只执行前10步 |
| 策略噪声 | 与官方标准评测一致：action seed42、video seed43，每次读取固定生成器种子 |
| 夹爪 | 官方 min/max 反归一化，再映射、反号并取 sign，一次转换 |
| Plus | 四套任务、七类扰动，共10,030条，各1次，修正指令后缀 |
| 环境 | 现有独立 Plus Python/MuJoCo3.3.2，每任务新进程，seed10000，settle30步 |
| 上限 | Spatial/Object/Goal600步，Long700步 |
| 并行 | 四张 GPU 各一个模型实例，128个环境 worker，调度最大 batch16，动态领取任务 |
| 显存分块 | H10080GB 实测 batch16 峰值73.49GiB、batch32 OOM；调度批次在 GPU 内分成最多16个样本，遇到额外显存压力继续减半 |

相机变换在客户端完成旋转，在策略端调用官方 `compose_libero_camera_views`；没有经过 OpenWAM 的 L-shape 缩放。
动作直接以 LIBERO7 送入环境，没有经过 OpenWAM EEF10 转换。
批量 sampler 在 `vpp2_batch.py` 中复用 VPP2 的 VAE、CLIP、视频 K/V prefill 和动作预测模块，执行相同的10步 Euler 更新。每个样本独立复用 action42/video43 初始噪声，避免一次抽取整批随机数改变后续样本的噪声。
CPU测试对比原生单样本循环与批量循环；真实权重在两个不同 Plus 观测上测试 batch1/2/4/8/16。BF16 批量矩阵计算存在数值差异，实测前六维动作最大绝对差约0.011，夹爪决策一致；不宣称逐比特相同或闭环轨迹完全相同。

## 启动与恢复

首次准备官方源码（本机已完成），不改公共 Python 环境：

```bash
git clone https://github.com/roboterax/video-prediction-policy-2.git third_party/vpp2
git -C third_party/vpp2 checkout a103a1e1784d3e45014549a82238894d3ed85959
```

全量评测及中断恢复使用同一个命令：

```bash
bash scripts/eval_vpp2_plus.sh
```

输出默认保存在 `outputs/vpp2_plus/full_w128_b16`，`outputs/vpp2_plus/CURRENT` 指向当前运行。控制器按任务持久化结果，重启时只补未完成任务；`DONE` 标记出现且 summary 完整后才作为最终成绩。
`full_w128_b16/STOP` 文件可用于请求停止控制器；恢复前移除此文件。
2026-11-04按用户要求先从8个worker/batch1切换至128/32，再固定为128/16。原队列完成的216个任务已迁入正式128/16队列；来源记录保留在 `migration.json` 和逐任务 `source_attempt` 中。
128个渲染环境也会占用GPU显存：早期128/32尝试在文本编码器常驻GPU时触发了显存分块。因此正式128/16增加文本编码器CPU卸载，保留相同BF16权重与文本特征值。

```bash
tail -f outputs/vpp2_plus/full_w128_b16.log
cat outputs/vpp2_plus/CURRENT/status.json
cat outputs/vpp2_plus/CURRENT/libero-plus/summary.json
```

闭环冒烟测试输出在 `outputs/vpp2_plus/smoke_native_final`，限制动作步数，不作为正式分数。其他 `smoke_*` 目录保留了设备迁移及动作 batch 维适配时的诊断记录，不能用于计分。
运行日志、每个尝试的模型/worker 日志、SQLite 记录、逐任务结果和各扰动类别汇总均由既有 harness 保存。

## 验证

`tests/test_vla_harness_vpp2.py` 检查原始相机方向、官方缩放结果、原生8维状态、归一化、采样种子、prompt缓存和夹爪转换。
相关回归检查日志在 `outputs/vpp2_plus/batch16_tests.log`。新增测试覆盖批量采样噪声、不同图像/本体/文本的样本隔离、文本编码器卸载与缓存复用，以及 OOM 分块顺序。
初次扩大检查时遇到两项已有测试约定不一致：旧任务身份缺少当前默认指令修正字段，以及 startup 测试仍要求已移除的 merge 参数；没有为本次模型适配修改这些已有行为。

实现入口：`benchmarks/vla_harness/vpp2_policy.py`、`vpp2_model_server.py`；命令行选择 `--policy-backend vpp2`。

## 单卡真实权重吞吐

使用两个真实 Plus 观测交替组成批次，包含图像处理和动作反归一化，结果在 `outputs/vpp2_plus/batch_profile.json`。以下为短程测量，不等于全量闭环吞吐。

| Batch | 耗时/秒 | 样本/秒 | 峰值显存/GiB |
|---|---:|---:|---:|
| 1 | 0.570 | 1.75 | 46.73 |
| 2 | 0.765 | 2.61 | 48.53 |
| 4 | 1.078 | 3.71 | 52.08 |
| 8 | 1.841 | 4.35 | 59.21 |
| 16 | 3.454 | 4.63 | 73.49 |
| 32 | OOM | — | 超过80GB卡容量 |
