# OpenWAM encoder / representation 对比实验

2026-11-04：新增 [E00+ 动作相关稀疏读出](docs/E00_PLUS.md) 与 E00+-Dense 对照；继承 E00 baseline，末端接入 clean-visual residual router。实现与验收记录见该文档，尚无本实验的正式闭环成绩。

2026-11-02：E01 调参在当前 LV05 标准 LIBERO 完成后结束，LV2 已取消。随后自动执行 E07：M29 + RynnBrain 9B + DINOv3-L/16，独立特征 Cache → 一致性验收 → smoke → 从头10690步 → 标准LIBERO。配置与恢复命令见 [E07 说明](docs/E07_DINOV3_L.md)，控制器状态位于 `reports/E07_training_20261102/status.json`。

2026-11-02 晚追加：E07完整训练及标准LIBERO记录完成后，自动接续E08：M29 + RynnBrain 9B + 原版SigLIP-L/16-256，2 clean＋8 future、1280 tokens；原始特征Cache → GPU一致性与smoke → 从头10690步 → 标准LIBERO → result.md。见 [E08 配置与恢复说明](docs/E08_SIGLIP_DESIGN.md)，队列 `reports/E08_training_20261102`；不再执行E01调参任务。

2026-11-03 追加：当前E08标准LIBERO完成后，依次完整评测E07、E08的LIBERO-Plus，并记录七类和Overall至result.md。队列 `reports/E07_E08_plus_20261103`；见 [Plus协议与恢复](docs/E07_E08_LIBERO_PLUS.md)。

项目固定 M29 Transformer 主干、RynnBrain 1.1-9B 文本条件、LIBERO 数据与训练预算，
比较不同视觉 encoder、时序表示及必要适配层。视觉 encoder 冻结，M29 策略从头训练，
必要的 Projection/Adapter 可训练。

这是**固定策略主干下的表示方案比较**，不是仅替换 encoder 权重的严格单变量实验：
各方案的时序采样、条件位置、latent 通道、token 数和输入/输出投影存在差异。

## 当前状态

更新日期：**2026-10-29**。完整计数及结果来源见 [result.md](result.md)。

| 实验 | Encoder / representation | Tokens | LIBERO | LIBERO-Plus | 状态 |
|---|---|---:|---:|---:|---|
| E00 | V-JEPA 2.1 ViT-G，原生 video tubelet | 490 | — | — | 尚未正式训练 |
| E01 | DINOv3 H+，逐帧特征 | 980 | 94.35% | 68.08% | 训练和两项评测完成 |
| E02 | I-JEPA ViT-G/16，gated history difference | 490 | 92.75% | 69.87% | 训练和两项评测完成 |
| E03（E03-Patch） | Wan2.1 VAE encoder，latent Cache + 空间 4×4 Patch | 882 | 95.35% | 66.93% | 训练和两项评测完成 |
| E04 | Wan2.2 VAE encoder，latent Cache + 空间 2×2 Patch | 882 | 94.50% | 76.02% | 训练和两项评测完成 |
| E05 | VGGT-Ω 512，逐时间双视角联合空间特征 | 980 | 73.55% | 47.63% | 训练和两项评测完成 |
| E06 | FLUX.2 VAE，逐帧独立双视角、原生 packing/BN | 980 | 97.15% | 76.17% | 训练和两项评测完成 |

E03 成绩来自 **E03-Patch**，不是已弃用的原始 E03 训练。E03 在线编码 baseline
（3528 tokens）和计算等价的 E03-Cache 仍保留为独立配置。
E01–E06 已统一在 `/data/fcy/code/openwam-exp`：共享代码、配置、Encoder 权重、Cache、训练输出和评测记录。E05/E06 不再使用独立工作目录；旧版源码与原始配置保存在 `reports/workspace_unification/`。详情见 [E05方案](docs/E05_VGGT_OMEGA_DESIGN.md)、[E06方案](docs/E06_FLUX2_VAE_DESIGN.md)。

E06 已完成从头训练10,690steps，训练循环耗时约13小时29分钟；两项完整评测于2026-10-29 04:26结束。LIBERO为97.15%（1943/2000），LIBERO-Plus为76.17%（7640/10030）；各套件/类别和E01–E06对比见 [result.md](result.md)。

E01/E02 新增独立的 frame-cache 配置及 worker 批量 mmap 读取，保留原在线基线；CPU 数值和梯度检查完成，正式使用需先通过 GPU/batch64 验收，见 [Cache 说明](docs/E01_E02_FRAME_CACHE.md)。E01 Prediction→Shift 搜索队列复用现有 Base，依次执行 P1、P2、选择、S24、S60；只评测标准 LIBERO，见 [执行与恢复说明](docs/E01_PREDICTION_SHIFT_SEARCH.md)。

~~~bash
cat /data/fcy/code/openwam-exp/reports/E06_training/status.json
tail -c 4096 /data/fcy/code/openwam-exp/reports/E06_training/formal_training.log | tr '\r' '\n'
~~~

## 目录与入口

~~~text
openwam-exp/
├── README.md                         # 当前状态、结构与操作入口
├── result.md                         # E01–E06 成绩和原始结果链接
├── EXP_RESULTS_LOG.md                # 实验登记模板与历史记录
├── configs/
│   ├── protocol.yaml                 # 共享主干、训练预算和评测协议
│   ├── experiments.yaml              # E00–E06 与 E03 变体注册表
│   ├── experiments/                  # 各 encoder、数据与 Cache overrides
│   ├── model/                        # video/action backbone 与 encoder 配置
│   └── dataloader/
├── openwam/
│   ├── experiments.py                # 统一启动、资产与 experiment identity 校验
│   ├── model/architectures/           # M29 dual-system policy
│   ├── model/video_backbone/encoder/ # V-JEPA、DINOv3、I-JEPA、Wan VAE、VGGT-Ω、FLUX.2 VAE
│   ├── model/action_backbone/
│   ├── dataloader/                    # RGB、动作、归一化与 latent Cache reader
│   ├── text_condition/               # RynnBrain 离线文本特征
│   ├── train/                        # trainer、checkpoint、full-state resume
│   └── deploy/                       # 推理与 checkpoint 重建
├── scripts/                          # 训练、Cache、评测和审计入口
├── benchmarks/vla_harness/            # LIBERO / Plus 调度、重试和结果汇总
├── tests/                            # 配置、因果性、梯度、Cache、部署回归测试
├── assets/registry.json              # 资产路径、来源和文件校验信息
├── encoder/                          # 本地视觉预训练权重
│   ├── vjepa21-g/
│   ├── dinov3-hplus/
│   ├── ijepa-g/
│   ├── wan21-vae/
│   ├── wan22-vae/
│   ├── vggt-omega/
│   └── flux2-vae/
├── cache/
│   ├── text/                         # LIBERO / Plus 的 RynnBrain 文本 Cache
│   ├── wan21_libero_windows/          # E03 原始 VAE latent
│   ├── wan22_libero_windows/          # E04 原始 VAE latent
│   ├── vggt_omega_libero_stereo/       # E05 冻结双视角空间特征
│   └── flux2_libero_frames/            # E06 冻结原生 VAE latent
├── outputs/                          # 各实验 train / smoke / eval 产物
├── reports/
│   ├── launches/                     # 实际命令、identity、step timing
│   ├── E03_speed_optimization/        # E03 历史报告、控制器、评测原始记录
│   ├── E04_wan22/                    # E04 控制器、验收与训练评测日志
│   ├── E05_training/                  # E05 训练和评测完整记录
│   ├── E06_training/                  # E06 训练和评测完整记录
│   ├── workspace_unification/         # 目录合并清单、原始源码/配置及验收
│   └── cleanup_20261026/              # 清理清单与小型源码归档
├── docs/                             # 设计、验证边界与实验说明
├── examples/encoders/                # encoder 接入示例
└── third_party/                      # 本项目使用的第三方实现
~~~

| 常用入口 | 用途 |
|---|---|
| scripts/experiment.py | 统一 train、text cache、eval 命令 |
| scripts/train.py | 分布式训练底层入口 |
| scripts/precompute_wan_latents.py | Wan2.1/2.2 VAE 原始 latent Cache |
| scripts/precompute_vggt_features.py | E05 冻结 VGGT-Ω 双视角特征 Cache |
| scripts/precompute_flux2_features.py | E06 冻结 FLUX.2 VAE latent Cache |
| scripts/audit_assets.py | 资产和文本 Cache 审计 |
| scripts/summarize_experiment_timing.py | 汇总实际 launch 的 timing |
| scripts/eval_libero_then_plus.sh | LIBERO → Plus 持久调度 |
| benchmarks/vla_harness/run_resumable.py | 保留完成 task、重试未完成 task |

reports 中的控制器针对特定历史运行，不是通用入口，不应重复执行已完成流程。
正式回归测试、权重、Cache、checkpoint 和原始评测记录继续保留。

本次移除 E03 早期一次性验证/探测脚本 **7 个**：microbatch、CUDA graph、进程并行度、
reader、在线/Cache 数值检查及已经完成的数值校准续接脚本，其 JSON/日志结果保留。
清单、原文件 SHA256 和源码归档见 [清理清单](reports/cleanup_20261026/manifest.json)。
scripts/probe_wan_view_layout.py 和 scripts/wan_cache_graph.py 暂保留：它们未接入正式
训练，但属于当前 E04 的 source fingerprint，待本次运行结束后再处理。

## 模型、时序与比较边界

M29 video DiT 为 30 层、hidden 1536、FFN 7168、12 heads；action DiT hidden 1024、
FFN 4096。RynnBrain 1.1-9B 冻结，使用 text-only、avg_all_layers 路径离线生成
128×4096 BF16 文本 Cache，训练不在线常驻 9B 文本模型。

| 配置 | 原始 latent（C×T×H×W，双视角沿宽度打包） | 空间 Patch | clean / future | Tokens |
|---|---|---|---|---:|
| E00 | 1408×5×14×28 | 2×2 | 1 / 4 | 490 |
| E01 | 1280×10×14×28 | 2×2 | 2 / 8 | 980 |
| E02 | 1408×5×14×28 | 2×2 | 1 / 4 | 490 |
| E03 / E03-Cache | 16×9×28×56 | 2×2 | 1 / 8 | 3528 |
| E03-Patch | 16×9×28×56 | 4×4 | 1 / 8 | 882 |
| E04 | 48×9×14×28 | 2×2 | 1 / 8 | 882 |

E00–E02 视觉采样使用历史、当前和稀疏未来帧，位置为
[-4,0,4,8,12,16,20,24,28,32]，各 encoder 再组织为不同 latent：

- E00：原生 video tubelet。
- E01：history/current 独立保留，future 逐帧编码。
- E02：current + alpha * P(current-history)，future 每两帧取均值；
  ViT 冻结，门控、差分投影和 affine feature LayerNorm 可训练。

E03/E04 使用**连续 [0..32]、stride=1**，两个 224×224 视角独立编码。
仅加载冻结 Wan VAE 的 encoder 与 conv1，不加载 VAE decoder、生成模型 DiT、
CLIP 或原生文本模型。33 帧经原生因果时序编码得到 9 个位置：
**1 个当前条件位置 + 8 个未来监督位置，不是 9 个历史位置**。部署只编码当前双视角帧。

Wan2.1 下采样 8 倍，接空间 4×4 Patch；Wan2.2 下采样 16 倍，接 2×2 Patch。
两者每视角最终均为 7×7 网格，共 2×9×7×7=882 tokens，每个 token 的空间覆盖
尺度均为 32×32 原始像素。Patch 不做时间池化，输入/输出投影从头训练。
这对齐了空间粒度和 token 预算，但没有消除通道和预训练压缩方式的差异。

原始 latent Cache 已验证与在线编码一致，是计算等价优化；4×4 Patch 改变输入适配
结构。报告成绩需同时注明时序、Patch、token 数和适配方式，不能把全部差异归因于 encoder 权重。

## VAE Cache

Cache 只存原始、归一化后的 BF16 VAE latent，可训练 Projection/Adapter 留在训练图。
每个 33 帧窗口重置 native VAE 时序状态，不以整段 episode 编码后切片替代。
manifest 校验数据、预处理源码、VAE 权重 SHA256、版本、shape 和双视角布局。

| Cache | Windows | 单窗口 shape | 原始 tensor 容量 |
|---|---:|---|---:|
| Wan2.1 | 273,465 | 16×9×28×56 | 115.01 GiB |
| Wan2.2 | 273,465 | 48×9×14×28 | 86.26 GiB |

每片 256 个窗口；实际文件另有少量 header/manifest。生成采用 microbatch=1、CPU 预取、
有界异步写盘、分片锁和原子替换，支持断点合并及独立 mmap 读取。当前四卡配置为每 GPU
4 个编码进程，每进程 2 个 CPU worker、prefetch factor 2、最多 2 个待写盘任务。
完整 Cache、数值检查和短 smoke 通过后才启动正式训练。

## 常用操作

~~~bash
cd /data/fcy/code/openwam-exp
PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python

# 只展开配置，不训练
"$PY" scripts/experiment.py train E04 --dry-run

# 资产与文本 Cache 审计
"$PY" scripts/audit_assets.py --validate-caches

# 只估算 VAE Cache 容量，不进行 GPU 编码
"$PY" scripts/precompute_wan_latents.py --experiment E04
~~~

需要重建 Cache 时，先确认四卡空闲；已有分片会复用：

~~~bash
"$PY" -m torch.distributed.run --standalone --nproc_per_node=16 \
  scripts/precompute_wan_latents.py --experiment E04 --build \
  --workers-per-gpu 4 --loader-workers 2 --prefetch-factor 2 --write-inflight 2
~~~

通用训练命令如下；**当前 E04 已在运行，不要重复启动**。VAE 实验正式训练前需有完整 Cache。

~~~bash
"$PY" scripts/experiment.py train E04 --gpus 0,1,2,3

# 独立 smoke，不作为正式训练起点
"$PY" scripts/experiment.py train E04 --gpus 0,1,2,3 --smoke-steps 4
~~~

### 固定训练标准与恢复

| 项目 | 配置 |
|---|---|
| GPU / batch | 4× H100 80GB；每卡 64，global batch 256，梯度累积 1 |
| 训练预算 | 10 epochs，10,690 steps，seed 42 |
| 优化器 | AdamW，LR 1e-4，betas=(0.9,0.95)，weight decay 0.01 |
| 学习率计划 | 5% warmup，cosine，最低为 base LR 的 1% |
| Loss / noise shift | video/action 均为 x-prediction，权重均为 1；shift 42 / 5 |
| Runtime | BF16、ZeRO-1、gradient checkpointing、SDPA，无 optimizer offload |
| Checkpoint | 每 500 步，保留最近 3 组，保存 full state |

checkpoint_step_*.safetensors 是推理权重；accel_state_step_*/ 保存 optimizer、scheduler、
trainer/dataloader 位置和各 rank RNG。精确恢复要求代码、配置、数据/Cache、world size 和
batch 一致，experiment_identity.json 会拒绝不兼容恢复。当前运行期间不改变其哈希范围
内的源码和配置；本次 README、docs 和 reports 临时脚本清理不改变训练源码指纹。

~~~bash
RUN_DIR=/absolute/path/to/run
touch "$RUN_DIR/STOP_TRAINING"
# 等待安全保存并退出；恢复前移除停止标记，保持原代码与配置
"$PY" scripts/experiment.py train E04 --gpus 0,1,2,3 --resume "$RUN_DIR"
~~~

### 完整评测

训练完整结束、最终权重保存并释放 GPU 后，使用最终固定预算 checkpoint，不与训练并行。
LIBERO 为 2,000 trials；Plus 使用 corrected text、10,030 trials，只有完整评测才报告
overall。两者采用同步 10 步 denoising、10 步 action execution，关闭 velocity cache。

~~~bash
RUN_DIR=/absolute/path/to/run
CKPT=checkpoint_step_10690.safetensors
"$PY" scripts/experiment.py eval "$RUN_DIR" "$CKPT" \
  --benchmark libero --output "$RUN_DIR/eval_libero"
"$PY" scripts/experiment.py eval "$RUN_DIR" "$CKPT" \
  --benchmark libero-plus --output "$RUN_DIR/eval_libero_plus"
~~~

完整结果以 [result.md](result.md) 及其链接的 summary.json、原始记录为准，保留错误/无效计数。

## 进一步阅读

- [共享实验协议](configs/protocol.yaml)、[实验索引](configs/experiments.yaml)
- [资产注册表](assets/registry.json)、[encoder 本地资产](encoder/README.md)
- [E03 VAE 设计](docs/E03_WAN_VAE_DESIGN.md)、[E03 性能与 Cache 优化记录](docs/E03_SPEED_OPTIMIZATION.md)
- [E04 后台流程和故障恢复](docs/E04_WAN22_QUEUE.md)
- [新 encoder 接入说明](examples/encoders/README.md)
- [原始实验计划](docs/EXPERIMENT_PLAN.md)（历史设计，实际运行以解析配置和 run identity 为准）


E05/E06 的统一入口（先检查配置，不启动训练）：

~~~bash
cd /data/fcy/code/openwam-exp
/data/fcy/basics/miniconda3/envs/openwam/bin/python scripts/experiment.py train E05 --dry-run
/data/fcy/basics/miniconda3/envs/openwam/bin/python scripts/experiment.py train E06 --dry-run
~~~

最终 checkpoint 分别位于 `outputs/m29_rynn9b_vggt_omega_spatial/train/s42/2026-10-26_23-52-12/` 和 `outputs/m29_rynn9b_flux2_vae/train/s42/2026-10-28_10-09-17/`。迁移清单索引：`reports/workspace_unification/latest.json`。历史 experiment identity 保留原训练源码指纹；原配置已归档，当前配置只迁移工作目录路径，不修改模型参数或评测分数。


## GitHub 分支与本地资产

`encoder-comparison` 分支发布 E01–E06 的源码、配置、设计文档、测试和完整评测汇总。`result.md` 引用的12份原始评测 `summary.json` 随源码保存，便于核对套件、类别及成功数。

预训练权重、RGB数据、特征Cache、checkpoint/优化器状态、录像、原始日志和本地源码归档保留在实验服务器，不上传Git。`assets/registry.json`、`docs/`及各Encoder配置记录了权重来源和本地路径；在其他机器运行时需准备相应资产并调整路径。CPU渲染用的本机OSMesa动态库也不随仓库发布。
