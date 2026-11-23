# fusion-WAM / OpenWAM-Exp

**M29 世界—动作模型的视觉表示对比、动作读出实验与公共评测框架。**

本仓库基于 OpenWAM，保留原项目命名和 Apache-2.0 许可证。核心研究固定 M29 Joint Attention 主干、RynnBrain 1.1-9B 文本条件、LIBERO 数据和训练预算，比较冻结视觉 Encoder 及必要适配层；另提供 FM-VJEPA、VPP2 的原生评测适配器。

更新：**2026-11-05**。这里区分已实现、已验证、已完成评测和后续计划；历史设计文档中的“正在运行/待执行”以本页和结果记录为准。

| 阅读目标 | 入口 |
|---|---|
| 实验结果与套件/扰动明细 | [result.md](result.md) |
| 每阶段改动、实验与状态变化 | [CHANGELOG.md](CHANGELOG.md) |
| 40个实验配置的注册表 | [configs/experiments.yaml](configs/experiments.yaml)，本页完整索引见下文 |
| 固定主干、训练预算、评测协议 | [configs/protocol.yaml](configs/protocol.yaml) |
| 训练/离线文本/评测统一入口 | [scripts/experiment.py](scripts/experiment.py) → [openwam/experiments.py](openwam/experiments.py) |
| 仿真平台接入、RoboCasa 训练与 DROID 评测 | [各平台指南](benchmarks/README.md) |
| 性能测量和评测框架回归测试 | [性能工具](scripts/profiling/) · [VLA Harness 测试](tests/benchmarks/vla_harness/) |
| 资产路径、文件大小、来源和校验 | [assets/registry.json](assets/registry.json) |
| 本次评测速度优化的实现与验证 | [实现报告](docs/EVALUATION_RUNTIME_IMPLEMENTATION.md)、[完整方案](docs/ROBOSCOPE_EVALUATION_SPEED_PLAN.md) |

## 1. 先回答：是否需要下载新权重？

| 要做的事情 | 是否需要新权重 | 还需要什么 |
|---|---|---|
| 给已有模型启用本次公共评测提速 | **不需要**；沿用当前策略和 Encoder | 更新代码，在原评测命令后加 `--client-action-queue`；保留模型原来的环境、配置和文本特征 |
| 切换 E01–E08 的视觉 Encoder | **需要对应预训练 Encoder**，已有同版本文件可复用 | 数据、对应配置、RynnBrain 文本 Cache；缓存训练还需构建并验收视觉 Cache |
| 从头训练 E00 / E00+-Dense / E00+ | 复用同一 V-JEPA 2.1 与 RynnBrain，无新增 Router 预训练权重 | Router 随新策略训练；Dense/Sparse 是独立实验 |
| 评测训练完成的 E 系列策略 | 需要**对应实验训练出的策略 checkpoint** | 保存的 `config.yaml`、`normalization_stats.npy`、Encoder/文本资产及适用 sidecar |
| 只评测 FM-VJEPA 或 VPP2 | 需要各自发布/训练权重 | 各自原生 Encoder、文本/归一化配置；不使用 RynnBrain 替代其原生文本条件 |
| 运行配置解析、代码阅读或通信专项探针 | 不需要模型权重 | 对应 Python 依赖；通信探针使用替身策略 |

GitHub **不包含**预训练权重、RGB 数据、完整特征 Cache、训练 checkpoint、优化器状态和大规模运行日志。仓库中的本地绝对路径不是公共下载地址；**尚未提供 E 系列策略 checkpoint 的统一公开下载包**。下载源码不等于能立即复现表中成绩，需自行训练或另外取得对应训练产物。

## 2. 主实验与完整成绩

### 2.1 E00–E08 与 E00+ 消融

以下 E 系列均为 M29 + RynnBrain 9B。LIBERO 为40个任务×50次，共2,000 trials；LIBERO-Plus 为 corrected instructions、10,030 trials。数字来自 [result.md](result.md)，仅列完整评测。

| ID / 配置 | 视觉表示或改动 | clean / future；视觉 tokens | LIBERO | LIBERO-Plus |
|---|---|---|---:|---:|
| [E00](configs/experiments/E00.yaml) | V-JEPA 2.1 ViT-G，原生视频 tubelet | 1 / 4；490 | 未登记完整成绩 | 未登记完整成绩 |
| [E00+-Dense](configs/experiments/E00+-Dense.yaml) | E00 + 动作条件化残差视觉读出，稀疏权重0 | 同 E00 | 未完成正式闭环评测 | 未完成正式闭环评测 |
| [E00+](configs/experiments/E00+.yaml) | Dense + top-k attention mass loss | 同 E00 | 未完成正式闭环评测 | 未完成正式闭环评测 |
| [E01](configs/experiments/E01.yaml) | DINOv3 H+，逐帧双视角 | 2 / 8；980 | **94.35%** | **68.08%** |
| [E02](configs/experiments/E02.yaml) | I-JEPA G，历史门控差分、未来成对均值 | 1 / 4；490 | **92.75%** | **69.87%** |
| [E03-Patch](configs/experiments/E03-Patch.yaml) | Wan2.1 VAE，原始 latent Cache + 空间4×4 Patch | 1 / 8；882 | **95.35%** | **66.93%** |
| [E04](configs/experiments/E04.yaml) | Wan2.2 VAE，原始 latent Cache + 空间2×2 Patch | 1 / 8；882 | **94.50%** | **76.02%** |
| [E05](configs/experiments/E05.yaml) | VGGT-Ω 512，逐时间双视角联合空间特征 | 2 / 8；980 | **73.55%** | **47.63%** |
| [E06](configs/experiments/E06.yaml) | FLUX.2 Klein VAE，独立逐帧双视角、原生 packing/BN | 2 / 8；980 | **97.15%** | **76.17%** |
| [E07](configs/experiments/E07.yaml) | DINOv3-L/16，自监督视觉表示 | 2 / 8；980 | **95.00%** | **70.54%** |
| [E08](configs/experiments/E08.yaml) | 原版 SigLIP-L/16-256，原生256分辨率 | 2 / 8；1280 | **94.40%** | **67.16%** |

E03 的已报告成绩属于 **E03-Patch**，不是原始3528-token的 `E03` 或 `E03-Cache`。各方案时序、通道、分辨率、token数和必要投影不同，这是固定主干下的**表示方案比较**，不能将全部差异归因于预训练 Encoder 权重。

### 2.2 E01 预测类型、噪声 shift、损失权重消融

表中 `x/v` 顺序为 video/action。除注明变化外固定 action shift5、video shift42、相同训练预算。结果为标准 LIBERO，未因此补跑 Plus。

| 结果标签 | 设置 | LIBERO | 对应主要配置 |
|---|---|---:|---|
| E01-Base / E01-S42 | x/x，video shift42 | 94.35% | `E01`，复用原结果 |
| E01-P1 | x/v | 91.65% | `E01-P1-Cache`；在线 `E01-P1` 也保留 |
| E01-P2 | v/x | 92.25% | `E01-P2-Cache-W2`；旧中断运行不计分 |
| E01-P3 | v/v | 90.65% | `E01-P3-Cache-W2` |
| E01-S12 | x/x，video shift12 | 83.95% | `E01-S12-xx-Cache-W2` |
| E01-S24 | x/x，video shift24 | 91.20% | `E01-S24-xx-Cache-W2` |
| E01-S60 | x/x，video shift60 | 92.75% | `E01-S60-xx-Cache-W2` |
| E01-LV05 | x/x，λ_video=0.5，λ_action=1 | 91.85% | `E01-LV05-Cache-W2` |
| E01-LV2 | λ_video=2 | **已取消，未训练** | 配置保留，不作为已完成实验 |

`-Cache` 表示缓存冻结 Encoder 原始输出；`-W2` 表示每个 DataLoader worker 使用2个 CPU 线程，不表示2卡、2层模型或新的预训练权重。S24/S60 的其他 x/v 组合有配置，但不代表均已运行。单训练种子结果不构成统计显著性证据。详细过程见 [E01消融记录](docs/E01_PREDICTION_SHIFT_SEARCH.md)。

### 2.3 外部策略与公共评测框架

| 后端 | 模型/训练归属 | 原生动作使用 | 入口与权重说明 |
|---|---|---|---|
| `openwam` | 本仓库 E 系列 M29 | 当前评测默认执行前10步；以配置为准 | [model_server.py](benchmarks/vla_harness/model_server.py) |
| `fm-vjepa` | 原生 FM-VJEPA-Policy；非 E 系列训练实验 | 预测32，执行16；逐控制步 t-4 历史 | [FM适配说明](benchmarks/vla_harness/FM_VJEPA.md) |
| `vpp2` | 发布的 VPP2 Video10k + Action30k；非 M29/RynnBrain | 预测32，执行10，10步 Euler | [VPP2说明](docs/VPP2_LIBERO_PLUS.md) |

VPP2 在本项目 Plus 协议下已完成 **8,342/10,030 = 83.17%**，无无效记录。来源是本地 `outputs/vpp2_plus/full_w128_b16/libero-plus/summary.json`，该运行产物未随 GitHub 发布。它是不同规模/结构模型的外部评测结果，不能加入“仅替换视觉 Encoder”的单变量结论，也不是本次新传输路径的加速验收。

## 3. 全部40个已注册配置

注册表为 [configs/experiments.yaml](configs/experiments.yaml)。以下逐项列出，避免混淆结果标签与可执行实验名；“已注册”不等于“已训练/已评测”。所有条目都可通过 `scripts/experiment.py train ID --dry-run` 解析。

| ID | 配置文件 | 基础 / 主要差异 |
|---|---|---|
| `E00` | [E00.yaml](configs/experiments/E00.yaml) | vjepa21_g；490 tokens |
| `E00+` | [E00+.yaml](configs/experiments/E00+.yaml) | 继承 E00；残差 Router；稀疏权重0.01 |
| `E00+-Dense` | [E00+-Dense.yaml](configs/experiments/E00+-Dense.yaml) | 继承 E00；残差 Router；稀疏权重0.0 |
| `E01` | [E01.yaml](configs/experiments/E01.yaml) | dinov3_hplus；980 tokens |
| `E02` | [E02.yaml](configs/experiments/E02.yaml) | ijepa_g；490 tokens |
| `E01-Cache` | [E01-Cache.yaml](configs/experiments/E01-Cache.yaml) | dinov3_hplus；980 tokens；冻结逐帧特征 Cache |
| `E02-Cache` | [E02-Cache.yaml](configs/experiments/E02-Cache.yaml) | ijepa_g；490 tokens；冻结逐帧特征 Cache |
| `E03` | [E03.yaml](configs/experiments/E03.yaml) | wan21_vae；3528 tokens |
| `E03-Cache` | [E03-Cache.yaml](configs/experiments/E03-Cache.yaml) | wan21_vae；3528 tokens；冻结 VAE latent Cache |
| `E03-Patch` | [E03-Patch.yaml](configs/experiments/E03-Patch.yaml) | wan21_vae；882 tokens；冻结 VAE latent Cache |
| `E04` | [E04.yaml](configs/experiments/E04.yaml) | wan22_vae；882 tokens；冻结 VAE latent Cache |
| `E05` | [E05.yaml](configs/experiments/E05.yaml) | vggt_omega_512；980 tokens |
| `E06` | [E06.yaml](configs/experiments/E06.yaml) | flux2_vae；980 tokens |
| `E07` | [E07.yaml](configs/experiments/E07.yaml) | dinov3_l；980 tokens；worker线程=2 |
| `E07-Cache` | [E07-Cache.yaml](configs/experiments/E07-Cache.yaml) | 继承 E07；冻结逐帧特征 Cache |
| `E08` | [E08.yaml](configs/experiments/E08.yaml) | siglip_l；1280 tokens；worker线程=2 |
| `E08-Cache` | [E08-Cache.yaml](configs/experiments/E08-Cache.yaml) | 继承 E08；冻结逐帧特征 Cache |
| `E01-P1` | [E01-P1.yaml](configs/experiments/E01-P1.yaml) | 继承 E01；x/v；video shift42 |
| `E01-P1-Cache` | [E01-P1-Cache.yaml](configs/experiments/E01-P1-Cache.yaml) | 继承 E01-Cache；x/v；video shift42 |
| `E01-P2` | [E01-P2.yaml](configs/experiments/E01-P2.yaml) | 继承 E01；v/x；video shift42 |
| `E01-S24-xx` | [E01-S24-xx.yaml](configs/experiments/E01-S24-xx.yaml) | 继承 E01；x/x；video shift24 |
| `E01-S24-xv` | [E01-S24-xv.yaml](configs/experiments/E01-S24-xv.yaml) | 继承 E01；x/v；video shift24 |
| `E01-S24-vx` | [E01-S24-vx.yaml](configs/experiments/E01-S24-vx.yaml) | 继承 E01；v/x；video shift24 |
| `E01-S60-xx` | [E01-S60-xx.yaml](configs/experiments/E01-S60-xx.yaml) | 继承 E01；x/x；video shift60 |
| `E01-S60-xv` | [E01-S60-xv.yaml](configs/experiments/E01-S60-xv.yaml) | 继承 E01；x/v；video shift60 |
| `E01-S60-vx` | [E01-S60-vx.yaml](configs/experiments/E01-S60-vx.yaml) | 继承 E01；v/x；video shift60 |
| `E01-P2-Cache` | [E01-P2-Cache.yaml](configs/experiments/E01-P2-Cache.yaml) | 继承 E01-Cache；v/x；video shift42 |
| `E01-P2-Cache-W2` | [E01-P2-Cache-W2.yaml](configs/experiments/E01-P2-Cache-W2.yaml) | 继承 E01-Cache；v/x；video shift42；worker线程=2 |
| `E01-P3-Cache-W2` | [E01-P3-Cache-W2.yaml](configs/experiments/E01-P3-Cache-W2.yaml) | 继承 E01-Cache；v/v；video shift42；worker线程=2 |
| `E01-S24-xx-Cache-W2` | [E01-S24-xx-Cache-W2.yaml](configs/experiments/E01-S24-xx-Cache-W2.yaml) | 继承 E01-Cache；x/x；video shift24；worker线程=2 |
| `E01-S60-xx-Cache-W2` | [E01-S60-xx-Cache-W2.yaml](configs/experiments/E01-S60-xx-Cache-W2.yaml) | 继承 E01-Cache；x/x；video shift60；worker线程=2 |
| `E01-S12-xx-Cache-W2` | [E01-S12-xx-Cache-W2.yaml](configs/experiments/E01-S12-xx-Cache-W2.yaml) | 继承 E01-Cache；x/x；video shift12；worker线程=2 |
| `E01-LV05-Cache-W2` | [E01-LV05-Cache-W2.yaml](configs/experiments/E01-LV05-Cache-W2.yaml) | 继承 E01-Cache；x/x；video shift42；λ_video=0.5；worker线程=2 |
| `E01-LV2-Cache-W2` | [E01-LV2-Cache-W2.yaml](configs/experiments/E01-LV2-Cache-W2.yaml) | 继承 E01-Cache；x/x；video shift42；λ_video=2.0；worker线程=2 |
| `E01-S24-xx-Cache` | [E01-S24-xx-Cache.yaml](configs/experiments/E01-S24-xx-Cache.yaml) | 继承 E01-Cache；x/x；video shift24 |
| `E01-S24-xv-Cache` | [E01-S24-xv-Cache.yaml](configs/experiments/E01-S24-xv-Cache.yaml) | 继承 E01-Cache；x/v；video shift24 |
| `E01-S24-vx-Cache` | [E01-S24-vx-Cache.yaml](configs/experiments/E01-S24-vx-Cache.yaml) | 继承 E01-Cache；v/x；video shift24 |
| `E01-S60-xx-Cache` | [E01-S60-xx-Cache.yaml](configs/experiments/E01-S60-xx-Cache.yaml) | 继承 E01-Cache；x/x；video shift60 |
| `E01-S60-xv-Cache` | [E01-S60-xv-Cache.yaml](configs/experiments/E01-S60-xv-Cache.yaml) | 继承 E01-Cache；x/v；video shift60 |
| `E01-S60-vx-Cache` | [E01-S60-vx-Cache.yaml](configs/experiments/E01-S60-vx-Cache.yaml) | 继承 E01-Cache；v/x；video shift60 |


## 4. 目录、配置与资产关系

```text
fusion-WAM/                          # 服务器目录名为 openwam-exp
├── README.md / CHANGELOG.md         # 总览、入口与阶段变更
├── result.md                        # 成绩、成功数和历史来源
├── configs/protocol.yaml            # 共同训练/数据/评测协议
├── configs/experiments.yaml         # 实验ID注册
├── configs/experiments/*.yaml       # 每组 Encoder、Cache、预测、shift、Router overrides
├── configs/train.yaml               # Hydra训练总入口
├── configs/model/dual_system.yaml   # M29结构及Router默认配置
├── configs/model/video_backbone/encoder/ # 各Encoder接口与默认资产路径
├── configs/deploy.yaml              # 部署/采样默认配置
├── configs/dataloader/              # LIBERO读取、视角、历史、动作接口
├── assets/registry.json             # 本地资产路径与部分固定hash
├── openwam/experiments.py           # 配置合成、preflight、identity及统一启动
├── openwam/model/                  # 冻结Encoder、M29、Action Expert、Router
├── openwam/dataloader/              # RGB/原始特征Cache reader与collator
├── openwam/text_condition/          # 离线文本特征、指纹与Cache
├── openwam/train/ / deploy/         # 训练、checkpoint、推理重建
├── scripts/                        # 统一入口、Cache生成、队列、探针
├── benchmarks/vla_harness/          # LIBERO/Plus公共调度与模型adapter
├── tests/                          # 回归、协议、梯度、Cache和传输测试
├── docs/                           # 每阶段设计与实施报告
└── third_party/                    # 随源码保存的第三方实现，许可证各自保留
```

`encoder/`、`cache/`、`weights/`、大部分 `outputs/`/`reports/` 为本地资产或运行目录，不会由克隆自动创建。少量已版本化的 E01–E06 `summary.json` 保留用于核对成绩；其他文档中的本地报告路径可能在 GitHub 不存在。`RoboScope-Kit/` 与 `genfirst/` 是独立本地仓库，未包含在本仓库；当前公共动作队列不要求安装 RoboScope。

### 4.1 换机器先改哪些位置

1. 在 [assets/registry.json](assets/registry.json) 中为**选中实验**设置预训练模型目录，核对名称、文件大小、revision与可用hash；不要用同名但不同尺寸/版本权重替代。
2. 修改 [configs/protocol.yaml](configs/protocol.yaml) 的 `data.dataset_dir`、文本Cache路径和运行环境记录；检查对应实验 YAML 的 `data_overrides`，许多 Cache 路径仍是服务器绝对路径。实际训练解释器来自启动命令的 `python` / `sys.executable`，不是仅改 `runtime.python` 就能切换；shell评测包装器使用 `OPENWAM_PYTHON`。
3. 检查 [openwam/experiments.py](openwam/experiments.py) 的 `LIBERO_CATALOG`、`PLUS_CATALOG`。它们仍指向原服务器的指令目录，并非跨机器自动发现；通过统一入口 cache/preflight/eval 前需提供等价目录或调整这两个常量。也可用下面显式参数的文本生成脚本准备Cache。
4. 统一入口 cache 使用 `assets/pretrained/rynnbrain-9b`，需链接到准确的 RynnBrain 目录；评测时核对 checkpoint 配置中的 Encoder 路径、文本路径和 sidecar。只修改环境变量不会自动覆盖所有已保存绝对路径。
5. 为 LIBERO 和 Plus 分别准备正确的外部 benchmark checkout。通过 `--libero-path` 指定当前评测对应的 checkout；Plus 还需 `--libero-plus-python` 指定其独立解释器。

当前尚不是“一条命令在任意机器自动下载并复现实验”的发行包。先 `--dry-run` 核对展开配置，再做对应资产检查和小规模验收。

## 5. 环境与权重准备

### 5.1 Python 与模拟环境

主项目要求 Python≥3.10，依赖见 [pyproject.toml](pyproject.toml)。在匹配本机 CUDA 的新环境中安装；不要覆盖正在训练的公共环境：

```bash
git clone https://github.com/HaoYangDu-NTU/fusion-WAM.git
cd fusion-WAM
# 先按本机CUDA安装兼容的PyTorch，再安装项目和随仓库的harness
python -m pip install -e '.[dev]'
python -m pip install -e third_party/vla-evaluation-harness
export PYTHONPATH="$PWD/third_party/vla-evaluation-harness/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENWAM_PYTHON="$(command -v python)"
```

模拟环境另见 [标准LIBERO](benchmarks/libero/README.md)、[LIBERO-Plus环境说明](benchmarks/libero-plus/README.md)。当前 harness 检查 `mujoco==3.3.2`；Plus 使用独立 Python3.10 子进程保留官方初始化/RNG顺序。[requirements.lock.txt](benchmarks/vla_harness/requirements.lock.txt) 是模拟依赖快照，不是整个训练、FM、VPP2环境的通用锁文件。FM/VPP2还需各自原生依赖。`third_party/osmesa/` 的本机库未发布；CPU renderer 需要自行准备，正式对比固定同一 renderer。

### 5.2 按实验选择视觉权重，不必下载所有模型

名称、文件尺寸与可用hash以 [资产注册表](assets/registry.json) 和各实验文档为准：

| 用途 | 必须匹配的资产 | 本地建议位置 / 说明 |
|---|---|---|
| 全部 E 系列文本 | RynnBrain **1.1-9B**，32层、4096维；text-only、avg_all_layers | `assets/pretrained/rynnbrain-9b` → 自行准备的模型目录；训练使用离线特征 |
| E00 / Dense / Sparse | `vjepa2_1_vitg_384.pt`、对应manifest，`target_encoder` | `encoder/vjepa21-g/`；[下载器](scripts/download_assets/download_visual_encoder.py) 中有模型来源 |
| E01及其消融 | DINOv3 **H+**，1280维，注册表指定 revision | `encoder/dinov3-hplus/`；不能用默认 DINO-B 代替 |
| E02 | I-JEPA **ViT-G/16**，1408维，`checkpoint.pth.tar` | `encoder/ijepa-g/` |
| E03系列 | Wan2.1 的 `Wan2.1_VAE.pth` | `encoder/wan21-vae/`；[登记来源](assets/registry.json)，只用Encoder/conv1 |
| E04 | Wan2.2 TI2V-5B 中的 `Wan2.2_VAE.pth` | `encoder/wan22-vae/`；只用Encoder/conv1 |
| E05 | 官方 VGGT-Ω 512 的 `vggt_omega_1b_512.pt` | `encoder/vggt-omega/`；[来源与接口](docs/E05_VGGT_OMEGA_DESIGN.md) |
| E06 | **FLUX.2-klein-4B VAE BF16** 导出的Encoder与BN配置 | `encoder/flux2-vae/{config.json,encoder_only.safetensors}`；[导出/packing说明](docs/E06_FLUX2_VAE_DESIGN.md) |
| E07 | `facebook/dinov3-vitl16-pretrain-lvd1689m`，1024维 | `encoder/dinov3-l/`；[固定revision/hash](docs/E07_DINOV3_L.md) |
| E08 | `google/siglip-large-patch16-256`，**原版SigLIP**，不是SigLIP2 | `encoder/siglip-large-patch16-256/vision_encoder/`；[纯视觉导出说明](docs/E08_SIGLIP_DESIGN.md) |

E00–E08 的 M29 策略按协议**从头初始化**，无需另下 Wan DiT 预训练权重。E03/E04只使用 VAE Encoder，不需要下载整套视频生成 DiT、CLIP、T5。E06的历史实验源是Klein，下载器中的其他FLUX.2默认选项不能直接视为同一资产。`scripts/download_assets/` 提供辅助工具，但其默认项不保证覆盖全部实验的指定版本。

新评测运行时不改变 checkpoint tensor 结构，不需重新训练策略或重新下载 Encoder；它也不会自动产生缺失的文本Cache。

### 5.3 RynnBrain 离线文本Cache

训练Cache为 `cache/text/rynnbrain1p1-9b/libero/avg_all_layers/`，Plus为 `cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text/`。共享文本输出为128×4096 BF16，训练时不在线常驻9B模型。

下面参数均需替换为真实路径；命令会使用模型/GPU生成特征：

```bash
python scripts/precompute_libero_text.py \
  --text-encoder-type rynnbrain1p1-9b --model-path /path/to/RynnBrain1_1-9B \
  --vlm-text-embedding-method avg_all_layers --dataset-dir /path/to/libero-data \
  --output-dir cache/text/rynnbrain1p1-9b/libero/avg_all_layers

python scripts/precompute_libero_plus_text.py \
  --text-encoder-type rynnbrain1p1-9b --model-path /path/to/RynnBrain1_1-9B \
  --vlm-text-embedding-method avg_all_layers --text-mode corrected_text \
  --libero-path /path/to/LIBERO-plus --libero-plus-python /path/to/plus-env/bin/python \
  --output-dir cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers
```

Plus脚本在输出目录下再创建 `corrected_text/`，不要重复拼接该层。具有准确指令catalog时可用 `--catalog-manifest` / `--catalog-dir` 复用；不能把RynnBrain、UMT5或FM原生T5的缓存互换。Plus脚本无catalog时通过独立解释器导出任务，当前导出路径会设置OSMesa，需要相应导入依赖。统一别名 `python scripts/experiment.py cache libero --dry-run` 可查看默认命令，但会使用前述服务器catalog路径。

### 5.4 冻结视觉Cache：训练加速，不是新预训练权重

| 实验 | 构建入口 | 验收 / 特点 |
|---|---|---|
| E01/E02/E07/E08 | [precompute_image_features.py](scripts/precompute_image_features.py) | [validate_image_features.py](scripts/validate_image_features.py)；实际时间戳去重，mmap，冻结原始grid |
| E03/E04 | [precompute_wan_latents.py](scripts/precompute_wan_latents.py) | 原生33帧窗口重置时序状态，存VAE原始latent；E03-Patch复用E03 Cache |
| E05 | [precompute_vggt_features.py](scripts/precompute_vggt_features.py) | 每时间点联合双视角空间特征 |
| E06 | [precompute_flux2_features.py](scripts/precompute_flux2_features.py) | 单帧原生packing与固定BN后的latent |

只缓存冻结输出，不缓存可训练Projection、Router或时序适配器。历史/未来索引、padding、权重及预处理指纹必须一致；失败验收不能强行跳过。视觉缓存往往需要几十至数百GiB，本地已有完整且匹配的Cache可复用。部署仍在线编码真实观测，**评测不依赖训练视觉Cache**。

```bash
# 图像Encoder：默认只估算，--build才生成；E02/E07/E08替换实验名
python scripts/precompute_image_features.py --experiment E01
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/precompute_image_features.py --experiment E01 --build
python scripts/validate_image_features.py --experiment E01

# Wan VAE：默认只估算；--build才构建全量
python scripts/precompute_wan_latents.py --experiment E04
# E05/E06构建器没有--build开关，直接执行会生成Cache，先查看--help
python scripts/precompute_vggt_features.py --help
python scripts/precompute_flux2_features.py --help
```

## 6. 训练、恢复与实验队列

统一配置入口为 `configs/protocol.yaml` + `configs/experiments/ID.yaml`，底层训练为 [scripts/train.py](scripts/train.py) 与 [openwam/train/openwam_trainer.py](openwam/train/openwam_trainer.py)。

```bash
# 不加载训练权重，不启动GPU训练；展开配置写入reports/resolved/
python scripts/experiment.py train E07-Cache --dry-run
python scripts/experiment.py train 'E00+' --dry-run

# 先短程验收，再正式训练；正式对比固定4卡和batch
python scripts/experiment.py train E07-Cache --gpus 0 --batch-size 1 --smoke-steps 4
python scripts/experiment.py train E07-Cache --gpus 0,1,2,3 --seed 42

# 恢复同一实验、同一运行，不是加载另一个实验的checkpoint微调
python scripts/experiment.py train E07-Cache --gpus 0,1,2,3 --seed 42 \
  --resume /path/to/the/same/run
```

| 共享训练项 | 设置 |
|---|---|
| Backbone | M29：30层Joint Attention；video hidden1536、12 heads；action hidden1024 |
| 预算 | seed42，10 epochs；参考273,465窗口，10,690 optimizer steps；实际按锁定sampler推导 |
| Batch | 4卡×64，global256，累积1 |
| 优化 | AdamW、LR1e-4、betas(0.9,0.95)、weight decay0.01；5% warmup、cosine |
| 默认预测/噪声 | video/action x/x，shift42/5；λ_video=λ_action=1；消融单独覆盖 |
| Runtime | BF16、ZeRO-1、gradient checkpointing、SDPA |
| 保存 | 每500步、最近3组、最终checkpoint；full-state用于精确恢复 |

输出为 `outputs/<profile.name>/train/s42/<timestamp>/`，配置解析和身份记录在 `reports/resolved/`、`reports/launches/`。推理权重是 `checkpoint_step_*.safetensors`；精确续训还需 `accel_state_step_*/` 的优化器、scheduler、采样与各rank RNG。`experiment_identity.json` 会检查源码、配置、数据与资产；跨版本改动不能当作原运行无差别续训。

| 自动流程 | 入口 | 范围 |
|---|---|---|
| E01预测/shift/Loss消融 | [e01_ablation_queue.py](scripts/e01_ablation_queue.py) | 支持 `--single-experiment`、`--result-label`；历史任务顺序见实验文档 |
| E07/E08 Cache→验收→smoke→训练→LIBERO | [e07_train_eval_queue.py](scripts/e07_train_eval_queue.py) | `--experiment E07` 或 `E08`；默认带历史前序队列依赖 |
| E07/E08 Plus补评测 | [e07_e08_plus_queue.py](scripts/e07_e08_plus_queue.py) | 已有训练产物的两组Plus评测 |
| E00+匹配微调pilot | [e00_plus_finetune.py](scripts/e00_plus_finetune.py) | 已有E00权重→三组smoke→初始B0评测→B0/B1/B2各1000步+标准LIBERO；等待GPU空闲 |
| E00+冻结主干对照 | [e00_plus_frozen_router.py](scripts/e00_plus_frozen_router.py) | 原始E00配置→Decoder-only / FP32 Dense Router各1000步→相同600条Plus pilot；[GPU空闲队列](scripts/e00_plus_frozen_router_wait_eval.py) |
| 标准LIBERO→Plus持久评测 | [eval_libero_then_plus.sh](scripts/eval_libero_then_plus.sh) | 独立checkpoint、结果校验、断点重试 |

这些控制器包含本机路径和历史依赖，不能在新机器直接复用旧队列目录。阅读 `--help`，用新目录并明确任务；不要把旧报告文件存在等同于任务已完成。

## 7. E00+：Action-Relevant Sparse Readout

E00 → E00+-Dense → E00+ 对应 B0/B1/B2。实现位于 [action_router.py](openwam/model/architectures/utils/action_router.py)，接入点为 [joint_self_attn.py](openwam/model/architectures/dual_system/joint_self_attn.py) 的 joint loop 后、action decoder 前。

- Action hidden 作Q，最终 clean visual hidden 作K/V；512维、8 heads、top-k=32。
- 保留原 action hidden，以可学习 gate 加残差，初始有效gate=0.01。
- Sparse版增加 `mean(1 - topk_attention_mass)`；λ从第1000步后开始上升，在5000步达到0.01；Dense版λ始终0。
- 保留稠密主干和全部视觉token，属于soft稀疏读出，不是硬裁剪或已证实的因果解耦。
- 已有数学、mask、梯度、重载、真实资产CPU验收；尚未登记正式B0/B1/B2闭环收益。

新Router没有外部预训练权重。从头训练命令同第6节，实验ID替换为 `'E00+-Dense'` 或 `'E00+'`。

2026-11-10新增匹配微调入口 [e00_plus_finetune.py](scripts/e00_plus_finetune.py)，配置为 [e00_plus_finetune.yaml](configs/e00_plus_finetune.yaml)。使用已核验的M29/RynnBrain9B/V-JEPA E00 step10690，为B1/B2显式追加相同Router初值，逐项验证共有权重未变，之后继续严格加载。三组同源、LR1e-5、四卡global batch256，各微调1000步；B2稀疏项使用100步warmup+400步ramp。该pilot与原10690步从头训练分别记账，尚无闭环收益结论。运行顺序为 `--prepare` → `CUDA_VISIBLE_DEVICES='' ... --validate-cpu` → `--run`，不需下载新预训练权重。具体命令、路径、诊断和停止方式见 [E00_PLUS.md](docs/E00_PLUS.md)。

后续记录见 [B0/B1/B2实验日志](docs/E00_PLUS_EXPERIMENT_LOG_20261110.md)、[公平性与机制诊断](docs/E00_PLUS_DIAGNOSIS_20261111.md)、[Gate精度与Attention Collapse检查](docs/E00_PLUS_GATE_COLLAPSE_20261111.md)。2026-11-12的[冻结主干实验](docs/E00_PLUS_FROZEN_ROUTER_20261112.md)已完成两组1000步微调，使用FP32 Gate与残差计算，并从checkpoint保留原始模型配置；配置入口为 [E00-FrozenDecoder](configs/experiments/E00-FrozenDecoder.yaml) 和 [E00+-FrozenDense](configs/experiments/E00+-FrozenDense.yaml)。复用现有E00与Router初始化权重，无需新下载；本机路径需要在其他机器上调整。600条Plus仅作为pilot，完成情况以独立报告为准。

## 8. LIBERO / LIBERO-Plus 评测入口

### 8.1 已训练的 E 系列：推荐包装入口

先准备训练产物和各自的文本Cache，再执行：

```bash
export OPENWAM_PYTHON="$(command -v python)"
RUN_DIR=/path/to/trained/run
CKPT=checkpoint_step_10690.safetensors

# 包装器会按Encoder/文本/Plus配置准备部署
bash scripts/eval_libero_vjepa.sh "$RUN_DIR" "$CKPT" \
  --benchmark libero --gpus 0,1,2,3 --workers 32 --max-batch-size 8 \
  --libero-path /path/to/LIBERO \
  --render gpu --render-gpus 0,1,2,3 --client-action-queue \
  --output-dir outputs/eval_my_model/libero

TEXT_EMBEDDING_DIR="$PWD/cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text" \
bash scripts/eval_libero_vjepa.sh "$RUN_DIR" "$CKPT" \
  --benchmark libero-plus --gpus 0,1,2,3 --workers 32 --max-batch-size 8 \
  --render gpu --render-gpus 0,1,2,3 --client-action-queue \
  --libero-path /path/to/LIBERO-plus --libero-plus-python /path/to/plus-env/bin/python \
  --output-dir outputs/eval_my_model/libero-plus
```

示例workers/batch仅为起点，不是所有Encoder的最优配置。包装器虽名为 `eval_libero_vjepa.sh`，也负责当前其他Encoder检查和部署配置。`--smoke` 为短程联调，不是正式成绩；`--dry-run` 不启动评测，但包装器可能进行本地部署目录准备。只展开统一评测命令可用：

```bash
python scripts/experiment.py eval "$RUN_DIR" "$CKPT" \
  --benchmark libero --output outputs/eval_preview --dry-run
```

统一 `experiment.py eval` 尚未暴露 `--client-action-queue` 参数；启用新路径请使用上述包装器，或下面持久控制器。底层入口为 [run_eval.py](benchmarks/vla_harness/run_eval.py)，**已经准备好的部署目录**可直接传 `--ckpt-dir`；不要跳过Plus文本准备而把原训练目录当Plus部署目录。

### 8.2 持久评测、动态任务与恢复

```bash
# 两阶段控制器，默认先LIBERO再Plus；每次使用独立输出目录
bash scripts/eval_libero_then_plus.sh "$RUN_DIR" "$CKPT" outputs/eval_pair \
  --gpus 0,1,2,3 --workers 32 --max-batch-size 8 \
  --render-gpus 0,1,2,3 --client-action-queue
```

底层 [run_resumable.py](benchmarks/vla_harness/run_resumable.py) 支持 `--benchmarks libero-plus`、`--dynamic-tasks`、Plus专用并发设置及 `--wait-for-idle`。动态任务领取仅用于单次独立Plus任务；标准LIBERO按整任务分片，保持50次连续trial的RNG语义。重启用同一命令和输出目录，仅补未完成任务。更改开关/配置时开新目录，不修改旧 `job.json` 强行恢复；有 `DONE` 且完整summary才算完成。

持久控制器尚未暴露 `--libero-path` / `--libero-plus-python`：默认从仓库父目录的 `benchmarks/LIBERO`、`benchmarks/LIBERO-plus` 以及现有Conda相邻 `libero-plus` 环境取资源，文本准备也会使用默认Cache布局。新机器先配置等价布局，或修改控制器路径后再运行；需要显式指定任意路径时先使用第8.1节的单次入口。不要将单次入口支持的参数直接传给持久控制器。

### 8.3 协议与结果文件

| 项目 | 标准LIBERO | 本项目LIBERO-Plus |
|---|---|---|
| 规模 | 4 suites，40任务×50 trials=2000 | 7类扰动，10030任务×1 trial |
| 环境随机数 | 任务级原生seed42、连续trial | 每任务新进程，seed10000后首次导入，不额外 `env.seed` |
| 初始化 | 对应trial初始状态，settle30 | trial0初始状态，settle30 |
| 最大控制步 | Spatial/Object/Goal600，Long700 | 同左 |
| 文本 | 标准任务指令 | 默认修正后缀，必须匹配corrected文本Cache |
| 成功率 | 成功数/有效trial，四套件及总体 | 成功数/有效trial，七类别及总体；非类别简单平均 |

E系列默认同步10步去噪、每次执行10个动作，关闭velocity cache；FM/VPP2保留各自原生采样和动作执行协议。LIBERO-PRO、LIBERO-OOD与LIBERO-Plus是不同benchmark，不混用成绩。当前项目协议也不等同于RoboScope默认seed7/settle10/更短horizon。

单次attempt通常含 `run.json`、配置、模型/worker日志、SQLite和summary；持久控制器另有 `status.json`、`attempts/` 和 `DONE`。SQLite/录像/全量日志未发布。部分历史 `result.md` 链接指向服务器本地报告，GitHub上不可点击并不等于不存在原实验。

### 8.4 FM 与 VPP2 需要各自资产

FM选择 `--policy-backend fm-vjepa`，还需 `--fm-python`、`--fm-encoder`、`--fm-text-cache` 以及原生 `.pt` checkpoint、`resolved_args.json`、`dataset_stats.json`；见 [FM说明](benchmarks/vla_harness/FM_VJEPA.md)。

VPP2的源码不随本仓库发布，先按 [固定版本说明](docs/VPP2_LIBERO_PLUS.md) 准备 `third_party/vpp2/` 和 `weights/vpp2/`。至少需要发布的Video/Action权重、config、dataset stats及原生Wan/CLIP/VAE/UMT5/tokenizer资产；以 [validate_assets](benchmarks/vla_harness/vpp2_policy.py) 为准。它不能只拿一个Action文件运行。

```bash
# 权重和独立Plus环境就绪后：仅评测Plus，原32预测/10执行协议
python -m benchmarks.vla_harness.run_resumable \
  --policy-backend vpp2 --benchmarks libero-plus \
  --ckpt-dir weights/vpp2 --ckpt-name libero/action_step030000.pt \
  --gpus 0,1,2,3 --workers 32 --max-batch-size 8 --render-gpus 0,1,2,3 \
  --dynamic-tasks --client-action-queue --output-dir outputs/vpp2_plus_new_runtime
```

本机预设 [eval_vpp2_plus.sh](scripts/eval_vpp2_plus.sh) 是历史128/16配置，不能假定适合所有模型或所有机器。上述新路径尚未完成真实权重闭环收益验收。

## 9. 本次公共评测速度优化：适用于后续模型

2026-11-05的改动属于评测运行时，**不改变模型结构、权重或基准协议**。M29、FM、VPP2已接入同一公共实现。

```text
原路径：每个控制步 → 上传观测 → 服务端返回缓存中的一个动作
新路径：worker逐步执行/保留历史 → 动作耗尽时上传必要观测
                                    ↓
                          原生动态batch推理 → 返回可执行动作块
```

| 公共模块 | 实现 |
|---|---|
| 能力声明、执行/预测长度、历史、batch/observe能力、阶段统计 | [execution.py](benchmarks/vla_harness/execution.py) |
| 块级返回、复用动态batch、episode/控制步校验 | [chunk_server.py](benchmarks/vla_harness/chunk_server.py) |
| worker动作队列、逐控制步历史、重置清理、断线保护 | [client.py](benchmarks/vla_harness/client.py) |
| 模型接入 | [M29](benchmarks/vla_harness/model_server.py)、[FM](benchmarks/vla_harness/fm_model_server.py)、[VPP2](benchmarks/vla_harness/vpp2_model_server.py) |
| 公共统计汇总 | [performance.py](benchmarks/vla_harness/performance.py) |
| CPU通信配对探针 | [profile_evaluation_transport.py](scripts/profiling/profile_evaluation_transport.py) |

启用方式就是在 `run_eval`、`run_resumable` 或其包装命令后加 **`--client-action-queue`**；默认仍保留原路径。新模型继承公共服务并声明能力，原生预处理/归一化/采样留在adapter；必须逐步observe的模型不能直接省略中间观测，不支持原生batch的模型不能假装批量执行。

已完成的验证：**158项通过**，2项已有记录导出API不兼容测试单独排除。真实WebSocket + 替身策略下，多会话合批、历史对齐、episode重置、块中途成功/上限、重放防护均有测试。双相机4会话×60步通信探针中：

| 场景 | 观测请求 | 图像数组payload | 两路径预测输入与动作 |
|---|---|---|---|
| 无历史，每次执行10步 | 240→24 | 90→9 MiB | 一致 |
| 历史[-4,0]，每次执行10步 | 240→24 | 90→27 MiB | 一致 |

这只是通信路径验证，**不等于端到端评测快10倍，也没有证明真实闭环成功率不变**。真实权重GPU闭环、端到端速度仍待配对测量。当前仍逐步生成图像和执行 `env.step`，没有跳过Plus噪声；模拟子进程→worker的逐步Pipe传输也仍存在。

```bash
# 不加载模型/模拟器/GPU的专项探针
MSGPACK_PUREPYTHON=1 python -m scripts.profiling.profile_evaluation_transport \
  --history-offsets=-4,0 --output outputs/transport_probe.json

# 启动评测后，汇总某一attempt的公共统计
python -m benchmarks.vla_harness.performance /path/to/attempt
```

统计在 `performance/*.jsonl`，记录请求往返、实际batch、环境初始化/step、预处理等；CPU wall阶段可能重叠，不是CUDA kernel计时。RuntimeMonitor另采样cgroup CPU、内存与压力。后续自动选择workers/batch、持久性能profile、子进程 `step_chunk`、常驻进程池及MPS **尚未实现**，不能将计划视为本次收益。详细边界见 [实现报告](docs/EVALUATION_RUNTIME_IMPLEMENTATION.md)。

## 10. 变更、测试与进一步阅读

每个阶段改了什么、是否要新权重、配置与入口分别在哪里，见 [CHANGELOG.md](CHANGELOG.md)。旧E01/E02在线配置、E03原始3528-token配置与历史结果仍保留；Cache和Patch变体不覆盖原实验名。

本次公共路径的轻量测试：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MSGPACK_PUREPYTHON=1 \
python -m pytest -q tests/test_evaluation_execution.py \
  tests/benchmarks/vla_harness/test_vla_harness_openwam.py tests/benchmarks/vla_harness/test_vla_harness_startup.py
```

完整158项检查及两项已有依赖不兼容的说明见 [实施验证记录](docs/EVALUATION_RUNTIME_IMPLEMENTATION.md)。部分Encoder测试需真实资产，GPU测试需空闲硬件；不能把缺失权重跳过或CPU替身检查当成完成闭环评测。

| 主题 | 文档 |
|---|---|
| 初始研究边界与统一协议 | [EXPERIMENT_PLAN](docs/EXPERIMENT_PLAN.md)、[protocol.yaml](configs/protocol.yaml) |
| E00+设计、Dense/Sparse对照与验收 | [E00_PLUS](docs/E00_PLUS.md)、[研究讨论](docs/CAUSALWAM_M29_ROUTING_RESEARCH_PLAN.md) |
| E01/E02帧Cache及预测/shift/Loss消融 | [Frame Cache](docs/E01_E02_FRAME_CACHE.md)、[消融过程](docs/E01_PREDICTION_SHIFT_SEARCH.md) |
| E03 VAE/Cache/Patch；E04 Wan2.2 | [E03设计](docs/E03_WAN_VAE_DESIGN.md)、[E03加速](docs/E03_SPEED_OPTIMIZATION.md)、[E04队列](docs/E04_WAN22_QUEUE.md) |
| E05 VGGT-Ω；E06 FLUX.2 VAE | [E05设计](docs/E05_VGGT_OMEGA_DESIGN.md)、[E06设计](docs/E06_FLUX2_VAE_DESIGN.md) |
| E07 DINOv3-L；E08 SigLIP-L；Plus | [E07](docs/E07_DINOV3_L.md)、[E08](docs/E08_SIGLIP_DESIGN.md)、[Plus补评测](docs/E07_E08_LIBERO_PLUS.md) |
| 公共评测、环境与恢复 | [harness](benchmarks/vla_harness/README.md)、[Plus协议](benchmarks/vla_harness/LIBERO_PLUS.md)、[恢复](benchmarks/vla_harness/RESUMABLE.md) |
| 外部模型评测 | [FM](benchmarks/vla_harness/FM_VJEPA.md)、[VPP2](docs/VPP2_LIBERO_PLUS.md) |
| 新Encoder接口 | [examples/encoders](examples/encoders/README.md) |

设计文档保留当时的方案和状态，不会自动回写所有历史段落。当前成绩以 `result.md` 与可核对的summary为准，具体模型以保存的resolved config和identity为准。项目代码许可证见 [LICENSE](LICENSE)，第三方代码与权重分别遵守其原始许可。
