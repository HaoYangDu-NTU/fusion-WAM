# Encoder 对比与适配分析

更新：2026-10-26。以下结论来自实际代码、已保存配置与 checkpoint，不将工程适配
统称为 encoder 的“原生特性”。已有实验成绩保持原记录，不回写成新的对照实验。

## 0. Clean 数量与历史观测的区别

这里的 baseline 指本项目 E00（M29 + RynnBrain + V-JEPA），不是把所有 Encoder
都假定为同一种时间压缩。Clean 是不加噪的条件 latent 数，不能直接当作原始观测帧数。

| 配置 | 部署可见的原始观测 | Clean latent | Future latent |
|---|---|---:|---:|
| E00 baseline | t=-4、0，V-JEPA tubelet 编码 | 1 | 4 |
| E01 | t=-4、0，分别保留 | 2 | 8 |
| E02 | t=-4、0，历史差分融合 | 1 | 4 |
| E03/E04 | 只有 t=0；没有额外过去帧 | 1 | 8 |
| E05 用户确认的主实验 | t=-4、0，分别保留 | 2 | 8 |

用户已明确选择 E05 保留历史＋当前两个 clean，八个 future，共980 tokens。
时间网格为 [-4,0,4,8,12,16,20,24,28,32]，对齐 E00/E01/E02 的原始观测信息，
不为了保持单 clean 添加额外时间融合。先前 current-only/882 方案已归档至隔离副本
reports/E05_previous_9time。单 clean 并不自动意味着单帧，E00 和 E02 都包含历史。

## 1. 同一 M29 主干不等于严格的 encoder-only 对照

E01/E02 固定了 M29 Transformer、RynnBrain 1.1-9B 文本条件、action backbone、
LIBERO 数据来源、训练预算、优化器、loss 定义和闭环评测协议。
但是两者的条件组织、未来时间压缩、token 数和 encoder 外可训练参数不同。
因此当前成绩可以比较完整 representation 接入方案，不能把差异全部归因于预训练 encoder。

| 项目 | E01 DINOv3 | E02 I-JEPA | 新 E05 VGGT-Ω spatial-only |
|---|---|---|---|
| 输入相对时间 | -4,0,4,8,12,16,20,24,28,32 | 同 E01 | 同 E01 |
| 每视角 RGB | 224×224 | 224×224 | 224×224 |
| Encoder 内跨视角 | 两个视角独立 | 两个视角独立 | 每个时间位置双视角联合 |
| Encoder 内跨时间 | 无 | ViT 无跨时间；之后增加适配 | 无；时间只作为独立 batch 轴 |
| Clean | history/current 各一个，共 2 个 | current + alpha·P(current-history)，共 1 个 | 历史和当前各一个，共2个 |
| Future | 8 个独立位置 | 每两帧平均，4 个位置 | 同 E01，8 个位置 |
| 原始 feature C×T×H×W | 1280×10×14×28 | 1408×5×14×28 | 2048×10×14×28 |
| 空间 Patch | 2×2 | 2×2 | 2×2 |
| M29 tokens | 980 | 490 | 980 |
| 输出 feature norm | 非仿射 LN，eps=1e-5 | 可训练仿射 LN，eps=1e-5 | 同 E01，非仿射 LN |
| 空间输入 Projection | clean 直接 Conv；future 256 bottleneck | 同机制 | 同机制 |
| 额外 encoder-side 训练参数 | 0 | 差分投影、Gate、仿射 LN | 0 |

DINOv3 和 I-JEPA 都是图像 encoder；逐帧读取 spatial features 符合其接口。
E02 的历史差分 Gate 与 future pair mean 是工程设计，不是 I-JEPA 原生的视频时序能力。
E01 保留历史与当前、不做未来平均，也是一种接入选择，而非唯一原生协议。

## 2. OpenWAM_v2 是否对不同 encoder 做适配

有。检查的是 `/data/fcy/code/OpenWAM_v2/openwam/model/video_backbone/encoder/`。
这个目录提供统一 VideoEncoder contract，但没有强制各 encoder 的内部适配完全相同。

- DINOv3：逐帧编码、去 CLS/registers、恢复空间网格、首帧保留而后续每 4 帧平均，
  再做非仿射 LN。默认使用普通输入 Conv3d 和输出 Linear，没有当前 E01 的可选不池化、
  history/current 双 clean、独立相机与 clean/future bottleneck 接入。
- V-JEPA 2.1：条件视频和未来视频分别编码，使用原生 tubelet，再根据配置进行额外
  时间平均、视角拆分、特征归一化和输入 bottleneck；还支持不同特征层融合选择。
- Wan VAE：使用原生因果 VAE latent，保留首帧、后续约 4 倍时序压缩。
- 指定目录中没有 I-JEPA 实现；E02 不是从这里的现成 I-JEPA 路径直接沿用。

原版 DINO wrapper 本身不拆相机；如果传入的是宽度拼接 RGB，会当作一张图编码。
原版 launcher 当前默认值还包含 AdaFuse 等可选实验设置，不能把默认脚本无条件等同于
最初 M29 baseline。本项目 E00 明确选用 last_layer / last_layer，并关闭额外平均池化。

代码依据：

- [原版 DINOv3](../OpenWAM_v2/openwam/model/video_backbone/encoder/dinov3.py)
- [原版 V-JEPA](../OpenWAM_v2/openwam/model/video_backbone/encoder/vjepa21.py)
- [原版 Projection](../OpenWAM_v2/openwam/model/video_backbone/encoder/bottleneck.py)
- [当前 E01](configs/experiments/E01.yaml)、[当前 E02](configs/experiments/E02.yaml)

## 3. 当前 Wan VAE 与 OpenWAM_v2 原版 Wrapper 的差别

| 项目 | OpenWAM_v2 默认 Wan VAE Wrapper | 当前 E03/E04 |
|---|---|---|
| 加载内容 | 完整 VAE，可编码与解码 | 只加载 encoder、conv1，删除 decoder/conv2 |
| 相机布局 | Wrapper 直接编码传入视频，不主动拆相机 | 两个 224×224 视角独立编码，再拼 latent |
| 原生时序压缩 | 因果首帧 + 后续约 4 倍压缩 | 同类原生机制；配置明确为连续 33 帧 |
| 输入 Projection | 所有时间位置共享普通 Conv3d | clean 直接投影，future 使用 256 bottleneck |
| 空间 Patch | 默认 2×2 | 原始 E03 为 2×2；已评测 E03-Patch 为 4×4；E04 为 2×2 |
| 离线 latent Cache | 该 Wrapper 没有对应实现 | 保存原始冻结 latent，并做身份/完整性校验 |
| 像素解码 | 支持 | 不支持，控制任务只使用动作输出 |

删除未使用 decoder、离线 Cache 属于计算/存储优化；Cache 数值等价只针对当前相同输入
和编码路径，不意味着与原版不同视角布局的整条路径数值等价。
独立视角编码、clean/future 分支 Projection 和 E03-Patch 的 4×4 Patch 都是表示/结构差异。
即使 E04 也使用 2×2 Patch，其接法仍不等同于原版默认 VAE Wrapper。

E03-Patch 和 E04 都保留 9 个 latent 时间位置：1 个当前条件 + 8 个未来位置，
不是 9 个历史位置。Wan2.1 空间下采样 8 倍后接 4×4 Patch，Wan2.2 下采样 16 倍后
接 2×2 Patch，均得到每视角 7×7，共 882 tokens；这不消除预训练压缩方式和通道差异。

代码依据：[原版 Wrapper](../OpenWAM_v2/openwam/model/video_backbone/encoder/wan22_vae.py)、
[当前 Wrapper](openwam/model/video_backbone/encoder/wan_vae.py)。

## 4. Projection 与可训练参数的实际差别

E01/E02 数值从各自最终 checkpoint 的 tensor shapes 与 freeze prefixes 统计；
E05 额外使用正式维度的完整模型在 meta device 上构造，并执行训练器同一 freeze_modules
接口，核实实际参数量，不分配 GPU。不包含 encoder 的 buffer 为空。

| 项目 | E01 | E02 | E05 |
|---|---:|---:|---:|
| 输入 Projection + 输出 Head | 17,440,768 | 19,145,216 | 27,667,456 |
| Encoder 外的额外适配参数 | 0 | 1,985,281 | 0 |
| 非冻结 tensor 数 / E05 实际可训练参数 | 1,914,457,168 | 1,918,146,897 | 1,924,683,856 |

E02 额外参数包括 1408×1408 差分投影、1 个 gate 与 2816 个仿射 LN 参数。
Projection 机制相同不意味着参数量完全相同：原始通道不同，必要的输入/输出投影大小不同。
这与另加 Cross-Attention Adapter 是两种性质不同的变化。

## 5. E05 的当前决策与验证边界

主实验仅使用冻结 VGGT-Ω 最终聚合层空间 Patch Features；内部 camera tokens、registers
和 frame/inter-frame attention 原样保留，不额外向 M29 暴露 registers。
删除显式 Register Cross-Attention、Q/K/V/Out Adapter、Gate 与额外 register 条件支路。
无 Anchor-Pair、跨时间前缀、额外 loss 或几何监督；E05-R 只作为未来可选消融名称保留。

E05 采用系统公平：固定策略主干、数据、预算、loss、评测，允许合理的 encoder 适配。
复用 E01 的 Projection/target 机制和时间网格，保留过去帧与当前观测。与 E02 的历史差分、
未来平均、LN 可训练性、490 tokens 不会被描述为已经对齐。
E05 的双视角联合编码也必须报告为区别于 E01/E02 独立相机编码的因素。

旧 Register 融合设计保留为历史方案，不进入主实验。
为避免改变正在运行的 E04 源码指纹，实际实现与测试位于隔离副本：
`/data/fcy/code/openwam-exp`，主工作区只新增文档与状态索引，没有替换运行中代码。

真实 VGGT-Ω 512 权重、真实 LIBERO reader、CPU batch1 的缩小策略及完整 30 层策略
已分别完成两步前向/反向 smoke；并验证了固定 target、投影更新、encoder 无梯度和未来扰动不影响 clean。
**完整策略已在 CPU batch1 验证，但这不是四卡 batch64 的训练开销测试。** E04 仍占用四卡，未并发运行 GPU smoke，
也未启动完整 E05 训练。当前 512 checkpoint 不能标为尚未取得授权的 416-Reproduction。

对应代码及原始检查：

- [E05 Encoder](openwam/model/video_backbone/encoder/vggt_omega.py)
- [E05 Profile](configs/experiments/E05.yaml)
- [真实 CPU smoke](reports/E05_full_policy_cpu.json)
- [参数审计](reports/E05_parameter_audit.json)
- [回归检查](reports/E05_regression_tests_cpu_only.log)


## 6. E05 双视角输出与缓存验证（2026-10-26）

用户明确选择保留 VGGT 原生双视角联合编码，输出继续沿用 E01–E04 的宽度拼接。
[front_t,wrist_t] 为原生 sequence 轴，不同时间为独立 batch；最终空间 patch 各自按
row-major 还原，再按 front 左、wrist 右拼成 [B,2048,10,14,28]，原有 2×2 Projection
得到 [B,1536,10,7,14]，共 980 tokens。Camera/Registers 仅在原生网络内部参与聚合。

Cache 保存冻结输出（含冻结的非仿射 LN），不保存可训练 Projection。身份包括权重 SHA256、
分辨率、归一化、固定双视角顺序、同时间上下文和布局；每条内容键覆盖确切预处理 RGB，
所以相同输入可跨重叠窗口复用，随机增强不同则不会错误命中。读写有进程锁和原子发布。
这条接入仍需解码 RGB 以确定内容键，不宣称消除了数据解码成本。

真实两个相邻窗口请求 20 个时间点，只保存 11 个唯一双视角特征；BF16 读回和在线输出
逐元素一致。每时间点 1,605,632 字节，按 273,465 帧估计约 408.93 GiB，未去重按窗口
存十份则约 4089.29 GiB。只生成约 16.8 MiB 验证缓存，没有全量生成。
CPU 四线程实测见隔离副本 reports/E05_cache_profile_cpu.json；GPU 正式 batch 吞吐仍待验证。

扩展回归中的 5 个失败已在完全未修改的主工作区复现：4 个旧 meta 测试的伪文本 cache 与
其选用的 RynnBrain2B/Cosmos3 类型不匹配，以及 1 个旧 Wan2.2 CPU 测试要求严格零容差、
实际约 2e-7 的卷积舍入差异。没有为 E05 修改这些不相关代码。E05 专项测试通过。

2026-10-26 补充：用户确认历史＋当前2 clean/8 future后，完整30层 CPU 两步训练、真实
checkpoint 严格重载、Train/Eval clean 一致性及 [32,10] 有限动作再次验证通过。
当前历史条件回归113 passed、2 skipped（GPU集成）；保持原M29/Action/训练预算不变。


## E06 FLUX.2 VAE 方案（2026-10-26）

已按用户要求记录E06方案，详见 [E06_FLUX2_VAE_DESIGN.md](docs/E06_FLUX2_VAE_DESIGN.md)。
主实验建议复用E05已确认的 [-4,0,4,...,32]，2 clean＋8 future，980 tokens；
FLUX.2作为原生2D图像VAE逐帧、逐视角编码，空间latent沿宽度拼接，保留原packing/固定BN，
使用原M29 clean直接Projection/future bottleneck256，不复制原Wrapper的人为四帧平均。
同一baseline固定的是策略/数据/预算/loss/评测，不强制FLUX模拟Wan原生时序压缩。

资产下载到 encoder/flux2-vae。dev仓库当前401，实际来源为官方公开FLUX.2-klein-4B的vae组件，
BF16、明确revision和SHA，不标记为flux2.2或已验证相同的dev FP32原件。
提取Encoder-only 68,863,720字节，34,425,856参数；真实加载、严格重载、有限特征和固定BN已验证。
规划去重缓存约25.56GiB；E06完整训练集成和全量Cache尚未实现或启动。


2026-10-26晚：E04完整评测和result记录完成（LIBERO94.50%、Plus76.02%）。
用户已授权E05正式训练。E05四卡GPU缓存probe通过，后台pipeline正生成按帧去重冻结特征Cache，
随后自动进行四卡每卡64的4步容量/反传检查，通过后从头训练10,690steps；
主状态见 reports/E05_training/status.json。仍是2 clean＋8 future/980 tokens，
M29/RynnBrain/action/loss/batch预算不变，未授权E06训练。


E05后续：缓存尾部验证因跨rank缺少同步曾中止，已补全局完成屏障和索引续建，未改变表示。
273,465帧已完成，GPU完整窗口Cache/Online逐元素一致；四卡正式batch64的4步smoke通过。
E05正式训练已于2026-10-26 23:52从头启动，目录见result.md E05节。
