# E05：M29 + RynnBrain 1.1-9B + Frozen VGGT-Ω

更新：2026-10-29。采用 System-Level Fairness，比较 Encoder＋必要适配的整体方案。
固定策略主干、数据、训练预算、loss、评测；不声称严格隔离预训练目标的因果影响。
旧 Register 方案保留在 E05_VGGT_OMEGA_REGISTER_PROPOSAL_SUPERSEDED.md；旧 10 时间位置
设计与测试记录保留在 reports/E05_previous_10time。

## 当前实现及时间语义

实现、配置、权重、Cache、训练输出及评测记录已统一到 `/data/fcy/code/openwam-exp`。E05 已完成10,690步训练及完整评测：LIBERO 73.55%（1471/2000），LIBERO-Plus 47.63%（4777/10030）。后文按日期保留早期设计/验证记录；当前完整结果见 `result.md`。

用户已明确确认 **2 clean＋8 future，共980 tokens**。时间网格为
`[-4,0,4,8,12,16,20,24,28,32]`，历史 `t=-4` 和当前 `t=0` 分别保留为两个 clean。
Future 为 `t=4,8,...,32`，预测覆盖范围不变；没有增加时间融合或额外时序压缩。
Episode 开始处的历史帧由原 reader 在 Episode 内裁剪，与 E01 相同。

E00 baseline 只有一个 clean，但由历史＋当前经 V-JEPA tubelet 得到；E02 也读取历史和当前，
通过差分融合为一个 clean。因此此方案对齐 E00/E01/E02 的可观测输入信息，不能只按 clean 数比较。
E03/E04 没有额外过去帧，这一既有实验差异仍需记录。原 current-only/882 方案代码、配置、
文档和报告已归档至 `reports/E05_previous_9time`，不作为当前 E05 主实验。

## 实际 Pipeline

```text
RGB: [B,3,10,224,448]（reader 接口的宽图）
 → 拆成独立 Front/Wrist，固定 Front 第一、Wrist 第二
 → [B*10,2,3,224,224]（原生 sequence=双视角；时间=独立 batch）
 → 冻结官方 Aggregator，内部保留 Camera、16 Registers、全部 Frame/Inter-frame Attention
 → 最终层 [B*10,2,213,2048]（17 个特殊 token + 196 个 spatial patches）
 → 只取空间 patches；冻结的非仿射逐 token LN，eps=1e-5
 → 按各视角 row-major 还原，沿宽度拼接 [B,2048,10,14,28]
 → 原有 scheduler：历史和当前两个 clean 保持干净，未来加噪
 → 原有 BottleneckPatchEmbed (1,2,2)，future bottleneck=256
 → [B,1536,10,7,14] → [B,980,1536]
 → 原有 30 层 M29 + Action Expert
```

原生 Aggregator 不接收横向拼接 RGB，也不分别调用两次单图像 Encoder。
宽度拼接只发生在编码后的空间特征输出端，遵循用户追加的布局要求。
每视角 14 列可被 2×2 patch 整除，所以投影不会跨相机边界取同一个空间 patch。
Camera token 的第一/后续图像身份不等同于机器人相机语义，固定顺序并验证输出映射，
不声称交换两个视角后具有置换等变性。

## 权重、参数与训练机制

- checkpoint：`encoder/vggt-omega/vggt_omega_1b_512.pt`，严格加载官方 aggregator 前缀。
- SHA256：`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`。
- 使用真实 512 权重在 224 输入做分辨率迁移，不能标为 416-Reproduction。
- 冻结 aggregator：907,918,336 参数；Encoder 可训练参数 0，eval/no_grad。
- 总参数 2,832,602,192；可训练 1,924,683,856（完整模型实际构建并应用正式 freeze 规则）。
- 原有输入 Projection 15,076,352，输出 Head 12,591,104；总计 27,667,456。
- 不另加 Cross-Attention、Gate、显式 Register Projection、几何 head 或 loss。
- Projection 没有 VGGT 预训练权重，跟随原有 M29 from-scratch 策略训练。
  相比 E01 必要的输入/输出通道映射多 10,226,688 参数，必须记录为适配差异。
- RynnBrain 保持原有冻结 text-only avg_all_layers 离线文本 Cache，不计入当前常驻模型参数。
- batch256、4卡每卡64、seed42、10 epochs/预期10,690 steps、AdamW/LR 等共享协议不变。

复用原 `prepare_inputs/compute_loss`、x-prediction 转换、scheduler、mask 和 action loss。
监督目标是冻结特征，不是可训练 Projection 后的特征，因此不会随投影优化而漂移。
未来真实特征仅进入加噪路径和监督目标，clean 仅历史和当前，且排除 clean 的 video loss。
修改未来像素不改变两个 clean；Projection 和 M29 Q 投影均有有限非零梯度；Encoder 无梯度。
两步 smoke 只能检查有限值和更新路径，不代表收敛或训练性能结果。

## Cache

`feature_cache_dir=null` 默认在线。设置该项后，按每个时间点的有序双视角输入缓存，
保存冻结 Encoder 输出（含固定 LN），不缓存可训练 Projection。部署重载默认禁用此缓存，在线处理可观测的历史与当前 RGB。

身份命名空间包含 checkpoint 文件 SHA256、分辨率、归一化、相机顺序、宽度布局和同时间上下文模式；
内容键覆盖精确预处理 RGB、shape、dtype。相同 Episode/时间输入相同即可复用，完全相同像素跨 Episode
复用也不改变语义；不同随机增强不会错误命中。当前主实验 color jitter 关闭。
进程锁防止重复冷编码，临时 safetensors 原子替换，按需读取，FP16/BF16 存储。

真实相邻两个窗口请求 20 个时间点，11 个唯一缓存、9 次复用；在线、填充、读回逐元素一致。
每时间点 1,605,632 字节≈1.53 MiB，按全数据帧数估计约 408.93 GiB；
如果按窗口重复存十份则约 4089.29 GiB。只生成 16.8 MiB probe，没有全量构建。
实际总量取决于像素去重、增强以及解码一致性；409 GiB 是确定性输入估算，不是已经生成的容量。

CPU 四线程：两个窗口在线 12.23s，冷填充 6.99s，热读 0.220s，约139 MiB/s。
当前接入仍需解码 RGB 计算内容键，并未消除解码成本。预填充可使用独立 `StereoFeatureCache`
与 Encoder 接口；`profile_e05_cache.py` 固定只检查两个窗口，不能误触全量数百 GiB 构建。
GPU 在线/缓存数值差异和正式 batch I/O 吞吐尚未测量，不能用 CPU 结果估计四卡 step time。

## 验证与开销

- E05 专项：7 项，包括 joint stereo/width mapping、无未来泄漏、冻结、缓存、并发、分辨率网格、
  Projection/M29 反传及 tiny checkpoint 严格重载、LIBERO/Plus 接口。
- 真实 checkpoint＋真实 LIBERO reader＋完整 30 层 M29/Action Expert：CPU batch1 两步训练通过。
- 真实完整 checkpoint 写入临时目录后，经正式部署 loader 严格重载成功，临时权重随后删除。
- 重载 Encoder 特征、Train/Eval clean 一致，最大绝对差 0；原生部署输出 [32,10] 有限动作。
- LIBERO-Plus 的 checkpoint 准备路径与公共 contract 已验证，没有执行完整 benchmark。
- 当前历史条件回归：113 passed、2 skipped（GPU 集成测试因四卡占用未运行）。
- 上一版扩展回归的5个旧失败已在完全未修改主工作区复现，原始日志保留于归档；
  本次没有为 E05 修改不相关代码。

完整模型 CPU 四线程 batch1：Encoder 10 时间点 5.60s；
第二步 M29/Action Forward 2.45s、Backward 4.84s；
1536 维空间 Projection 独立测量 0.024s。
包含保存/重载的进程峰值 RSS 30.88 GiB。
这些是 CPU 时间和主机内存，不是 GPU step time 或 GPU 显存。正式四卡 batch64 的稳态吞吐、
峰值显存和推理延迟尚未测量：E04 占用四卡，未并发运行 GPU 测试。
本次 smoke 使用2步 AdamW 验证有限值、梯度和更新，不以 loss 单调性判断收敛，
也不替代正式带 warmup/scheduler 的训练表现。

## 文件与复现

新增/修改重点：

- `openwam/model/video_backbone/encoder/vggt_omega.py`、`vggt_feature_cache.py`、registry import。
- `third_party/vggt_omega/`：本地原版代码副本，Aggregator 未修改。
- `configs/model/video_backbone/encoder/vggt_omega.yaml`、`configs/experiments/E05.yaml`、实验/资产注册。
- `benchmarks/vla_harness/checkpoint_contract.py`、`run_eval.py`、`prepare_libero_plus_svae.py`：原协议入口适配。
- `tests/test_vggt_omega_encoder.py`、`scripts/smoke_e05_spatial.py`、`scripts/profiling/profile_e05_cache.py`。

```bash
cd /data/fcy/code/openwam-exp
PY=/data/haoyang/miniconda3/envs/openwam/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$PY" -m pytest tests/test_vggt_omega_encoder.py -q
CUDA_VISIBLE_DEVICES='' "$PY" scripts/smoke_e05_spatial.py --full-policy --steps 2
CUDA_VISIBLE_DEVICES='' "$PY" scripts/profiling/profile_e05_cache.py
"$PY" scripts/experiment.py train E05 --dry-run
```

`checkpoint` 与 `image_size` 可配置；wrapper 的 512 网格接口已用结构测试覆盖，
但 reader 和评测画布当前固定224，切换分辨率必须同步调整数据/评测输入并重新验证，
不能仅改 image_size 就宣称官方分辨率全流程可直接训练。416 权重仍未取得，不用512权重冒充416版。

主工作区代码/配置 source hash 保持
`bd7418b6255a065544860bc635840a0044ff60385166d9289fa146db8978dab8`。
E05 已有完整 CPU smoke 条件；正式训练前仍建议完成独占 GPU 容量/稳态短测，
实际 cache 还未全量生成。没有自动训练、评测、或停止 E04。


## 正式训练自动队列（2026-10-26晚）

E04已完整结束，LIBERO 94.50%（1890/2000），LIBERO-Plus 76.02%（7625/10030），
结果及全部套件/类别明细已写入主工作区result.md。用户已明确授权随后开始E05正式训练。

历史pipeline现位于 `reports/E05_training/pipeline.py`，
主状态文件 `/data/fcy/code/openwam-exp/reports/E05_training/status.json`，
顺序为：完整冻结特征Cache → 完整模型四卡batch64短测4步 → 从头正式训练。
Global batch256、seed42、10epochs/10,690steps不变；不从smoke checkpoint暖启动。

实际训练缓存目录：`/data/fcy/code/openwam-exp/cache/vggt_omega_libero_stereo`。
新脚本 `scripts/precompute_vggt_features.py` 用原reader的单相机解码与原布局函数，
每实际时间点联合编码Front/Wrist一次，跳过重复的未来窗口解码与action/parquet处理；
四卡并行、CPU预取、线程异步原子写盘，按精确RGB内容键复用特征。

已验证单帧读取与完整窗口当前图像逐像素相同；四卡64帧probe和包含两个clean的完整窗口
验证Cache/Online逐元素一致。全量cache完成后每rank再验证两个完整窗口均命中缓存，
一致性或容量/梯度检查失败即停止后续阶段，保持原checkpoint与训练参数。

当前正在生成全量缓存；只有正式训练launch记录出现后才算已开始正式训练。
不启动E05完整benchmark，也不启动E06。


2026-10-26 23:52更新：缓存生成阶段发现缺少跨卡完成同步，最快rank提前验证了仍未写完的
其他rank尾帧。修复为所有异步writer关闭后进行Gloo barrier；按已有索引检查原子发布文件续建，
保留25.5万帧已有特征。四卡新建probe、全命中续建probe、全量273,465帧及完整窗口校验均通过。

正式batch的四卡4步smoke通过，最后一步约6.59s，PyTorch峰值分配约30.04GiB/卡；
实际正式训练已从头启动于 `/data/fcy/code/openwam-exp/outputs/m29_rynn9b_vggt_omega_spatial/train/s42/2026-10-26_23-52-12`。
当前参数保持2 clean/8 future、980tokens、global batch256、seed42、10,690steps，
不从smoke checkpoint warm-start。原GPU pending/缓存生成状态仅为历史阶段记录。
