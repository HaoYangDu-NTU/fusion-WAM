# 实验与工程变更记录

更新：2026-11-05。按本仓库可核对的实验阶段记录“改了什么、入口、配置、权重与结果”，不是将历史文档中所有待办视为已完成。40个配置的逐项索引见 [README](README.md)，完整分项成功率见 [result.md](result.md)。

## 2026-11-12：E00+冻结主干与FP32 Router

- 增加Decoder-only与Dense Router冻结主干对照配置及[统一入口](scripts/e00_plus_frozen_router.py)，两组均已完成1000步微调；记录见[实验报告](docs/E00_PLUS_FROZEN_ROUTER_20261112.md)。
- 增加FP32 Gate、FP32残差计算和最终动作投影的dtype兼容处理；默认配置保留原实验行为。
- 支持显式可训练参数白名单，以及从微调checkpoint保留完整模型配置，防止当前profile默认值改变冻结表示路径。
- 增加独立600条Plus pilot入口与[GPU空闲等待队列](scripts/e00_plus_frozen_router_wait_eval.py)，保留已有权重和原始实验产物。
- 补充B0/B1/B2闭环记录、公平性诊断及Gate/Attention分析文档。

## 2026-11-10：E00+同源权重微调pilot

- 新增 [入口](scripts/e00_plus_finetune.py) 和 [配置](configs/e00_plus_finetune.yaml)，不改变原从头训练的E00/Dense/Sparse配置。
- 复用已核验的M29 + RynnBrain9B + V-JEPA E00 step10690。只追加Router参数，2151个共有tensors逐项相等，B1/B2初始化权重相同，保持严格checkpoint加载。
- B0/B1/B2各1000步、LR1e-5、global batch256；稀疏项100步warmup、400步ramp。有限队列含正式batch的GPU smoke、初始B0及三组微调后标准LIBERO；等待现有GPU任务结束，不自动启动Plus。
- 真实权重CPU严格训练加载、有限值和B2生产部署加载通过；20项定向回归通过。GPU训练及闭环收益以队列实际结果为准。
- 详情：[E00+设计与微调协议](docs/E00_PLUS.md)。

## 2026-11-05：公共评测运行时 F0/F1

- **改动**：从服务端逐步返回缓存动作，增加可选的 worker 本地动作队列；历史按实际控制步更新，动作耗尽时才请求下一块。保留原生动态 tensor batching。
- **模型覆盖**：M29、FM-VJEPA、VPP2共用能力声明、服务和客户端；不同执行长度/历史要求留在adapter声明。未来模型不必复制调度代码。
- **可靠性**：episode token/控制步校验、重置清空缓存、对模拟器复用图像buffer保留独立历史、禁止块级会话断线后无状态重放。
- **统计**：环境初始化/step、观测准备、请求、模型调用、真实batch与OOM计数；增加cgroup CPU配额及CPU/内存/IO压力采样；按进程累计快照汇总。
- **入口**：`run_eval` / `run_resumable` / 包装脚本加 `--client-action-queue`。`scripts/experiment.py eval` 尚未透传此开关。
- **代码**：[execution.py](benchmarks/vla_harness/execution.py)、[chunk_server.py](benchmarks/vla_harness/chunk_server.py)、[client.py](benchmarks/vla_harness/client.py)、[performance.py](benchmarks/vla_harness/performance.py)。
- **权重**：不需要新增权重或重新训练；现有checkpoint和冻结Encoder继续使用。此开关属于新运行配置，不能无记录混入旧续跑任务。
- **验证**：158项通过；2项已有记录导出API不兼容测试排除。真实WebSocket/替身策略通信探针在240控制步下，将请求240→24；预测次数、预测输入和动作一致。真实权重端到端速度与闭环差异尚未测量。
- **未实现**：自动并发搜索、性能profile持久化、模拟子进程 `step_chunk`、跨任务常驻池、MPS。Plus子进程逐步传图仍存在。
- **文档**：[第一批实现报告](docs/EVALUATION_RUNTIME_IMPLEMENTATION.md)、[优化方案](docs/ROBOSCOPE_EVALUATION_SPEED_PLAN.md)。

## 2026-11-04～05：VPP2 原生评测接入与完整 Plus

- **改动**：增加 VPP2 原生资产/配置加载、图像与本体状态预处理、video prefill + action Euler的原生tensor batch、OOM拆批与顺序保持。文本编码器在cache miss时移到GPU、完成后回CPU；这不是已实现的全量离线UMT5缓存。
- **入口**：[eval_vpp2_plus.sh](scripts/eval_vpp2_plus.sh)、`run_eval/run_resumable --policy-backend vpp2`；真实权重短测为 [profile_vpp2_batch.py](scripts/profile_vpp2_batch.py)。
- **代码**：[vpp2_policy.py](benchmarks/vla_harness/vpp2_policy.py)、[vpp2_batch.py](benchmarks/vla_harness/vpp2_batch.py)、[vpp2_model_server.py](benchmarks/vla_harness/vpp2_model_server.py)。
- **权重**：需要VPP2自己的Video step10,000 + Action step30,000、原生Wan/CLIP/VAE/UMT5/tokenizer与stats；与E系列冻结Encoder和RynnBrain不同。
- **协议**：预测32、执行10；10步Euler；action/video noise seeds42/43；两相机原生处理，8维state，夹爪只转换一次。
- **结果**：完整Plus 8,342/10,030=83.17%，无无效记录。四卡、128 workers/max batch16的历史尝试含216条迁入任务；实际长期microbatch8，不能把它作为新客户端路径的速度或成功率结果。
- **文档**：[VPP2配置与验证](docs/VPP2_LIBERO_PLUS.md)。

## 2026-11-04：E00+-Dense / E00+ 动作读出实验

- **实验**：B0=E00；B1=E00+-Dense；B2=E00+。新增action-conditioned clean-visual residual readout，Dense与Sparse只改变正则权重。
- **结构**：Joint loop后、action decoder前；Q为action，K/V为最终clean visual；width512、8 heads、top-k32、有效gate初值0.01。
- **损失**：Sparse版增加top-k attention mass正则，optimizer step1000后渐增，5000时λ=0.01；保留稠密主路径，不硬裁剪token。
- **代码/配置**：[action_router.py](openwam/model/architectures/utils/action_router.py)、[E00+.yaml](configs/experiments/E00+.yaml)、[Dense.yaml](configs/experiments/E00+-Dense.yaml)。
- **入口**：`scripts/experiment.py train 'E00+'` / `'E00+-Dense'`；真实资产验收 [smoke_e00_plus.py](scripts/smoke_e00_plus.py)。
- **权重**：复用E00冻结V-JEPA/RynnBrain；Router从头训练。旧E00 checkpoint不含Router参数，不能当作E00+精确恢复；无已验收通用warm-start迁移工具。
- **状态**：CPU真实资产、梯度、严格重载检查已完成；没有正式B0/B1/B2闭环收益结论。
- **文档**：[E00+实施与验收](docs/E00_PLUS.md)、[研究讨论](docs/CAUSALWAM_M29_ROUTING_RESEARCH_PLAN.md)。

## 2026-11-02～03：E07 DINOv3-L、E08 SigLIP-L 与 Plus补评测

- **E07**：冻结DINOv3-L/16，24层、1024维；保留独立双视角、2 clean+8 future、980 tokens；新增独立DINO-L Cache身份。
- **E08**：原版SigLIP-L/16-256，原生256输入；纯视觉Patch输出，不用MAP全局池化和文本塔；2 clean+8 future、1280 tokens。新增native stereo reader/transform、独立空间Cache身份、部署尺寸/sidecar重建，旧Encoder默认224保持兼容。
- **配置**：[E07](configs/experiments/E07.yaml)、[E07-Cache](configs/experiments/E07-Cache.yaml)、[E08](configs/experiments/E08.yaml)、[E08-Cache](configs/experiments/E08-Cache.yaml)。
- **入口**：[e07_train_eval_queue.py](scripts/e07_train_eval_queue.py) 支持 `--experiment E07/E08`；Plus补评测为 [e07_e08_plus_queue.py](scripts/e07_e08_plus_queue.py)。统一训练入口同样可用。
- **权重**：E07需要准确DINOv3-L；E08需要准确原版SigLIP-L-256及纯视觉导出，不能复用本地SigLIP-B或SigLIP2冒充。
- **结果**：E07 LIBERO95.00%、Plus70.54%；E08 LIBERO94.40%、Plus67.16%，均完成。旧文档中的等待前序/待评测段落保留为历史流程。
- **文档**：[E07](docs/E07_DINOV3_L.md)、[E08](docs/E08_SIGLIP_DESIGN.md)、[Plus](docs/E07_E08_LIBERO_PLUS.md)。

## 2026-10-30～11-02：E01 Cache、Prediction/Shift/Loss消融

- **Cache**：E01/E02增加独立 `-Cache` 配置，按实际时间戳去重保存冻结原始grid，mmap shard读取、worker预取、异步原子写盘；可训练Projection和E02历史差分适配仍在线执行。原在线配置不覆盖。
- **入口**：[precompute_image_features.py](scripts/precompute_image_features.py)、[validate_image_features.py](scripts/validate_image_features.py)、[prepare_e01_p1_cache.py](scripts/prepare_e01_p1_cache.py)。生产batch64验收与Cache identity绑定。
- **预测消融**：P1=x/v，P2=v/x，P3=v/v，原E01=x/x；增加future质量诊断。P1 91.65%、P2 92.25%、P3 90.65%，标准LIBERO。
- **运行调整**：在线P1转独立Cache流程；P2旧step1957运行停止后，使用 `E01-P2-Cache-W2` 从头重启。`W2`限制每个DataLoader worker为2线程，防止线程过量，不改变模型。
- **Shift**：固定x/x，video shift24/60/12，action shift5；标准LIBERO依次91.20%、92.75%、83.95%，shift42复用原E01的94.35%。配置中的其他x/v组合不代表全部跑过。
- **Loss**：λ_video0.5取得91.85%；λ_video2已取消，未训练。λ_action保持1；λ_video1复用原E01。日志video loss已加权，不能直接与原始未加权损失混比。
- **配置**：[完整注册表](configs/experiments.yaml) 中所有 `E01-P*`、`E01-S*`、`E01-LV*` 及Cache/W2变体；本页不重复将所有可执行名当成新结果。
- **调度**：[e01_ablation_queue.py](scripts/e01_ablation_queue.py)；支持限定单一实验、独立结果标签、前序完成校验、独立GPU锁与自己的full-state恢复。
- **权重**：全部复用原DINOv3-H+与RynnBrain，配置变化不要求下载新Encoder；每组策略从头训练，不能复用其他组最终checkpoint冒充自身结果。
- **文档**：[Cache边界](docs/E01_E02_FRAME_CACHE.md)、[完整消融过程](docs/E01_PREDICTION_SHIFT_SEARCH.md)。

## 2026-10-25～29：E03 Cache/Patch、E04–E06 与工作区统一

### E03：Wan2.1 VAE

- 保留原E03在线3528 tokens；E03-Cache只替换冻结编码计算；E03-Patch增加空间4×4 Patch，将token预算降至882，属于结构变化。
- 33连续帧原生时序编码，每窗口重置时序状态；1 clean+8 future。Cache存归一化冻结latent，不存训练投影。
- [配置](configs/experiments/E03-Patch.yaml)、[构建器](scripts/precompute_wan_latents.py)、[设计](docs/E03_WAN_VAE_DESIGN.md)、[性能实现](docs/E03_SPEED_OPTIMIZATION.md)。
- 只需Wan2.1 VAE权重，不需整套14B视频模型；已报告95.35%/66.93%来自E03-Patch，弃用原训练不混入结果。

### E04：Wan2.2 VAE

- 原生因果VAE48通道、1 clean+8 future、空间2×2 Patch、882 tokens；独立原始latent Cache。
- [配置](configs/experiments/E04.yaml)、[流程](docs/E04_WAN22_QUEUE.md)，构建复用 `precompute_wan_latents.py --experiment E04`。
- 需Wan2.2 VAE权重；完整LIBERO94.50%、Plus76.02%。

### E05：VGGT-Ω

- 官方512 checkpoint的冻结aggregator，在转移分辨率224下逐时间联合Front/Wrist，不把时间串进空间聚合器；2 clean+8 future、980 tokens。没有额外显式register注入。
- [配置](configs/experiments/E05.yaml)、[构建器](scripts/precompute_vggt_features.py)、[设计](docs/E05_VGGT_OMEGA_DESIGN.md)。旧 [register提案](docs/E05_VGGT_OMEGA_REGISTER_PROPOSAL_SUPERSEDED.md) 已被取代。
- 需VGGT-Ω对应权重；完整LIBERO73.55%、Plus47.63%。

### E06：FLUX.2 VAE

- 原生2D VAE逐帧独立双视角，不加入额外时间压缩；posterior mean、2×2 packing和固定预训练BN，2 clean+8 future、980 tokens。
- [配置](configs/experiments/E06.yaml)、[构建器](scripts/precompute_flux2_features.py)、[设计](docs/E06_FLUX2_VAE_DESIGN.md)。
- 需FLUX.2 Klein VAE BF16导出的Encoder和配置；完整LIBERO97.15%、Plus76.17%。

### 工作区与恢复

- E01–E06统一到 `openwam-exp`，E05/E06不再依赖独立旧工作目录；保留必要模型区别与原训练结果。
- 增加资产/数据/预处理/源码指纹、规范化run identity、精确full-state恢复与独立smoke。训练预算不通过smoke checkpoint续接替代。
- 历史控制器和迁移报告路径见旧文档，绝大部分运行产物不随Git发布；新机器使用当前源码与显式资产路径，不复制旧服务器队列就绪状态。

## 2026-10：E00–E02 基础表示比较

- 固定M29、RynnBrain 1.1-9B离线text-only/avg_all_layers、标准LIBERO数据和10,690步参考预算。
- E00为V-JEPA 2.1原生tubelet baseline；E01为DINOv3-H+逐帧表示；E02为I-JEPA门控历史差分与未来成对均值。各自保留原生接口，不强行对齐所有token数。
- [统一入口](scripts/experiment.py)、[协议](configs/protocol.yaml)、[最初研究边界](docs/EXPERIMENT_PLAN.md)。
- E01完整94.35%/68.08%；E02完整92.75%/69.87%；E00本注册实验未登记完整成绩。资产表中的旧RynnBrain-2B M29或其他数据集checkpoint只是参考，不是本9B baseline的成绩或初始化。

## 继承的评测基础设施与早期验证记录

这些能力早于本轮F0/F1，不能重复声称本次新实现：

- 任务级分片、逐trial RNG、原生动态图像/动作batch、session动作缓冲、raw无损传输、GPU owner thread、本地SQLite WAL、失败记录与持久队列恢复。
- Plus每任务新模拟进程，保持seed→首次导入→环境构造顺序；corrected/original文本分开缓存，默认当前比较使用corrected。
- Sensor Noise中glass/zoom局部等价加速、watchdog只在真实step完成后报告进度；历史验证与边界见 [Plus文档](benchmarks/vla_harness/LIBERO_PLUS.md)。
- FM原生策略adapter保留32预测/16执行、逐步历史和原生T5缓存，见 [FM说明](benchmarks/vla_harness/FM_VJEPA.md)。
- 旧S-VAE部署准备、checkpoint sidecar与严格重载保留，见 [prepare_libero_plus_svae.py](benchmarks/vla_harness/prepare_libero_plus_svae.py)；其历史成绩不属于当前E00–E08对比。

## 文档整合与发布说明

- 2026-11-05：将当前代码快照发布到 `HaoYangDu-NTU/fusion-WAM` 的main；保留目标初始化历史，未上传本地浅克隆依赖缺失的旧历史。
- 本次README整合40个注册配置、已完成与取消状态、训练/评测/Cache入口、资产准备和路径迁移限制；不新增模型实验、结果或权重。
- 后续新实验应同步更新：`configs/experiments.yaml` → 对应配置/设计文档 → 实际结果来源 → `result.md` → 本文件与README；仅有代码或配置时明确标为未测量。
