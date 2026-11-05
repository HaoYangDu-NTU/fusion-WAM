# E05：冻结 VGGT-Ω，空间 Patch + 几何 Registers

状态：2026-10-26 设计方案和权重准备。尚未注册 E05、修改当前训练路径或启动训练。
保留 M29 + RynnBrain 1.1-9B 的策略主干、action backbone、训练参数、loss 定义和评测协议。

## 权重和代码

- 原版代码：`/data/haoyang/vggt-omega`，本地 commit `39a0cb8`。
- 已准备：`encoder/vggt-omega/vggt_omega_1b_512.pt`，4,576,706,117 bytes。
- 官方 SHA256：`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`。
- 本次从已存在的本机文件校验并复制，不声称重新下载成功。
- 文件来源：`facebook/VGGT-Omega`，HF revision `1041e80fc0e911235d3426b0a3d9a81075111a53`。
- 原始官方文件包含 aggregator 和 camera/depth heads；模型只加载 aggregator，排除其他 heads。
- 资产清单：`encoder/vggt-omega/manifest.json`。
- 官方推荐 benchmark 使用 416-Reproduction；当前下载返回 gated repo HTTP 401。
  512 版只先作为结构联调资产；正式实验使用哪个版本必须记录清楚，不冒充 416 版。
- 416-Reproduction 官方大小 4,576,703,488 bytes，SHA256
  `e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896`。

不直接移植 `/data/haoyang/myWAM_frozen_vggt` 的 VGGTO/MoT：它删除 camera token、
修改 register 表并融入另一策略结构，无法作为本项目仅替换表示侧的基准。

## 冻结边界

冻结整个原版 aggregator，包括图像 backbone、frame blocks、inter-frame blocks、
camera tokens 和 register tokens。保持 eval、no_grad，运行 BF16，不只冻结图像 backbone。
保留原版 camera token、16 个 registers 与 patch 的内部交互；外部读取最后聚合层的
2048 维 patch features 和 16 个 register features，去除 camera token 后单独保存 registers。
不使用 camera/depth/text-alignment heads，不增加几何监督，也不替换 RynnBrain。

## 时间与双视角：因果前缀

训练观测沿用连续 33 帧 `[0..32]`，stride=1，front/wrist 固定顺序。
双视角作为独立图像 token 序列联合进入 aggregator，不在 RGB 上拼成一张宽图。

原版跨帧 attention 是双向的；一次编码全部 33 帧再取首帧会产生未来泄露。
第一版使用原版权重和 attention，在以下独立前缀上调用，避免擅自修改 pretrained attention：

| 输出位置 | 输入前缀 | 读取的空间特征 |
|---|---|---|
| clean | 当前帧 0，两个视角 | 帧 0 的两个视角；同时读取 R0 |
| future 1 | 帧 0..4，两个视角 | 帧 4 的两个视角 |
| future 2 | 帧 0..8，两个视角 | 帧 8 的两个视角 |
| … | … | … |
| future 8 | 帧 0..32，两个视角 | 帧 32 的两个视角 |

每次前缀重新执行，固定图像顺序，不沿用未经验证的跨调用缓存。
这是多视角视频上下文编码，不是逐帧独立 encoder。保留 1 clean + 8 future 时间位置。
末端 readout 是本实验的时序适配，不声称等价于 Wan VAE 的 4 帧原生时序压缩；
VGGT-Ω 自身也不是带原生视频因果掩码的编码器。

部署只使用当前双视角计算 clean patch 和 R0，与训练 clean 调用完全一致。

## 空间 Patch 与 Registers 接入 M29

第一版候选输入为每视角 224×224，与 E01–E04 的可见 RGB 分辨率一致；这属于相对
官方 416/512 配方的分辨率迁移，需要真实权重验证。416 分辨率作为独立配置，不能
把 26×26 网格无说明地压成 7×7，或混报分辨率不同的成绩。

| 阶段 | Shape |
|---|---|
| frozen raw patch | `[B,2048,9,14,28]`，两个视角各 14×14 |
| observed registers R0 | `[B,2,16,2048]`，只来自帧 0 的联合双视角调用 |
| 原有可训练 spatial 2×2 patch projection | `[B,1536,9,7,14]` |
| 每时间、每视角的 patch queries | `[B,9,2,49,1536]` |
| Register Adapter 后 | 同上，保持空间顺序 |
| M29 序列 | `[B,882,1536]`，无额外 register 序列 |

空间 projection 沿用现有 clean 直接投影、future bottleneck（256）的机制，不跨视角合并。
Patch 是真实空间局部合并，不能替换为 Global Pooling 或只保留 Registers。

新增必要的 Register Adapter 放在视觉输入边界：

```
H = ExistingPatchProjection(z_t)
G = CrossAttention(Q(H), K(R0), V(R0))
H_fused = H + alpha * Out(G)
```

- Q：1536→256；K/V：2048→256；Out：256→1536；4 heads，head dim 64。
- 逐视角执行；每个 7×7 patch 查询该视角全部 16 个 registers，register 本身已包含
  encoder 的多视角交互信息。两个视角共享 Adapter 权重。
- 不平均 registers，不把 register index 冒充二维空间坐标，不增加 M29 的 token 数。
- 投影使用无 bias 线性层，约 1,835,008 个权重，另加一个 gate；alpha 零初始化。
  第一步允许只有 gate 获得梯度，后续步骤必须验证 Q/K/V/Out 也获得有效梯度。
- 不增加 M29 Transformer 层，不改 action attention、RynnBrain 文本路径或 RoPE 网格。

## 防止另一类泄露：未来 Registers 与扩散噪声

未来 RGB prefix 产生的 patch 是固定的监督目标，按原有 video scheduler 加噪、训练 x-prediction。
Adapter 的未来 queries 来自**加噪后的 patch latent**，不能直接使用干净的未来 patch。
模型只额外接收观测 R0，绝不能将 ground-truth future registers 作为干净 K/V 条件。
未来 registers 保留在冻结 encoder 内部计算中，第一版不额外向策略提供或增加监督。

目标仍为冻结、确定性特征，不把可训练 Adapter 的输出作为其自身漂移的 diffusion target。
原有 clean_condition_causal / joint self-attention mask 保持不变；输入边界的正确性不能
仅靠 M29 mask 补救。部署中每个 denoising step 使用同一次当前观测的 R0。

## Cache 与计算成本

Cache 存冻结 raw patch 和 R0；可训练空间 projection、Register Adapter 和 gate 不能缓存。
复用预取、有界异步写盘、分片锁、原子替换、mmap 按需加载和严格 manifest。
新增 manifest 需锁定完整图像顺序、prefix 划分、读出层、当前 register 提取方式、
分辨率、预处理、code commit 与 checkpoint SHA256。

224 配置，273,465 windows：

- raw patch BF16：3,680.36 GiB，约 **3.59 TiB**。
- R0：每窗口两个视角各 16×2048，约 **33.38 GiB**；约 **3.63 TiB** 合计，不含 header。
- 九次独立前缀累计处理 306 张视角图像/窗口，显著大于一次 33 帧双视角的 66 张。
- 以上是容量/工作量推算，不是实测耗时或吞吐承诺。先做少量真实窗口容量与读取测试。
- 不能为减小 Cache 擅自加 PCA、额外学习压缩、时间池化，或缓存可训练投影输出。
- 可另行验证复用确定性的逐图像 backbone 中间结果，但不能用它替代最终几何特征；
  也不能直接把前缀之间的双向 aggregator 激活当作可增量复用的 causal KV cache。

## 训练参数与验收

保持 M29 30 层、video hidden1536、action hidden1024、RynnBrain 1.1-9B 文本 Cache、
global batch256、10 epochs、10,690 steps、AdamW1e-4、BF16、ZeRO1、checkpoint 策略，
原有 video/action loss 定义及权重不变。视觉 target 采用固定、非仿射的逐 token 通道
LayerNorm 作为候选归一化，需锁定顺序/epsilon并记录这是表示适配，而非 VAE 原生尺度。
原始 Cache 位于该确定性归一化之前；在线/Cache 路径必须使用相同实现。

正式接入前必须完成：

1. 原版 aggregator 严格加载、所有参数冻结；不容忍缺失或意外的 encoder tensors。
2. 验证 patch/register shapes；882 token 数、双视角边界、原有 mask/RoPE 不变。
3. 改变未来 RGB 时，clean patch、R0、较早 prefix 输出不变。
4. 固定 noisy latent 与 R0，改变未暴露的未来 registers 不得改变输入 Adapter 输出；
   固定 clean/noise 后，更改监督标签不得通过额外支路直接改变 policy 输入。
5. 保持空间局部性：改变局部 patch 应改变相应投影；改变 R0 应能影响 Adapter；
   不能出现只有 registers 或全局池化真正起作用的实现。
6. 在线/Cache 数值一致；Encoder 无梯度，Adapter/gate 和原有训练模块有效更新；
   固定 target 不随 Adapter 参数更新而变化。
7. 检查特征尺度、loss、显存、稳态 step time 和 Cache I/O，不预先承诺与 VAE 相同速度。
8. 普通/批量部署、strict reload、当前观测-only LIBERO/Plus 路径、checkpoint-local sidecar。
9. E04 运行期间保持其源码指纹不变；E05 模型接入在隔离副本中开发验证，或在 E04 完成后落地。

完整评测仍为最终 checkpoint 的 LIBERO 2,000 trials、corrected-text Plus 10,030 trials，
同步 10 步 denoising、10 步 action execution、关闭 velocity cache。

## 来源

- 官方代码：https://github.com/facebookresearch/vggt-omega
- 官方复现说明：https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md
- 官方权重：https://huggingface.co/facebook/VGGT-Omega
- 本地实际接口：`/data/haoyang/vggt-omega/vggt_omega/models/aggregator.py`。
