# Wan2.1 / Wan2.2 VAE encoder

2026-10-25。用户明确要求两代 VAE 的编码器，生成 DiT 方案已取消。

## 权重

| 资产 | 独立路径 | 官方文件大小 | SHA256 |
|---|---|---:|---|
| Wan2.1 VAE | `encoder/wan21-vae/Wan2.1_VAE.pth` | 507,609,880 bytes | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| Wan2.2 VAE | `encoder/wan22-vae/Wan2.2_VAE.pth` | 2,818,839,170 bytes | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |

两份文件均已和官方 SHA256 核对。已有下载中的正确 VAE 文件整理到独立目录，
没有重复下载或复制生成主干。官方 VAE 文件含编码和解码两侧；运行时只加载
`encoder.*` 与输出 posterior 参数的 `conv1.*`，采用均值 latent 并执行原生 mean/std
归一化。decoder、conv2、生成 DiT、CLIP、UMT5 均不加载、不进入 checkpoint。

官方来源：

- [Wan2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/blob/main/Wan2.1_VAE.pth)
- [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/Wan2.2_VAE.pth)

Wan2.1 VACE-1.3B 与 I2V-14B 中的 `Wan2.1_VAE.pth` 哈希相同。
1.3B/14B 是生成主干型号，不能当作 VAE 的参数规模。

## 实验

| 项目 | E03：Wan2.1 VAE | E04：Wan2.2 VAE |
|---|---|---|
| 编码侧参数 | 53,596,928 | 149,637,088 |
| 训练视频 | 连续 33 帧 `[0..32]` | 同左 |
| 时间压缩 | 原生 /4，首帧独立 | 同左 |
| clean / future | 1 / 8 | 1 / 8 |
| latent channels | 16 | 48 |
| 空间压缩 | /8 | /16 |
| 双视角 latent | `B×16×9×28×56` | `B×48×9×14×28` |
| M29 patch | `(1,2,2)` | `(1,2,2)` |
| policy tokens | 3528 | 882 |

两个 224×224 视角独立进行原生视频 VAE 编码，沿宽拼接。M29 的 view embedding 与
局部 RoPE 网格由 encoder spatial_compression 推导，不再固定为 JEPA 的 14×14。

causal VAE 的首个 latent 只依赖当前帧；训练直接使用视频首 latent 作为 clean，
future latents 为 detached targets。部署仅编码当前帧，已验证与完整视频的首 latent
逐元素一致。不额外使用五帧历史、DiT timestep 或预训练生成模型的条件分支。

冻结整个 VAE encoder；训练 M29 video/action policy 和输入输出投影，从头初始化。
任务文本继续使用 RynnBrain 1.1-9B 缓存。AdamW、学习率、全局 batch 256、10 epochs /
预计 10,690 steps、seed 42、BF16、ZeRO-1、gradient checkpointing 沿用现有协议。

采样比 E00/E01/E02 更密集，token 数也不同；需在结果中记录这些差异。

## 接入与验证

- 代码：`openwam/model/video_backbone/encoder/wan_vae.py`。
- 配置：`configs/experiments/E03.yaml`、`E04.yaml`。
- 两种模型的 source 权重均只筛选编码侧，并 strict load。
- checkpoint 使用本地 `wan_vae/encoder.json` 重建，不依赖训练源目录。
- 已接入 LIBERO/Plus 结构验证、sidecar 重定位、普通与 batch 推理。
- 真实权重 GPU 检查：`reports/wan_vae_encoder_check.json`。
- 官方权重校验：`reports/wan_vae_assets_verified.json`。
- VAE 与相关代码回归：`reports/wan_vae_tests.log`（138 passed）。

两种真实 VAE 的输出与因果性检查均通过，检查中 encoder peak allocated memory
均低于 0.6 GiB；该数值不含训练中的 M29、optimizer 与 batch 64 激活。

## 本次训练安排

按此前授权先验收 E03（Wan2.1 VAE）的四卡每卡 batch 64、完整状态保存和恢复。
通过后从头启动正式 E03；不使用 smoke 的 optimizer/scheduler 作为正式训练起点。
控制器会在失败时拦住正式启动。

```text
tmux: openwam-exp-wan21-vae
status: reports/training_20261025_wan_vae/status.json
logs: reports/training_20261025_wan_vae/{capacity,resume,formal}.log
```

本次未安排 E04 正式训练。旧 DiT smoke 日志保留在 `reports/training_20261025_E03/`，
已停止且不再作为当前实验。
