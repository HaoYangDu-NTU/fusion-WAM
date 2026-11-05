# V-JEPA 2.1 / E00 轻量级视频 Decoder 设计

2026-10-26。状态：设计完成，尚未实现。

本方案为 E00 的 WM/MoT 分支输出训练一个独立视频 decoder，把 V-JEPA 2.1 latent
解码为包含 `-4` 在内的 10 帧：

```text
[-4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
```

decoder 在模型结构和训练流程上逐项对齐 V-RAE 第一阶段 reconstruction training：
冻结视觉 encoder，只训练 decoder；显式 latent noise、L1/LPIPS/Gram、AdamW constant
LR、EMA、50 epochs 以及第 30 epoch 后 GAN 的 schedule 都沿用 V-RAE。它不与 MoT、
ActionDiT 或 world-model denoising loss 联合训练。

## 1. 设计目标

- 复用 E00 当前使用的 V-JEPA 2.1 encoder 和 `seperate_encode_view=true`。
- encoder 后将 packed latent 拆成头部、腕部两个独立 decoder 样本。
- 一个 window batch 为 `B` 时，decoder 实际 batch 为 `2B`。
- 默认输出完整 10 帧，不隐式丢弃 `-4` 或当前帧。
- 在同一份代码中支持多种模型尺寸，通过 `decoder.yaml` 切换。
- 数据直接复用当前 MoT 训练使用的 LIBERO dataloader，不新增 dataset、sampler、
  独立 cache reader 或训练脚本。
- 支持 `online` 和 `cache` 两种训练模式；cache 构建与读取逻辑放在
  `decoder_trainer.py` 内，通过 `train.yaml` 开关。
- 新增代码只放在一个 `video_decoder` 包和一个 `decoder_trainer.py` 中。
- decoder 内部结构、训练超参、loss、latent noise 和 GAN schedule 尽量逐项对齐 V-RAE。

### 1.1 非目标

- 不与 MoT/world-model/action 联合训练。
- decoder 损失不反传进 V-JEPA encoder 或 E00 policy。
- 不新增单独的 dataloader、cache reader 或 cache CLI；cache 仍是同一 trainer 的运行模式。
- smoke 阶段允许临时关闭 GAN，但正式 baseline 沿用 V-RAE 第 30 epoch 启用 GAN 的配置。
- 不声称 10 个稀疏观测帧可以恢复全部 33 个原始帧。

## 2. 最小文件分布

```text
openwam-exp/
├── configs/
│   └── decoder/
│       ├── train.yaml
│       └── decoder.yaml
├── openwam/
│   ├── model/
│   │   └── video_decoder/
│   │       ├── __init__.py
│   │       └── vjepa21_lite.py
│   └── train/
│       └── decoder_trainer.py
└── docs/
    └── VJEPA21_LITE_DECODER_DESIGN.md
```

其中：

| 文件 | 职责 |
|---|---|
| `configs/decoder/train.yaml` | 训练、dataloader/encoder 引用、optimizer、loss、checkpoint |
| `configs/decoder/decoder.yaml` | V-RAE 对齐结构参数和 tiny/small/base/large/vrae_xl profile |
| `video_decoder/__init__.py` | 导出 decoder 与 view split/pack helper |
| `video_decoder/vjepa21_lite.py` | V-RAE 同构 decoder、3D RoPE、split/pack、config 解析 |
| `decoder_trainer.py` | dataloader 复用、online/cache 模式、加噪、loss、训练循环、EMA/checkpoint |

不新增单独 dataloader、sampler、cache reader、评估脚本，也不拆分 rope/block/config/
checkpoint 文件。cache 文件是运行产物，不进入源码目录。

### 2.1 直接复用的现有文件

| 现有文件 | 用途 |
|---|---|
| `configs/dataloader/libero.yaml` | 与当前 MoT 训练相同的数据配置 |
| `configs/model/video_backbone/encoder/vjepa21.yaml` | E00 使用的 V-JEPA encoder 结构 |
| `openwam/dataloader/registry.py` | `build_dataset(...)` |
| `openwam/model/video_backbone/encoder/vjepa21.py` | `build_video_encoder(...)` 和 encoder forward |
| `openwam/train/utils/*` | seed、optimizer、EMA、checkpoint、wandb 等通用工具 |

`decoder_trainer.py` 使用 OmegaConf 读取现有 dataloader/encoder YAML，再合并
`train.yaml` 中的少量 overrides，不复制这些配置内容。

## 3. 固定数据契约

### 3.1 E00 encoder 输出

E00 encoder 配置为 V-JEPA 2.1 ViT-G、`history_num_tubelets=1`、
`use_avg_t_pooling=false`、`seperate_encode_view=true`、`vjepa21_forward=video`。
其 packed latent 为：

```text
[B, 1408, 5, 14, 28]
```

其中 `5` 个 temporal chunks，每个 chunk 对应 2 帧；最后 28 个空间位置由 front/wrist
两个 `14` 宽 latent grid 拼接。`tokens=490` 是 WM DiT 再做 `(1,2,2)` patch 后的
token 数，不是 decoder 输入。decoder 必须读取 patch 前、split view 后的完整 latent。

### 3.2 decoder 输入与 view split

```text
packed:  [B, 1408, 5, 14, 28]
views:   [B, 2, 5, 1408, 14, 14]
batch:   [2B, 5, 1408, 14, 14]
```

split/pack 在 `vjepa21_lite.py` 中集中实现，其他位置不得自行 reshape：

```python
def split_vjepa_latents(packed):
    # [B,C,T,H,2W] -> [B,2,T,C,H,W]
    b, c, t, h, packed_w = packed.shape
    w = packed_w // 2
    return packed.reshape(b, c, t, h, 2, w).permute(0, 4, 2, 1, 3, 5)


def pack_vjepa_latents(views):
    # [B,2,T,C,H,W] -> [B,C,T,H,2W]
    b, v, c, t, h, w = views.shape
    return views.permute(0, 3, 2, 4, 1, 5).reshape(b, c, t, h, v * w)
```

view 顺序固定为 `front/head=0`、`wrist=1`。flatten 后的 batch 顺序为：

```text
[front_0, wrist_0, front_1, wrist_1, ..., front_B-1, wrist_B-1]
```

若未来启用可选 view embedding，id 顺序为：

```python
view_ids = torch.arange(2, device=latents.device).repeat(B)
```

baseline `view_conditioning=none` 不创建也不传入 `view_ids`。

### 3.3 10 帧像素目标

当前 dataloader 返回 `video=[0,4,...,32]` 共 9 帧，以及
`clean_condition_images=[-4,0]` 共 2 帧。pixel target 与 encoder 输入统一使用同一个
10-frame helper：

```python
frames = [*clean_condition_images, *video[1:]]
# [-4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
```

不得写成 `clean_condition_images + video`，否则当前帧 0 会重复，输出变成 11 帧。

每帧由 dataloader 以 `224x448` 双视角 packed PIL 形式提供。像素 target 使用原始
`[0,1]` RGB；encoder 输入使用 V-JEPA 自己的 ImageNet normalization。

### 3.4 packed RGB 的 view split/pack

```python
def split_packed_frames(frames):
    # [B,T,C,H,2W] -> [B,2,T,C,H,W]
    b, t, c, h, packed_w = frames.shape
    w = packed_w // 2
    return frames.reshape(b, t, c, h, 2, w).permute(0, 4, 1, 2, 3, 5)


def pack_decoded_views(views):
    # [B,2,T,C,H,W] -> [B,T,C,H,2W]
    b, v, t, c, h, w = views.shape
    return views.permute(0, 2, 3, 4, 1, 5).reshape(b, t, c, h, v * w)
```

decoder 单样本输出为 `[2B,10,3,224,224]`；需要 packed 对比视频时恢复为
`[B,10,3,224,448]`。

## 4. 配置设计

### 4.1 `configs/decoder/train.yaml`

该文件只保存训练逻辑和现有配置引用：

```yaml
project:
  seed: 42
  output_dir: outputs/vjepa21_decoder

encoder:
  config: configs/model/video_backbone/encoder/vjepa21.yaml
  checkpoint: null             # 可选：从 E00 checkpoint 加载 video_encoder.*
  overrides:
    model_path: /path/to/vjepa21-g
    history_num_tubelets: 1
    use_avg_t_pooling: false
    use_wan_style_patch_grid: true
    seperate_encode_view: true
    vjepa21_forward: video
    condition_fusion_method: last_layer
    future_fusion_method: last_layer

dataloader:
  config: configs/dataloader/libero.yaml
  overrides:
    num_frames: 33
    video_stride: 4
    window_stride: 1
    history_num_tubelets: 1
    seperate_encode_view: true
    color_jitter:
      enabled: false

cache:
  mode: auto                    # off | auto | build | read
  root: cache/vjepa21_decoder
  window_stride: 32             # cache 构建时覆盖 dataloader.window_stride
  store_pixels: true            # 同时缓存 10 帧 uint8 target，训练时不再解码视频
  dtype: bf16
  shard_size: 256
  build_batch_size: 64
  build_num_workers: 4
  require_complete: true

decoder:
  config: configs/decoder/decoder.yaml
  profile: base

training:
  window_batch_size_per_gpu: 16  # 4 GPU: window global 64, view global 128
  gradient_accumulation_steps: 1
  epochs: 50
  mixed_precision: bf16
  learning_rate: 2.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0
  grad_clip: null
  scheduler:
    name: constant
    warmup_steps: 0
  latent_noise_tau: 0.8
  ema_decay: 0.9995
  num_workers: 4
  save_steps: 1000

visualization:
  enabled: true
  every_n_epochs: 1
  max_samples: 4       # decoder view samples: front/wrist 成对排列
  fps: 2.5
  include_target: true

loss:
  l1: 1.0
  lpips: 1.0
  gram: 100.0
  temporal_difference: 0.0
  perceptual_frames_per_chunk: 2
  perceptual_chunk_size: 2
  backbone_checkpoint: /path/to/vgg16.pt
  calibration_checkpoint: /path/to/lpips_vgg.pt

gan:
  enabled: true
  generator_start_epoch: 30
  discriminator_start_epoch: 30
  update_interval: 5
  generator_weight: 0.3
  discriminator_weight: 0.3
  discriminator_checkpoint: /path/to/videomae_discriminator.pth
  gen_loss_type: ns_g_loss
  disc_loss_type: ns_smooth
  lecam: true
  lecam_weight: 0.001
  optimizer:
    lr: 2.0e-5
    betas: [0.9, 0.95]
    weight_decay: 1.0e-4
```

`decoder_trainer.py` 读取 `dataloader.config` 和 `encoder.config`，用 OmegaConf 合并
overrides。命令行仍可覆盖 `decoder.profile`、batch、LR 等字段。

### 4.2 `configs/decoder/decoder.yaml`

多种模型大小集中在一个文件中：

```yaml
profile: base

common:
  input_dim: 1408
  temporal_chunks: 5
  tubelet_size: 2
  latent_size: [14, 14]
  patch_size: 16
  image_size: [224, 224]
  output_frames: 10
  output_size: [224, 224]
  num_channels: 3
  mlp_ratio: 3.5555555555555554
  layer_norm_eps: 1.0e-12
  attention_dropout: 0.0
  attention_mode: clean_condition_causal
  attention_backend: auto
  rope_theta: 10000.0
  gradient_checkpointing: false
  view_conditioning: none
  num_views: 2

profiles:
  tiny:
    hidden_size: 384
    depth: 8
    num_heads: 6
  small:
    hidden_size: 512
    depth: 8
    num_heads: 8
  base:
    hidden_size: 512
    depth: 12
    num_heads: 8
  large:
    hidden_size: 768
    depth: 12
    num_heads: 12
  vrae_xl:
    hidden_size: 1152
    depth: 28
    num_heads: 16
```

profile 只改变 `hidden_size/depth/num_heads`。`vrae_xl` 与 V-RAE 原始 DecoderConfig
宽度/深度/heads 完全一致，仅把 `tubelet_size` 从 4 改为当前 E00 的 2，并改为单视角
`224x224`；它可作为 417M 上限基线。默认 `view_conditioning=none`，即 decoder 本体和
V-RAE 完全同构；embedding 只保留为后续可选消融。

```bash
python -m openwam.train.decoder_trainer \
  --config-path configs/decoder --config-name train \
  decoder.profile=tiny

# 强制在线编码，不使用 cache
python -m openwam.train.decoder_trainer --config-path configs/decoder --config-name train \
  cache.mode=off

# 只从完整 cache 训练
python -m openwam.train.decoder_trainer --config-path configs/decoder --config-name train \
  cache.mode=read
```

### 4.3 参数量

| profile | hidden | depth | heads | head dim | 参数量约 | BF16 权重 |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 384 | 8 | 6 | 64 | 14.35M | 27.4 MiB |
| small | 512 | 8 | 8 | 64 | 24.96M | 47.6 MiB |
| base | 512 | 12 | 8 | 64 | 36.63M | 69.9 MiB |
| large | 768 | 12 | 12 | 64 | 81.16M | 154.8 MiB |
| vrae_xl | 1152 | 28 | 16 | 72 | 416.90M | 795.2 MiB |

首轮建议：`D00 tiny` 做 smoke；`D01 small`、`D02 base`、`D03 large` 使用相同数据、
loss、噪声和训练预算比较。

## 5. 模型设计

`vjepa21_lite.py` 中的 decoder 不采用新的 U-Net/CNN 结构，而是复现 V-RAE
`VRAEDecoder` 的结构和计算顺序。尺寸可以缩小，但模块组成必须一致。

### 5.1 与 V-RAE 对齐的结构清单

| 模块 | 对齐要求 |
|---|---|
| input projection | `Linear(input_dim, hidden_size)` |
| spatial position | 可学习 `Parameter[1,14*14,hidden]`，用 2D sin/cos 初始化 |
| position resize | shape 不同时用 bicubic 插值 |
| Transformer block | 两层 pre-norm：`x=x+Attn(LN1(x))`，`x=x+MLP(LN2(x))` |
| attention | fused QKV `Linear(hidden,3*hidden)`，output `Linear(hidden,hidden)` |
| RoPE | V-RAE style `t/y/x` 三维 RoPE，作用于 Q/K，`rope_theta=10000` |
| MLP | `Linear(hidden,int(hidden*mlp_ratio)) -> GELU -> Linear(...,hidden)` |
| final norm | `LayerNorm(hidden, eps=1e-12)` |
| prediction | `Linear(hidden,tubelet*patch*patch*3)` |
| depatchify | V-RAE 相同的 8D reshape/permute 顺序 |
| initialization | scratch，不加载 V-RAE decoder 权重 |

`mlp_ratio=3.5555555555555554`、`attention_dropout=0`、`layer_norm_eps=1e-12`、
`attention_backend=auto`、`rope_theta=10000` 均直接沿用 V-RAE。decoder 支持
`full`、`chunk_causal` 和 `clean_condition_causal`。所有 attention mode 分支集中在
`DecoderAttention.forward` 的同一个 `if/elif` 结构中，不使用 attention mask。
训练默认使用与
MoT 一致的 `clean_condition_causal`：第一个 clean latent chunk 只能读取 clean
token，后续 future chunks 可以读取 clean 和全部 future tokens。RoPE 保留 V-RAE style
的轴维度分配、邻近 pair 旋转和尾部不旋转语义。

代码结构也按 V-RAE 的类职责拆分在同一个文件内：`DecoderConfig`、
`spatial_sincos`、`DecoderAttention`、`DecoderBlock`、`VRAEDecoder` 以及 V-RAE style
3D RoPE helper。核心计算顺序与 V-RAE 一一对应。

唯一必须相对 V-RAE 修改的 geometry 是：

| 项目 | V-RAE 原配置 | 本 decoder |
|---|---|---|
| image size | `256x256` | `224x224` 单视角 |
| latent grid | 由 encoder 决定 | `14x14` |
| tubelet size | 4 | 2，匹配 E00 native tubelet |
| output frames | T*4 | 5*2=10 |

### 5.2 前向

```text
z [Bv,5,1408,14,14]
  -> [Bv,5,14,14,1408] -> [Bv,980,1408]
  -> Linear(1408,hidden)
  -> add learnable 2D spatial position [14,14]
  -> N x pre-norm Transformer blocks
  -> LayerNorm
  -> Linear(hidden,2*16*16*3=1536)
  -> depatchify
  -> [Bv,10,3,224,224]
```

单视角序列长度为 `5*14*14=980`。在 `clean_condition_causal` 下，第一个
`14*14=196` 个 clean tokens 的 query 只 attend clean keys；后面 4 个 future chunks
不受限制，可以 attend clean 和全部 future keys。`chunk_causal` 和
`clean_condition_causal` 都通过多次 SDPA 调用实现，不构造 mask。约束作用在 latent
chunk 上，不是直接作用在输出 10 帧像素上。`full` 保留为 V-RAE RoPE/attention 对照，
`chunk_causal` 保留为低延迟部署选项。

### 5.3 view embedding 是可选扩展

为保持与 V-RAE 完全同构，默认 `view_conditioning=none`。front/wrist 已经是两个独立
batch 样本，共享同一 decoder，不传 view id：

```python
pred = decoder(view_z)
```

只有后续消融需要时，才可配置 `view_conditioning=embedding`，加入一个额外
`nn.Embedding(2,hidden)`。它不是 V-RAE 原结构，不能作为 baseline。

### 5.4 depatchify

```text
[Bv,5,14,14,2,16,16,3]
 -> [Bv,5,2,3,14,16,14,16]
 -> [Bv,10,3,224,224]
```

该 reshape/permute 顺序必须与 V-RAE `VRAEDecoder.depatchify` 一致。

## 6. 复用当前 MoT Dataloader

`decoder_trainer.py` 直接调用现有接口：

```python
dataset = build_dataset(dataloader_cfg, split="train")
loader = DataLoader(
    dataset,
    batch_size=training.window_batch_size_per_gpu,
    shuffle=True,
    num_workers=training.num_workers,
    collate_fn=list,
    pin_memory=True,
)
```

这与当前 `OpenWAMTrainer.build_dataloader()` 的关键行为一致：返回 sample list，不新增
reader、collate class 或 sampler。

online 模式下每个 step：

1. 从现有 dataloader 取得 B 个 window sample。
2. 用 `target_frames` helper 构造每窗 10 帧 packed target。
3. 用同样的 10 帧调用 `VJEPA21VideoEncoder.preprocess_video_batch()`。
4. 在 `no_grad()` 下调用 `encoder.batch_encode()` 得到 `[B,1408,5,14,28]`。
5. split 后 flatten 为 `2B` 个 view latent。
6. 像素 target 也拆成 `2B` 个 `[10,3,224,224]`。
7. 对每个 view latent 加噪后送入 decoder。
8. 输出与对应 view target 计算 loss。

V-RAE 训练配置还启用了 random flip 等数据增强；本方案优先复用 MoT dataloader，因此
首版不额外添加增强。若后续增加增强，必须对 decoder pixel target 和 E00 encoder input
应用完全相同的几何变换，避免 latent 与像素错位。

### 6.1 Cache 模式

`cache.mode` 支持：

| mode | 行为 |
|---|---|
| `off` | 每个 step 在线运行 E00 encoder |
| `build` | 使用当前 MoT dataloader 构建 cache，完成后退出 |
| `read` | 只从完整 cache 训练，缺项立即报错 |
| `auto` | cache 完整则直接训练；否则先构建再训练 |

cache 构建和读取都实现在 `decoder_trainer.py` 内，不新增 cache 脚本或 reader 模块。
构建阶段把现有 dataloader 的 `window_stride` 覆盖为 `cache.window_stride`，默认 32，
避免缓存 273,465 个高度重叠 window。

### 6.2 Cache payload

每个原始 window 存一条记录：

| 字段 | shape / dtype |
|---|---|
| `view_latents` | `[2,5,1408,14,14]` BF16 |
| `view_frames` | `[2,10,3,224,224]` uint8，`store_pixels=true` 时保存 |
| `valid_mask` | `[2,10]` bool |
| `window_id` | scalar/string，用于追踪 source window |

front/wrist 在写 cache 前已经 split；训练读取后直接 flatten 前两维得到 `2B` 个 decoder
样本，不需要再次运行 encoder，也不需要重新 split latent。

每个 window 的 BF16 latent 为 `5.27 MiB`。若缓存像素，uint8 target 约 `2.87 MiB`，
合计约 `8.14 MiB/window`。按 stride 32 约 8.5k windows 计算：

| payload | 大小约 |
|---|---:|
| latent only | 44 GiB |
| latent + uint8 pixels | 69 GiB |

若沿用 `window_stride=1` 缓存全部 273,465 windows，latent-only 也约为 `1.37 TiB`；
因此 cache 模式默认只构建 stride 32 的训练子集，不把全窗口 cache 作为默认行为。

保存像素可以避免训练时再次解码视频，并保证 latent 与 pixel target 严格来自同一次样本
构造。若磁盘紧张，可设 `store_pixels=false`，但必须用稳定 `window_id` 重新取同一
window，且任何随机增强都必须在 cache build 与读取两侧保持一致。

### 6.3 Cache 文件与校验

cache 是运行产物：

```text
cache/vjepa21_decoder/<fingerprint>/
├── manifest.json
├── shard_00000.pt
├── shard_00001.pt
└── ...
```

单 shard 默认 256 windows，先写临时文件再原子 rename，支持中断续建。分布式构建时按
global window index 分给 rank，同一 shard 只由一个 rank 发布，避免四卡重复编码和写冲突。
manifest 必须记录 dataset fingerprint、window stride、10 帧 offsets、view 顺序、
encoder checkpoint/asset fingerprint、packed/split shape、dtype 和 shard 数。

decoder profile（tiny/small/base/large/vrae_xl）不进入 fingerprint，所以所有尺寸共享同一份
cache。`read` 模式在训练前检查完整 key 覆盖；缺失、shape 不符或 encoder fingerprint
不一致时直接失败，不能静默切回 online。

## 7. Latent 噪声

在 split view 之后加噪，使 front/wrist 拥有独立噪声：

```python
sigma = torch.rand(Bv, 5, 1, 1, 1, device=z.device, dtype=z.dtype) * tau
noise = torch.randn_like(z)
z_train = z + sigma * noise
```

默认 `tau=0.8`，与 V-RAE reconstruction training 一致。V-JEPA 输出已经过
`feature_norm`，不再额外做第二次 LayerNorm。与 V-RAE 的 `apply_in_eval=false` 一致，
验证和可视化使用 clean latent，不加噪。

## 8. 损失

```text
L_rec = 1.0 * L1
      + 1.0 * LPIPS
      + 100.0 * Gram
      + 0.0 * temporal_difference
```

这与 V-RAE V-JEPA reconstruction recipe 一致。LPIPS/Gram 使用 V-RAE 的 VGG16
checkpoint 和 LPIPS calibration；输入从 `[0,1]` 转到 `[-1,1]` 后计算。Gram 在
`224x224` 单 view 上计算，不做 packed canvas resize。

decoder 已按 view 独立成 batch，因此：

- L1 直接作用于 `[2B,10,3,224,224]`。
- LPIPS 和 Gram 在单 view `224x224` 上计算。
- front/wrist loss 等权平均。
- 不对 packed `224x448` 直接计算 perceptual loss。
- `perceptual_chunk_size=2`、`perceptual_frames_per_chunk=2`，即 5 个 temporal
  chunk 的全部 10 帧都参与 LPIPS/Gram，对应 V-RAE 原配置的 all-frames 行为。

### 8.1 GAN 阶段

正式 baseline 与 V-RAE 一致，从第 30 epoch 开始：

```text
L_total = L_rec + 0.3 * L_adv
L_adv   = -log(sigmoid(D(reconstruction)))
```

生成器和判别器同 epoch 启动，判别器每 5 step 更新，使用
`ns_g_loss` / `ns_smooth`、LeCam regularization 和 V-RAE VideoMAE discriminator。
GAN 代码放在 `decoder_trainer.py` 中，不新增 loss/GAN 文件。smoke 阶段可临时
`gan.enabled=false` 排查重建问题，但正式尺寸比较必须使用相同 GAN 配置。


## 9. `decoder_trainer.py` 流程

### 9.1 初始化

1. 读取 `train.yaml`。
2. 读取并合并现有 dataloader/encoder YAML。
3. 读取 `decoder.yaml`，按 `profile` 解析模型尺寸。
4. 通过 `build_video_encoder()` 构造 E00 同款 encoder。
5. encoder 设 `eval()`、`requires_grad_(False)`。
6. 可选从 E00 checkpoint 加载 `video_backbone.video_encoder.*`。
7. 根据 `cache.mode` 校验、构建 cache 或选择 online loader。
8. cache build 完成后如果 mode=`build` 则退出；`auto/read` 使用 trainer 内部的
   `_CacheDataset` 训练，`off` 使用当前 MoT dataloader。
9. 构造 decoder；optimizer 只接收 `decoder.parameters()`。
10. EMA 只跟踪 decoder。

不得构造 MoT、ActionDiT 或完整 policy。

V-RAE Stage-1 还会训练负责把 encoder tubelet 压到 4 帧的 `TemporalAttentionPool`。
E00 已经通过 `history_num_tubelets=1` 和 `use_avg_t_pooling=false` 得到固定 5 个
2-frame tubelet，因此这里不再添加 temporal pooler；训练图与 V-RAE 的唯一职责差异是
“没有 pooler、只有 decoder”。

### 9.2 单步伪代码

```python
if cache_reader is not None:
    view_z, view_frames = next(cache_reader)  # [2B,...], [2B,...]
else:
    samples = batch
    packed_frames = build_10_frame_targets(samples)
    encoder_input = encoder.preprocess_video_batch(
        [encoder_frames(sample) for sample in samples]
    )
    with torch.no_grad():
        packed_z = encoder.batch_encode(encoder_input)

    view_z = split_vjepa_latents(packed_z).flatten(0, 1)
    view_frames = split_packed_frames(packed_frames).flatten(0, 1)

sigma = torch.rand(Bv, 5, 1, 1, 1, device=view_z.device) * tau
view_z = view_z + sigma * torch.randn_like(view_z)

pred = decoder(view_z)                 # view_conditioning=none, V-RAE-aligned
loss = reconstruction_loss(pred, view_frames)

if gan.active(epoch):
    loss = loss + 0.3 * gan.generator_loss(pred, epoch)

optimizer.zero_grad(set_to_none=True)
accelerator.backward(loss)
optimizer.step()
if gan.active(epoch):
    gan.discriminator_step(view_frames, pred)
ema.update(decoder)
```

### 9.3 默认超参

| 项目 | 默认值 |
|---|---|
| optimizer | AdamW |
| LR | `2e-4` |
| betas | `[0.9,0.95]` |
| weight decay | `0.0` |
| precision | BF16 |
| EMA | `0.9995` |
| grad clip | none，与 V-RAE recipe 一致 |
| window batch/GPU | 16 |
| decoder view batch/GPU | 32 |
| window global batch（4 GPU） | 64 |
| decoder view global batch（4 GPU） | 128 |
| epochs | 50 |
| schedule | constant |
| warmup | 0 |
| GAN | epoch 30 开始，generator weight 0.3 |

### 9.4 checkpoint

每个 run 只保存 decoder 和训练状态：

```text
outputs/vjepa21_decoder/<profile>/s42/<timestamp>/
├── decoder.safetensors
├── optimizer.pt
├── trainer_state.json
├── resolved_config.yaml
├── metrics.jsonl
└── visualizations/
    └── epoch_0001/
        ├── sample_000_front_idx00000000_decoded.mp4
        ├── sample_000_front_idx00000000_comparison.mp4
        ├── sample_001_wrist_idx00000000_decoded.mp4
        └── sample_001_wrist_idx00000000_comparison.mp4
```

checkpoint 不复制 V-JEPA encoder 权重，只保存 encoder config、asset/checkpoint
fingerprint、cache fingerprint（若启用）和 decoder 参数。resume 时 cache mode、
window subset 和 cache fingerprint 必须保持一致。

## 10. OpenWAM / E00 接入

训练阶段与 E00/MoT 完全解耦。推理时可以返回 latent 后解码：

```python
result = architecture.generate(..., decode_video=False, return_latents=True)

views = split_vjepa_latents(result["video_latents"])
decoded = decoder(views.flatten(0, 1))
frames = pack_decoded_views(decoded.reshape(B, 2, *decoded.shape[1:]))
```

后续也可以给 Wan video backbone 增加可选 `external_video_decoder`：

- `decode_video=false` 时 action benchmark 路径完全不变。
- `decode_video=true` 时，external decoder 内部执行 split -> decode -> pack。
- 不修改 V-JEPA encoder 的 `pixel_decode=false`。
- decoder 在 E00 checkpoint strict load 完成后再加载，避免改变旧 policy state dict。

## 11. 验证要求

首版不新增大量测试文件，但 trainer 启动和 smoke 必须覆盖：

- profile 只能从 `tiny/small/base/large/vrae_xl` 中选择。
- `[2B,5,1408,14,14] -> [2B,10,3,224,224]` shape 正确。
- split view 后 flatten 顺序固定为 front/wrist。
- split -> pack 对 latent 和 pixel 都能 round-trip。
- 10 帧严格为 `[-4,0,4,...,32]`。
- `clean_condition_causal` 下 clean chunk 输出对 future latent 扰动严格不变，同时
  future chunk 输出可以使用 clean 信息。
- `full` 和 `chunk_causal` attention 分支保持原行为。
- optimizer 只包含 decoder 参数。
- encoder 参数 `requires_grad=False` 且 backward 后 `grad is None`。
- 训练图中没有 MoT/ActionDiT。
- tiny 跑 100 step，loss 有限且 decoder 梯度非零。
- cache `build -> read` 与 online 模式在同一批固定 window 上得到逐元素一致的 per-view
  latent 和 pixel target。
- cache mode 在 profile 之间切换时复用同一 fingerprint；encoder/config 改变时必须
  产生不同 fingerprint。
- `cache.mode=read` 遇到缺 shard 或 fingerprint 不匹配时直接失败，不能静默切 online。
- 使用相同 `DecoderConfig` 时，decoder `state_dict` 的 module key 和 tensor shape 与
  V-RAE `VRAEDecoder` 一致；只允许 geometry 导致 spatial position tensor shape 不同。
- epoch 30 前后各跑一次 GAN smoke，确认 generator loss 只更新 decoder、判别器不更新
  encoder/decoder。
- 保存并重新加载 decoder 后固定输入输出一致。

## 12. 实施顺序

1. 新增 `configs/decoder/train.yaml` 和 `configs/decoder/decoder.yaml`。
2. 新增 `openwam/model/video_decoder/vjepa21_lite.py`，实现五档模型、view split/pack、
   V-RAE 同构 block/RoPE/depatchify，并保留可选 view embedding。
3. 新增 `openwam/train/decoder_trainer.py`，复用现有 dataloader、encoder 和 project
   training utils，并在同文件实现 `off/build/read/auto` cache 模式。
4. 先用 `cache.mode=off` 跑 tiny 100-step smoke，再做 build/read parity 与
   `cache.mode=auto` 验证。
5. 分别用 online 和 cache 各跑一组固定窗口对照，确认 latent、target 和 loss 一致。
6. 依次训练 `small/base/large`，正式多尺寸实验默认使用 cache。
7. 选择最优 profile 后接入 E00 推理，完成 WM latent 到 10 帧视频的解码。
