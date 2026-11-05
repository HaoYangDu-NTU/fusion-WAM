# E06 方案：M29 + RynnBrain 1.1-9B + FLUX.2 VAE

状态更新（2026-10-29）：E06已完成10,690步正式训练和两项完整评测。LIBERO 97.15%（1943/2000），LIBERO-Plus corrected text 76.17%（7640/10030）；评测于04:26（北京时间）结束，详细套件/类别与源summary见主项目 `result.md`。下文原“方案/后续”段落保留为设计过程记录，最新结果以上述完整验收为准。
名称使用 **FLUX.2 VAE**，不命名为“FLUX VAE2.2”；后者容易与 Wan2.2 混淆。

## 研究目标和主实验选择

采用既定 System-Level Fairness，固定 M29、RynnBrain、Action Expert、训练数据、预算、loss
和评测协议，比较 Encoder＋必要适配的完整方案。主实验沿用用户为 E05 确认的观测信息：

- 时间网格：`[-4,0,4,8,12,16,20,24,28,32]`。
- 历史 `t=-4` 与当前 `t=0` 分别保留，2 clean；未来8个时间位置为8个目标。
- 每时间点 Front/Wrist 两张224×224图像分别编码，输出 latent 按 Front 左、Wrist 右沿宽度拼接。
- FLUX.2 是原生二维图像 VAE；不额外增加四帧平均、时间池化、历史差分或融合模块。
- 不将整张左右拼接 RGB 当作一张图编码，避免改变单图 Encoder 的空间上下文。

选择逐帧编码的依据是二维 VAE 原生只处理单图；原 OpenWAM_v2 的四帧平均是外加的时间适配，
不是 FLUX.2 自身的时序压缩。980 tokens 已可控，不需要为了模仿 Wan 而引入这项压缩。
M29 继续负责这些时间位置之间的策略建模。

## Pipeline 与 Shape

```text
Reader: 9 个 current/future RGB + [history,current] references
 → 组装 [-4,0,4,...,32]： [B,3,10,224,448]
 → 拆开相机和时间： [B*10*2,3,224,224]，像素[-1,1]
 → Frozen FLUX.2 VAE Encoder + quant_conv
 → posterior mean： [B*10*2,32,28,28]，不随机采样
 → 原生2×2空间packing： [B*10*2,128,14,14]
 → 原生固定BN归一化，不增加LayerNorm
 → 按各相机空间网格还原，再沿宽度拼接： [B,128,10,14,28]
 → 原scheduler：历史/当前两个clean保持干净；未来8个latent加噪
 → 原BottleneckPatchEmbed：空间(1,2,2)，future bottleneck=256
 → [B,1536,10,7,14] → [B,980,1536]
 → 原M29 30层 + 原Action Expert
```

两次2×2操作性质不同：第一次是预训练 VAE 原生空间到通道的无损重排；
第二次是已有 M29 的可训练 Spatial Projection。没有重复叠加新的 Encoder 专属压缩。
每视角 latent 宽14、可被 M29 patch2整除，空间patch不会跨左右相机边界。

## 冻结、Projection 和 Target

冻结全部 Encoder、quant_conv 以及 BN 统计量，eval/no_grad；不加载 Decoder 参与计算。
BN 使用预训练 running_mean/running_var，不使用训练batch统计；避免缓存漂移与跨帧统计污染。

复用 M29 当前的 clean直接投影 / future bottleneck256：

| 模块 | 结构 | 参数 |
|---|---|---:|
| Clean Projection | Conv3d(128→1536, kernel=stride=(1,2,2)) | 787,968 |
| Future Projection | Conv3d(128→256,(1,2,2),无bias) + Conv3d(256→1536,1) | 525,824 |
| 输出Head | Linear(1536→128×4)，原unpatchify | 786,944 |
| 合计 | 必要的原有输入/输出投影 | 2,100,736 |

这些投影从头训练，无对应 VAE 预训练权重；没有额外 Cross-Attention、Gate、几何分支或loss。
完整 M29 的可训练参数预计1,899,117,136，由现有正式模型替换输入/输出层推算；
尚未构建完整E06模型核实，不能把预计值写成已完成训练模型的实测值。
已下载 Encoder 实际34,425,856参数，冻结后可训练参数0。

Diffusion target 使用投影前的冻结128维latent；训练Projection不会改变监督目标。
只用历史和当前作为无噪声条件，未来真实latent仅用于原加噪路径和监督。
保持原x-prediction定义、scheduler、attention mask、clean排除video loss、action loss。
训练与部署使用同一原生packing/BN；部署在线编码实际可得的历史与当前观测，
Episode开始处复用当前帧作为尚不存在的历史，保持现有reader/评测约定。

## Cache 方案

原生单帧Encoder无时间上下文，可以按 Episode/实际帧/相机缓存一次，跨滑动窗口复用。
缓存只保存原生packing＋固定BN后的冻结输出，绝不缓存可训练Projection。

- 每视角 `[128,14,14]` BF16为50,176字节；双视角每时间点100,352字节≈98KiB。
- 按现有273,465帧估计，去重后约25.56GiB，不计索引和文件头。
- 若按每窗口10个时间位置重复存储，则约255.58GiB，应避免。
- 建议按Episode分片存储，帧索引随机读取，复用现有预取、异步写盘机制；
  这些是后续实现计划，不是本轮已经完成的Cache程序。
- Key/manifest包含权重SHA、原生packing和BN规则、相机身份/顺序、分辨率、预处理和数据版本。
- 随机增强会改变输入，当前沿用关闭color jitter的协议；Cache与Online需真实样本数值验证。
- 先小样本测编码/读取吞吐，再构建全量；本轮没有生成全量Cache。

## 固定协议及差异

保持M29 30层/1536维、Action Expert 1024维及原Head，冻结RynnBrain1.1-9B文本Cache；
global batch256、4卡每卡64、seed42、10 epochs/预计10,690steps；原AdamW、LR、warmup、
scheduler、video/action loss及完整LIBERO/LIBERO-Plus协议不变。

| 实验 | Clean/Future | Token | Encoder内时序/视角交互 |
|---|---:|---:|---|
| E00 V-JEPA baseline | 1/4 | 490 | 原生时序；历史和当前压为1 clean |
| E01 DINOv3 | 2/8 | 980 | 逐帧、相机独立 |
| E02 I-JEPA | 1/4 | 490 | ViT逐帧，外加历史差分和未来两帧平均 |
| E03/Wan2.1已评测Patch方案 | 1/8 | 882 | 原生因果时空压缩，相机独立，无额外过去帧 |
| E04 Wan2.2 | 1/8 | 882 | 原生因果时空压缩，相机独立，无额外过去帧 |
| E05 VGGT-Ω | 2/8 | 980 | 每时间点双视角原生联合，无跨时间编码 |
| E06 FLUX.2 VAE（方案） | 2/8 | 980 | 逐帧、相机独立，无额外时间平均 |

E06与E01/E05具有相同观测网格及token预算，但latent通道、归一化、预训练目标、跨视角交互
与必要投影参数不同。系统公平不等于这些因素被完全控制，不能把效果差异全部归因于预训练任务。

## 已下载资产与真实来源

目录：`/data/fcy/code/openwam-exp/encoder/flux2-vae/`。

原目标 `black-forest-labs/FLUX.2-dev` 的当前访问返回401 GatedRepoError。
实际下载的是官方公开 `black-forest-labs/FLUX.2-klein-4B` 仓库的 **vae/** 组件，
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`，类名 `AutoencoderKLFlux2`。
其配置 `_name_or_path` 指向 FLUX.2-dev，但不据此宣称与dev原始FP32文件数值或字节完全相同。
本实验应明确标记“官方Klein仓库提供的FLUX.2 VAE，BF16”，不是“flux2.2”，也不是Klein生成器。

| 文件 | 大小 | 用途 |
|---|---:|---|
| config.json | 821 B | 官方VAE结构配置 |
| diffusion_pytorch_model.safetensors | 168,120,878 B | 官方VAE组件；包含Encoder/Decoder，保留作来源原件 |
| encoder_only.safetensors | 68,863,720 B | 提取并严格重载验证的Encoder＋quant_conv＋固定BN；不含Decoder |
| manifest.json | 小文件 | repo/revision/精度/各文件SHA256和访问记录 |
| encoder_load_validation.json | 小文件 | 真实权重加载、形状、有限值、固定BN和Projection检查 |

官方原件SHA256：`ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04`。
Encoder-only SHA256：`d03fa09ffe4e618a172cc0ba1c393d63c4517737b0182b29c52830c9c60eb2ef`。
没有下载FLUX生成Transformer、文本Encoder或整套生成Pipeline。

Encoder-only采用OpenWAM `FluxVaeEncoderCore` 原生key格式，不是完整Diffusers VAE格式；
现有Wrapper默认读取官方diffusion_pytorch_model.safetensors，不能直接把Encoder-only文件
冒充其输入。后续E06需显式实现该轻量加载入口，并覆盖逐帧/双clean适配。
原Wrapper的4倍时间平均本轮未修改；没有声称仅改config已能直接开始E06训练。

## 本轮验证范围与后续验收

已完成真实权重严格加载和Encoder-only严格重载，输出 `[4,128,14,14]` 有限，
重载输出最大绝对差0，BN running统计不变。CPU四线程四张224图约0.98s，
该一次测量不能当作GPU延迟或稳态训练吞吐。另用真实特征构造完整10位置网格，
独立检查现有Projection的980token几何和反传；此为适配接口检查，不是E06训练smoke。

后续实现后需要：真实reader完整数据路径、clean/future无泄漏、固定target、Encoder无梯度、
Projection/M29正常梯度、Cache/Online一致、正式维度短步smoke/显存、严格checkpoint重载、
Train/Eval历史条件一致、LIBERO/Plus接口。当前只完成方案及资产验证，不自动执行这些训练任务。

官方参考：
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294/vae
- https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py
- 本地原Wrapper：OpenWAM_v2/openwam/model/video_backbone/encoder/flux2_vae.py。


## 2026-10-28 实现和启动

- 新增 `flux2_framewise` 独立 Encoder，不改动旧 `flux2_vae` 的时序池化接口；固定预训练 BN，posterior mean＋原生空间 packing，无额外归一化。
- Reader/时间网格、M29、Action Expert及全部训练超参数实测与E01/E05配置对齐，仅 Encoder/通道及投影尺寸不同。
- CUDA VAE 每批16张独立图像，尾批重复填充后丢弃填充输出；固定BN不混合视角/时间。缓存批量编码与在线单时间位置编码通过逐元素一致性检查。
- Cache 按真实时间点去重，复用预取、异步写盘、进程锁及原子发布；全量预计25.56GiB，273,465个时间点。
- Train/Eval 使用相同输出布局 `[B,128,10,14,28]`；部署只编码可观测历史/当前，缓存关闭。已补齐checkpoint结构校验及LIBERO-Plus部署资产保存。
- CPU真实数据缩小策略2步smoke、旧FLUX Wrapper 14项回归通过；全尺寸CPU严格重载、GPU四卡batch64 smoke结果由后台流程验收后才允许正式训练。

全尺寸CPU验收实测：可训练参数1,899,117,136，冻结Encoder参数34,425,856；Projection/M29梯度非零有限，Target无梯度且优化后未变化；完整checkpoint严格重载后特征差0，Train/Eval clean特征差0，部署输出 `[32,10]` 有限。结果：`reports/E06_training/full_policy_cpu_acceptance.json`。

正式启动验收（2026-10-28 10:09）：全量273,465时间点Cache通过，8个完整窗口全部命中并与在线逐元素一致；四卡每卡batch64、4步smoke通过，各rank最后一步约5.70s、峰值23.84GiB，梯度范数有限且非零。LIBERO 40任务/2,000trials与Plus 10,030任务的CPU部署计划均通过，未执行benchmark。正式运行目录：`/data/fcy/code/openwam-exp/outputs/m29_rynn9b_flux2_vae/train/s42/2026-10-28_10-09-17`；从头训练、resume/finetune为空，源代码指纹与验收计划一致。

2026-10-29 目录统一：E05/E06实现、配置、训练输出和验证报告已合并到本项目。当前使用主目录的 `scripts/experiment.py`；checkpoint、Plus部署链接及文本Cache路径已迁移到主目录，原始源码/配置保存在 `reports/workspace_unification/`。
