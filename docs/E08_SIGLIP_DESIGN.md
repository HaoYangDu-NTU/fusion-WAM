# E08 方案：M29 + RynnBrain 1.1-9B + SigLIP-L/16

状态：2026-11-02，用户已授权E07训练和标准LIBERO结束后，接续E08训练与标准LIBERO。完整接口已实现，真实权重CPU策略、在线/缓存Loss和梯度一致性、独立checkpoint重载检查通过。全量Cache和GPU验收在E07完成后执行；当前E07继续原任务。

## 版本选择与研究含义

主方案采用原版 `google/siglip-large-patch16-256`（WebLI），不是SigLIP2。视觉塔24层、1024 hidden、16 heads、patch16，采用预训练原生256×256输入。它与E07的DINOv3-L在视觉塔层数和宽度上接近，适合比较图文对齐预训练与自监督视觉表示及各自必要适配的系统效果。两者数据、目标、分辨率等不同，不能把结果解释为只改变预训练Loss的因果消融。

本地已有 `encoder_probing/workplace/weights/siglip` 是原版SigLIP-B/16-224，不是L。主实验不将该Base权重误标为Large，也不将SigLIP2与原版混用。若以后优先追求224输入和980 tokens，可单独设计Base-224或Large-224位置插值消融，不纳入当前主方案。

权重目录：`encoder/siglip-large-patch16-256/`。官方revision `d0da9f876e7d66b4e250cd2450c3ba2ce735e447`；完整官方checkpoint为2608699800 bytes，SHA256 `e00d04afb3fb45b0113e5c79b6d6889eff5f240e9c54cfb149dab4a3e304e9a3`。下载及导出证明写入同目录provenance.json。已独立导出 `vision_encoder/`，运行时仅加载此目录，不加载SigLIP文本塔、对比logit_scale/bias和全局池化头。纯视觉权重保持原FP32，1213486816 bytes、389 tensors、303361024参数，SHA256 `2e1966a48603171af8786eaed869b1b8bf29fd820bfcf2756915a1d0803c8dda`；导出config显式 `vision_use_head=false`。

## 根据SigLIP特点选择接口

SigLIP的图文配对预训练带来语义对齐的视觉表示，但不保证每个Patch都具有同等强的局部动作可读性。采用原生空间Patch，保持预训练视觉Transformer内部完整交互；不新增图文对比Loss、全局语义分支、跨视角融合网络或显式几何模块。

提取最后一层经过原生post_layernorm的 `last_hidden_state`，每张图256×1024，对应16×16空间网格。SigLIP没有DINO式CLS/register前缀，不能使用DINO的`1+registers`截断。原生MAP attention pooling位于Patch输出之后，只产生全局pooler_output；E08不用这个全局向量。可以省去池化头的存储和计算，但必须逐值验证省略池化头不改变Patch输出。不要把SigLIP的预训练sigmoid loss误解为需要对Patch特征逐元素做sigmoid。

图像采用官方processor：各视角直接resize为256×256，BICUBIC，RGB/255后mean=std=(0.5,0.5,0.5)。不用DINO的ImageNet均值方差。主方案不对视觉位置embedding插值；224作为以后独立消融。

## Pipeline与Shape

```
当前/历史/未来双视角RGB
  → 各相机独立resize和SigLIP预处理
  → 冻结SigLIP视觉塔（把B×T×2展平为图像batch，不跨相机/时间attention）
  → last_hidden_state [B×T×2, 256, 1024]
  → 恢复 [B, 1024, T=10, 16, 32]，front在左、wrist在右
  → 原E01式逐Token无仿射LayerNorm，eps=1e-5
  → clean 2个位置：原有Conv3d空间Projection
  → future 8个位置：原有2×2 Patch→256维bottleneck→1536维
  → [B, 1536, 10, 8, 16]
  → M29 + 原Action Expert + RynnBrain 1.1-9B
```

Visual Tokens：`10×2×8×8=1280`，比E01/E07的980多30.6%；不使用Global Average Pooling，也不为了980再做非原生7×7重采样。两视角边界位于latent宽16处，与2×2 Patch对齐，不发生跨边界的Patch卷积。M29继承原宽度布局及位置编码，按实际8×16网格构造RoPE/mask。

保留SigLIP原生post_layernorm，然后用参数无关的Token级LayerNorm匹配现有Diffusion尺度；不把所有空间特征替换为单位长度的全局对比向量。该额外归一化属于WAM输入适配，需要在文档与数值检查中单独记录；它不等于官方SigLIP原始输出。

Future Diffusion Target为冻结Patch特征经过固定归一化后的网格，detach后构造，位于可训练Projection之前。可训练输入Projection不会改变监督目标。视频输出Head仍按原机制预测1024D网格的2×2 patch内容。输入Projection的参数量与E07相同；更高网格增加计算和激活，不增加M29本体参数。

## 时间与部署语义

沿用E01/E07的 `[-4,0,4,8,12,16,20,24,28,32]`：2 clean=[history,current]，8 future仅进入原有Video Diffusion Target和合法加噪路径。逐图编码，不增加跨时间编码、时间池化或未来条件。部署只在线编码真实可观测的[-4,0]双视角，未来由模型预测；episode开始时的历史填充沿用baseline。

固定原clean_condition_causal/mutual mask、clean无噪且不参与video loss、每步恢复clean等机制。缓存未来真实帧用于训练不会授权它们作为部署时的clean条件；需做改变未来RGB不影响clean输入的验证。

## 共享训练设置

M29 30层/1536D，原Action Expert，冻结RynnBrain1.1-9B avg_all_layers；scratch初始化；global batch256（4×64）、10690步、seed42；AdamW lr1e-4、betas0.9/0.95、wd0.01、clip1、cosine、warmup5%。Prediction x/x，video/action shift42/5，logit_normal，action scale1，λ_video=λ_action=1；不继承E01-LV05的0.5权重。只训练原有策略、输入Projection与输出Head。

评测继承相同LIBERO协议：final固定预算checkpoint、40 tasks×50 trials，10步去噪、velocity cache关闭、原action执行/reset协议。已加入E07之后的独立顺序队列，只安排标准LIBERO。

## Cache与所需代码适配

缓存每个episode真实timestamp的独立双视角Patch网格，存原生post_layernorm输出、位于额外固定归一化与可训练Projection之前。单条 `[1024,16,32]` BF16约1MiB；按273465个timestamp约267.06GiB，不按重叠训练窗口保存十份。预计容量是布局计算，不是已生成Cache的实测。

沿用CPU workers读取和预取、四GPU编码、有限异步原子shard写盘、mmap按需读取机制。独立cache目录与identity必须包括checkpoint SHA、视角顺序、原生256尺寸、BICUBIC、mean/std、last_hidden_state层、post_layernorm边界、dtype、代码及数据身份；冻结编码无随机增强，未来如启用RGB增强应重新设计缓存语义。

实现前已核对的现有限制：

- `lerobot_v3_reader.py` 的独立视角decode与尺寸被固定为224；`assemble_two_view_layout`同样固定224。
- `deploy/obs_preprocess.py`独立视角尺寸固定224，部署必须与训练共同适配。
- 现有image frame cache只支持dinov3/ijepa及14×28布局，不能把SigLIP仅改一个encoder名称后直接套用。

已在E08 profile中显式配置 `native_view_size=256` 和专用 `libero_native_stereo` reader，继承原LIBERO动作、历史采样、归一化和mask逻辑；独立实现原生256的decode/layout，原始RGB一次转换到256。原E01/E07 reader、224布局和缓存identity文件未修改；实际核对旧缓存code fingerprints全部匹配。部署端从保存配置读取native尺寸，缺省仍224。新增 `SiglipVideoEncoder` 遵循现有VideoEncoder接口，包含冻结模型加载、原生预处理、逐帧特征、空间Projection hooks和checkpoint sidecar重建；不使用DINO专属token截断。M29视角局部RoPE/view embedding、缓存输入网格、推理latent canvas均按Encoder原生尺寸处理，旧Encoder缺省224。

## 实现后的验收顺序

1. 官方权重hash、视觉塔strict加载、视觉输出Shape；证明无MAP头Patch输出与原版相同。
2. 输入前处理与官方processor一致；从数据源到训练、Cache、部署均为原生256。
3. Shape为B×1024×10×16×32→B×1536×10×8×16、1280 tokens；相机映射与Patch边界正确。
4. Encoder无梯度，Projection/M29/Action梯度正常，clean/future隔离，Target不随Projection更新漂移。
5. 小规模Cache与online生产batch64一致；测量容量、编码速度和读取吞吐。
6. Checkpoint strict reload与LIBERO rollout smoke，短步四卡性能；通过后再按指令完整训练。

训练时间与显存需要实测，不按较小Encoder或Token比例直接给出保证。缓存后仍有M29前反向开销，1280 tokens可能使训练慢于E07。

## 已完成的资产与接口探针

结果：[cpu_feature_probe.json](../reports/E08_siglip_assets/cpu_feature_probe.json)。官方源文件hash校验、导出纯视觉模型strict reload均通过，无missing/unexpected/mismatched keys；无池化头的Patch输出与带官方池化头输出逐值一致。4张CPU BF16输入得到 `[4,256,1024]`；按history/current和双视角恢复 `[1,1024,2,16,32]`。Projection测试将两个真实观测时间位置重复成十个位置，验证 `[1,1536,10,8,16]` 与1280 tokens，原输入Projection共7736320参数，五个参数张量均有非零有限梯度；Encoder无梯度。

该检查是权重与接口可行性探针，不是完整M29 smoke训练，不是生产GPU Cache/Online验收，也不提供训练速度估计。

后续完整接口CPU结果：[policy_cpu_smoke/smoke.json](../reports/E08_siglip_assets/policy_cpu_smoke/smoke.json)。使用真实SigLIP与真实LIBERO，策略缩减为1层、batch1；确认raw `[1024,10,16,32]`、1280 tokens，在线/缓存特征、Loss、Projection梯度逐值一致，冻结Encoder无梯度，M29与Projection梯度正常并更新。3个episode边界/普通索引metadata一致，仍需生产batch64的GPU验收。

独立部署资产重建结果：[layout_reload.json](../reports/E08_siglip_assets/layout_reload.json)。在0、64、273464三个索引，单timestamp Cache读取布局与完整窗口当前帧逐像素一致；将真实Encoder保存为临时safetensors，从checkpoint-local sidecar重建，源模型路径设为不可达后strict reload，输出逐值一致。临时大权重在验证后清理。

## 实际配置、队列与恢复

- `configs/experiments/E08.yaml`：在线接口；`E08-Cache.yaml`：正式缓存训练，继承同一模型与训练参数。
- `configs/model/video_backbone/encoder/siglip.yaml`、`openwam/model/video_backbone/encoder/siglip.py`：SigLIP接口。
- `openwam/dataloader/libero_native_stereo.py`、`transforms/native_stereo.py`：原生256读取/布局；不修改旧224 reader。
- `openwam/dataloader/spatial_frame_cache_specs.py`：新SigLIP身份，旧DINO/I-JEPA身份直接原样委派。
- `wan_backbone.py`、`deploy/obs_preprocess.py`：原生尺寸路由，旧Encoder默认224。
- 原Cache构建与验证脚本增加E08选项，沿用workers预取和异步写盘。

队列目录 `reports/E08_training_20261102`，前序 `reports/E07_training_20261102`。必须等E07 DONE/complete及2000条有效标准LIBERO结果，再获取GPU锁，构建原始特征Cache→GPU batch64逐值一致性→四卡4步smoke+checkpoint strict inference rollout→从头10690步→完整标准LIBERO→更新result.md→停止。不继承E07权重，不新增调参任务，不自动运行Plus。

```bash
/data/fcy/basics/miniconda3/envs/openwam/bin/python scripts/e07_train_eval_queue.py \
  --experiment E08 --queue reports/E08_training_20261102 \
  --after-queue reports/E07_training_20261102
```

复用已有GPU锁、独立stage进程、失败记录、完整状态断点恢复与评测去重机制；同目录重启复用已成功步骤。源码fingerprint变化需审计，不覆盖历史pin。`reports/E08_training_20261102/STOP`仅停止本队列任务。E07现有训练/评测进程和原checkpoint未停止或修改。

参考：[官方模型与预处理](https://huggingface.co/google/siglip-large-patch16-256)、[官方配置](https://huggingface.co/google/siglip-large-patch16-256/blob/main/config.json)、[SigLIP论文](https://arxiv.org/abs/2303.15343)、[Transformers原生实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)。
