# E07：M29 + RynnBrain 1.1-9B + DINOv3-L/16

2026-11-02 用户停止 E01 后续调参：当前 E01-LV05 的标准 LIBERO 继续完成并记录，E01-LV2 未训练且已取消。之后转入 E07，不再自动搜索超参数。

## 模型与参数

| 项目 | E07 |
|---|---|
| 视觉 Encoder | 官方 DINOv3 ViT-L/16 LVD-1689M，1024D、24层、16 heads、4 registers；完整冻结/eval/no_grad |
| 文本 | 冻结 RynnBrain 1.1-9B，avg_all_layers，128×4096 离线文本特征 |
| 图像与视角 | front/wrist 各224×224，独立编码后沿宽度拼接，ImageNet预处理 |
| 时间位置 | [-4,0,4,8,12,16,20,24,28,32]；2 clean＋8 future |
| 原始特征 | B×1024×10×14×28；只使用最终层 Spatial Patch Features，去除CLS/register输出 |
| 空间 Projection | 原有(1,2,2)；clean直接投影到1536，future经256维线性bottleneck再到1536 |
| M29输入 | B×1536×10×7×14，980 visual tokens |
| 策略主干 | 原M29 30层/1536 hidden，原Action Expert；scratch初始化 |
| Prediction / Noise | x/x；video/action shift42/5；logit_normal；action scale1 |
| Loss | λ_video=1、λ_action=1；原有mask、scheduler和target定义 |
| 训练 | global batch256（4×64）、seed42、10690 steps；AdamW lr1e-4、betas0.9/0.95、wd0.01、cosine、warmup5%、clip1 |
| 实现 | BF16、ZeRO1、gradient checkpointing；4 workers/rank、每worker2线程 |
| 评测 | 标准 LIBERO 40 tasks×50 trials；固定final checkpoint；10步去噪；velocity cache关闭；不自动运行Plus |

E07采用原E01 baseline设置，不继承LV05的λ_video=0.5，也不继承S12的shift12。更小的Encoder宽度自然改变输入Projection和视频输出Head的参数量，M29/Action Expert本体不变。实验比较的是DINOv3-L与H+的预训练表示及必要通道适配。

沿用已有可根据HF config自动读取hidden_size/patch_size的 `DinoV3VideoEncoder`，不复制模型代码、不新增融合模块。模型完整配置为 `configs/experiments/E07.yaml`；缓存训练配置为 `configs/experiments/E07-Cache.yaml`，共享 `configs/protocol.yaml`。

## 权重

目录：`encoder/dinov3-l`。官方来源：<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>，revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`。

公开副本来自 `camenduru/dinov3-vitl16-pretrain-lvd1689m` revision `3c276edd87d6f6e569ff0c4400e086807d0f3881`。仅在下载后完整SHA256与官方Hub LFS metadata一致时接受，不使用Phoenix微调权重。文件 `model.safetensors` 为1212559808 bytes；SHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`。下载与校验来源保存在 `encoder/dinov3-l/provenance.json`，资产登记于 `assets/registry.json`。

## Cache与顺序队列

队列 `reports/E07_training_20261102`，脚本 `scripts/e07_train_eval_queue.py`：

1. 等待当前 `reports/E01_LV05_cache_w2_20261101` DONE/complete，并核验完整标准LIBERO结果。
2. 获取共享GPU锁，在独立目录 `cache/dinov3_l_libero_frames` 生成冻结原始特征。双视角每个真实timestamp编码一次，CPU预取、异步原子shard写盘沿用原机制，不缓存可训练Projection。
3. 验证完整缓存及在线生产batch64的BF16逐值一致性。不通过则停止，不放行正式训练。
4. 四卡4步训练smoke，检查有限loss/有效梯度，保存checkpoint；严格重载并完成一个真实LIBERO rollout。
5. 从头完整训练10690步，之后标准LIBERO 2000 trials，记录四suite和Overall至result.md，再停止。

通道适配在 `openwam/dataloader/frame_cache_specs.py` 中由HF config读取。原E01/E02的identity保持逐字段不变，旧cache文件和原编码实现不修改；E07使用独立1024D cache identity。

cache单timestamp大小为1024×14×28×2 bytes=784KiB；总容量约204.46GiB（273465个timestamp），不按重叠窗口重复存10份。实际数量与容量由构建脚本estimate输出核实。

失败记录、GPU锁、自己的checkpoint/full-state断点恢复与结果去重复用已有Queue机制。重新启动同一脚本会复用已成功阶段；源码fingerprint变化时停止等待审计。用户STOP只作用于本队列拥有的进程。结果见 `reports/E07_training_20261102/RESULTS.md`。

```bash
/data/fcy/basics/miniconda3/envs/openwam/bin/python scripts/e07_train_eval_queue.py \
  --queue reports/E07_training_20261102 \
  --after-queue reports/E01_LV05_cache_w2_20261101
```
