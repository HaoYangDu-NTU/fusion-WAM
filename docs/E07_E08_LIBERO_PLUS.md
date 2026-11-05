# E07 / E08 LIBERO-Plus 顺序评测

2026-11-03用户追加授权：当前E08标准LIBERO完整结束并记录后，自动依次执行E07、E08完整LIBERO-Plus，记录result.md，然后停止。

## 协议与独立部署

- 每个模型完整七类共10030条，每个扰动task一次官方trial，不重复标准LIBERO、不重新训练。
- 七类数量：Objects Layout 1525、Camera Viewpoints 1599、Robot Initial States 1550、Language Instructions 1537、Light Conditions 1142、Background Textures 1076、Sensor Noise 1601。
- corrected instructions=true；原final step10690权重；10步sync去噪、每chunk执行10步动作、velocity cache关闭，沿用原执行/reset协议。
- 四卡，128 workers、max batch32，渲染分布0/1/2/3，Plus端口8994–8997；与E06完整Plus的优化资源配置一致。
- 独立部署路径为 `reports/E07_E08_plus_20261103/{E07,E08}/deployment`：checkpoint和原生视觉sidecar均链接原文件，只在独立config中切换Plus corrected文本缓存。
- 文本为同一冻结RynnBrain1.1-9B avg_all_layers，缓存 `cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text`。预检核对训练fingerprint、文本规格与官方全量语言；两组均1549个唯一prompt、missing=0，不在线重编码。
- E07仍224×224独立双视角/980tokens；E08仍原生256×256独立双视角/1280tokens，不改变表示接口。

## 执行与记录

队列目录 `reports/E07_E08_plus_20261103`，脚本 `scripts/e07_e08_plus_queue.py`。前序为 `reports/E08_training_20261102`，要求DONE/complete及完整2000条标准LIBERO验收；之后获取已有共享GPU锁。E07 Plus完整成功并记录后才执行E08 Plus。

复用 `run_resumable` 的Plus-only模式，独立输出、committed task去重、失败记录与重试续测。每组验收10030条唯一有效完成记录、七类完整计数、无invalid、corrected标志和summary算术一致。不得把部分结果填为最终Overall。每组完成后更新result.md主表的Plus成功率，以及独立的E07/E08七类对比表和原始summary链接。

`benchmarks/vla_harness/run_resumable.py`新增 `--benchmarks libero-plus`，避免重跑标准LIBERO。`prepare_libero_plus_svae.py`增加SigLIP原生sidecar支持，`checkpoint_contract.py`增加SigLIP权重头部/1024D空间几何/256输入/Projection/无pooling头验证；不改变模型或当前评测配置。

```bash
/data/fcy/basics/miniconda3/envs/openwam/bin/python scripts/e07_e08_plus_queue.py \
  --queue reports/E07_E08_plus_20261103
```

同目录恢复会复用验收结果和已完成task；不得改原调用参数。失败/STOP保留记录，前序失败不放行。队列 `STOP` 只作用于自己拥有的评测子进程。源码fingerprint变化需要审计，不能覆盖原pin。

验证：两个真实final checkpoint的Plus准备与完整文本预检均通过，见 `reports/E07_E08_plus_20261103/preflight.json`；Plus-only调度、前序等待、E07→E08顺序、拒绝部分/重复类别结果、报告去重和保留标准成绩等26项检查通过。当前E08标准LIBERO进程未停止、未修改。
