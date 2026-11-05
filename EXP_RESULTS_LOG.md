# OpenWAM encoder 对比实验结果

本文件只记录 openwam-exp 协议下的新实验。2026-10-22：infra 已实现，433 项 CPU 检查通过，见 reports/cpu_validation.log；9B 文本 cache 与真实四卡 smoke 待完成，尚未启动本工作区正式训练。

历史 Cosmos3 Nano、RynnBrain 2B、DexJoCo 和旧 I-JEPA 成绩不填入本表。三个 encoder 各用原生时序接口与归一化；不要求等 token 或等 FLOPs。

| ID | Text | Visual / 原生时序 | Tokens | Seed | Final step | LIBERO / 2000 | Plus / 10030 | Samples/s | 状态 |
|---|---|---|---:|---:|---:|---|---|---|---|
| E00 | RynnBrain 9B | V-JEPA 2.1 G / tubelet | 490 | 42 | — | — | — | — | CPU 验证；GPU 待验收 |
| E01 | RynnBrain 9B | DINOv3 H+ / 2 clean + 8 future | 980 | 42 | — | — | — | — | CPU 验证；GPU 待验收 |
| E02 | RynnBrain 9B | I-JEPA G / 门控差分 + 成对均值 | 490 | 42 | — | — | — | — | CPU 验证；GPU 待验收 |

E02 额外训练差分投影、门控标量和 affine LN，其余视觉 backbone 冻结。E01 是主实验原生 980-token 设计，不另设共享 490-token 或 S01 替代。

登记每个 run 时附：目录、seed、实际样本与 optimizer steps、experiment identity、权重 SHA256、cache fingerprint、训练环境和参数数目、吞吐/peak memory/GPU hours、resume 状态，以及评测 summary/SQLite、完整计数、错误数与退出状态。43/44 的复验完成后再报告 mean/std。
