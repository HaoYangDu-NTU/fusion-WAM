# 训练 → LIBERO → LIBERO-Plus 远程后台队列

两个入口共用同一套接续和断点恢复逻辑，仅选择各自的训练脚本：

| 机器 | 仓库 | 队列入口 | 训练入口 |
|---|---|---|---|
| GPU2 | `/workspace/code/OpenWAM` | `scripts/train_eval_libero.sh` | `scripts/train_libero_vjepa_1_3b_dev.sh` |
| haoyang | `/data/fcy/code/OpenWAM_v2` | `scripts/train_eval_libero_share_dir.sh` | `scripts/train_libero_vjepa_1_3b_dev_share_dir.sh` |

在 GPU2 的 tmux 中可直接启动新实验：

```bash
cd /workspace/code/OpenWAM
bash scripts/train_eval_libero.sh
```

在 haoyang 的 tmux 中可直接启动新实验：

```bash
cd /data/fcy/code/OpenWAM_v2
bash scripts/train_eval_libero_share_dir.sh
```

两个入口均在无参数时自动新建 `outputs/train_eval/exp_YYYYMMDD_HHMMSS` 并打印完整队列路径。恢复已有队列时，仍需把原队列目录作为参数传入；无参数调用会创建新实验。

需要指定目录或使用 nohup 时，原来的用法仍然有效：

```bash
cd /data/fcy/code/OpenWAM_v2
QUEUE=outputs/train_eval/experiment_$(date +%Y%m%d_%H%M%S)
mkdir -p "$QUEUE"
nohup bash scripts/train_eval_libero_share_dir.sh "$QUEUE" > "$QUEUE/launcher.log" 2>&1 < /dev/null &
```

流程完全在远程运行，断开 SSH 或关闭本地电脑不影响。入口依次执行：

1. 等待 GPU 0–3 和 8990–8993 端口空闲，运行一次对应机器的训练脚本。
2. 训练正常退出后，从本次日志的 `Saved final` 识别实际 run、最终 checkpoint 和步数；核验 N/N 完成记录、权重文件、保存资产和训练退出码。
3. 使用已有测评入口完成完整 LIBERO，再完成完整 LIBERO-Plus。测评非完整退出会自动续跑，直到两个完整结果和 `DONE` 均落盘才返回 0。

**不绑定模型设计。** 队列不指定 text encoder、bottleneck、融合方法或训练步数，也不覆盖训练参数和输出目录；训练取对应机器的当前训练脚本及调用环境，测评取本次保存配置和仓库当前的测评代码。每次实验只需维护原训练脚本、对应的测评适配和文本缓存。队列不会自行生成缺失缓存或修改模型代码；持续的代码/缓存错误会保留日志并按 30 秒间隔重试，修复后下一次自动接续。训练脚本若在排队期间变化，会停止以避免启动另一配置；这时应使用新队列目录。

Python 环境由当前仓库位置推导：GPU2 为 `/workspace/basics/miniconda3/envs/openwam/bin/python`，haoyang 为 `/data/fcy/basics/miniconda3/envs/openwam/bin/python`，也可显式设置 `OPENWAM_PYTHON`。haoyang 的 `/data` 与 GPU2 的 `/team_data` 是共享存储的不同挂载名，但 GPU2 的运行仓库在 `/workspace/code/OpenWAM`；队列直接调用本机训练/测评入口，不把另一台机器的绝对路径写入配置，也不自动替换已有 checkpoint 的路径。队列记录实际训练入口及哈希，恢复需使用原机器、原仓库、原队列目录和原入口，防止把两台机器的启动记录混用。旧 haoyang 队列仍可从 share_dir 入口恢复。

沿用测评协议：LIBERO 40 × 50、correct=false、32 workers / batch 8 / GPU 3 渲染；Plus 全部 7 类 10,030 task、correct=true、128 workers / batch 32 / GPU 1,3 渲染。两者使用四卡、raw、同步 10 步。新 run 的 `vla_harness_deploy.yaml` 从当前部署模板生成并关闭 velocity cache；已存在的部署文件保留。测评阶段清除 `TEXT_EMBEDDING_DIR`、`PLUS_DEPLOY_DIR` 和 `CORRECT_INSTRUCTION_SUFFIX` 环境覆盖，避免误用另一模型的缓存或部署目录。

每个队列只训练一次。训练失败或进程消失但缺少退出记录时停止并保留证据，不会自动重训。运行时的独占锁防止同一目录重复启动，同一仓库的多个队列按 GPU 锁依次执行；对于其他实验只等待资源释放。训练 worker 和测评监督进程在父进程意外退出后仍能继续。若整个机器/容器重启，需手动重新执行同一条启动命令并使用**原队列目录**：训练已成功则直接恢复测评，训练状态不明则停止核查。

查看状态和日志：

```bash
cat "$QUEUE/status.json"
cat "$QUEUE/run.json"                  # 训练结束后生成，指向本次实际 run
tail -f "$QUEUE/training/job.log"      # 训练输出
cat "$QUEUE/evaluation/status.json"   # 当前 benchmark、完成任务数、attempt
tail -f "$QUEUE/evaluation/supervisor.log"
cat "$QUEUE/evaluation/libero/RESULTS.md"
cat "$QUEUE/evaluation/libero-plus/RESULTS.md"
```

`training/` 保留命令、脚本 hash、worker/训练 PID 与启动时间、训练日志及退出码；`evaluation/` 保留每次测评的命令、退出码、原始 DB 和完成任务。`status.json` 的 `complete` 及队列根目录的 `DONE` 表示整个流程完成。

停止：`touch "$QUEUE/STOP"`，也支持 run、evaluation 及单项 benchmark 下的 STOP。STOP 会终止本队列启动的工作并非正常成功退出。若停止发生在测评阶段，移除自己创建的 STOP 后以**相同队列目录**重新运行即可接续；训练阶段停止后不会自动重训。不要删除启动记录、`completed/` 或历史 attempt。

## 仅对已有权重执行后台断点测评

```bash
bash scripts/eval_libero_then_plus.sh CKPT_DIR CKPT_NAME OUTPUT_DIR
```

外层 shell 监督控制进程，控制进程监督现有 `eval_libero_vjepa.sh` / harness。默认使用 GPU 0–3、32 workers、batch 上限 8、GPU 3 渲染、raw 无损传输。先执行完整 LIBERO（40 task × 50 trial），全部完成后执行完整 LIBERO-Plus（10,030 task × 1 trial）。Plus 默认使用修正指令文本，`correct_instruction_suffix=True`。旧原始指令队列恢复时需显式传入 `--correct-instruction-suffix false`；不会混用两种协议的结果。

- 每个完整 task 都保存到 `OUTPUT_DIR/<benchmark>/completed/`，含该 task 全部 episode 结果和来源 attempt。正常完成但 success=false 也永久保留。
- LIBERO 只有 0–49 共 50 条唯一、无基础设施错误的 trial 才算完成。未完成 task 从 trial 0 重启，保留任务内部连续 RNG 语义；不拼接不同 attempt 的部分 trial。
- Plus 的 task 只有一次 trial。已完成任务在后续所有重启中都会被排除。
- 每 10 秒读取已提交 SQLite/WAL 记录，原子保存完成任务；控制进程重启还会扫描历史 attempt，恢复尚未归档的提交。每次 attempt 的日志和原始 DB 均保留。
- 任意非完整退出（包括退出码 0 但任务未跑齐）均在 30 秒后继续未完成任务，不限制重试次数。不修改代码，不自动降低 batch 或更改配置，不把异常计为模型失败。
- 控制进程被杀后，外层 shell 会重新启动它。它先根据该任务独有的环境标记清理遗留进程，避免重复模型和端口冲突；不按 GPU 或进程名称杀其他任务。
- checkpoint 路径、配置哈希、权重文件状态和启动设置绑定到 `job.json`，防止混入另一模型的结果。

完成后，每个 benchmark 根目录包含跨 attempt 去重后的 `summary.json`、`RESULTS.md`、suite 汇总；Plus 另有 category 汇总。两项都完成后生成 `OUTPUT_DIR/DONE`。

查看进度：

```bash
cat OUTPUT_DIR/status.json
tail -f OUTPUT_DIR/supervisor.log
```

显式停止（最多约 10 秒被控制进程发现，随后清理本次进程）：

```bash
touch OUTPUT_DIR/STOP
```

继续：移除自己创建的 `STOP` 文件，再用完全相同的启动命令运行。不得删除 `completed/` 和历史 attempt。服务器/容器停止会同时终止 tmux 和监督进程；机器恢复后重新执行同一命令，已保存完成任务仍会跳过。

## 2026-09-30 实际启动

- SSH：`haoyang-pretrain`
- 仓库：`/data/fcy/code/OpenWAM`
- 模型环境：`/data/fcy/basics/miniconda3/envs/openwam/bin/python`
- Plus 环境：`/data/fcy/basics/miniconda3/envs/libero-plus/bin/python`
- checkpoint：`outputs/libero_vjepa_1_3b_dev_vv_av_s12.0/2026-09-30_02-09-59/checkpoint_step_10690.safetensors`
- tmux：`openwam_eval_10690`
- 输出目录：`/data/fcy/code/OpenWAM/outputs/sequential_eval/vjepa_1_3b_step10690_20260930`
- 精确启动命令：输出目录的 `launch_command.txt`。
- 自动化验证：75 项测试通过，包括正常失败任务永久保留、未完成任务重试、不跨 attempt 拼接 trial、SQLite WAL 恢复、控制进程崩溃重启，以及进程清理隔离。

只读启动核验命令（不会新建或停止测评）：

```bash
ssh haoyang-pretrain 'tmux has-session -t openwam_eval_10690 && cat /data/fcy/code/OpenWAM/outputs/sequential_eval/vjepa_1_3b_step10690_20260930/status.json && tail -20 /data/fcy/code/OpenWAM/outputs/sequential_eval/vjepa_1_3b_step10690_20260930/supervisor.log && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

模型副本依次加载。阶段为 libero 且 GPU 已分配，只证明后台任务启动；完成数量与最终成功率以持久化记录和汇总为准。
