# OpenWAM LIBERO on vla-evaluation-harness

此目录把 OpenWAM 官方 LIBERO 协议作为独立的 `OpenWAMLIBEROBenchmark` 接入 harness。使用现有 openwam 环境和现有 LIBERO 源码。harness 完整副本在 `third_party/vla-evaluation-harness`；原始 `/workspace/code/vla-evaluation-harness` 不变。

LIBERO-Plus 使用独立的 `OpenWAMLIBEROPlusBenchmark`，入口增加 `--benchmark libero-plus`。它保留官方 Plus 的每任务新进程、全局 seed=10000、每任务 1 trial，以及七类扰动统计。环境准备、文本缓存和启动命令见 [LIBERO_PLUS.md](LIBERO_PLUS.md)。下面的 2,000 次协议和已测成功率均指标准 LIBERO。

模型路径支持当前 **V-JEPA 2.1 + Wan TI2V、sync、CFG=1、EEF10** checkpoint。正式测评保留官方 10 步联合去噪、每次执行前 10 个 action、每个 task 50 个 trial、四个 suite 共 2,000 个 episode。视频流参与联合去噪，V-JEPA 不执行像素解码。

## 启动

### 公共动作队列与性能统计（2026-11-05）

OpenWAM/M29、FM-VJEPA 和 VPP2 共用 `EvaluationModelServer` 与 worker 侧
`EvaluationConnection`。在原有 `run_eval` 或 `run_resumable` 命令后添加
`--client-action-queue`，即可由 worker 消费动作块，只在重新规划时上传观测。
原路径仍为默认，便于配对验证；正式评测的 seed、步数上限、逐步 done 检查、
动作执行长度和计分循环保持各模型的既有协议。

模型通过握手声明执行长度和历史帧偏移；worker 每个控制步更新独立图像历史。
M29 使用自身 Encoder 的条件帧配置，FM 执行16步且读取前4步历史，VPP2执行10步。
块级请求携带 episode token 和控制步号，重置时清空缓存；连接中断时当前 episode
报错，避免将动作请求自动重放到丢失状态的新会话。

各进程的累计阶段统计保存在本次 attempt 的 `performance/*.jsonl`，包括环境初始化、
环境 step、图像准备、模型 batch 和请求往返。汇总时只取每进程最后一个累计快照：

```bash
python -m benchmarks.vla_harness.performance /path/to/attempt
```

统计为 CPU wall time，重叠阶段不能直接相加，也不等于 CUDA kernel 耗时。
不同模型的最优 workers/batch 仍须在真实渲染负载下分别测量。

不加载模型和模拟器的通信回归探针：

```bash
MSGPACK_PUREPYTHON=1 python -m scripts.profile_evaluation_transport \
  --history-offsets=-4,0 --output outputs/evaluation_runtime_validation/transport_history.json
```

该探针使用真实 WebSocket 和替身策略，核对预测输入、历史与动作，并测量通信路径。
其速度不能作为机器人闭环评测或真实模型的加速结果。实施进度和完整计划见
[框架优化方案](../../docs/ROBOSCOPE_EVALUATION_SPEED_PLAN.md)。

在 H100_gpu2-4gpus 上执行：

```bash
cd /workspace/code/OpenWAM
/workspace/basics/miniconda3/envs/openwam/bin/python -m benchmarks.vla_harness.run_eval \
  --gpus 0,1,2,3 --workers 32 --max-batch-size 8 --max-wait-time 0.01 \
  --render gpu --render-gpus 3 --image-format raw
```

默认 checkpoint 目录：

```
outputs/libero/vjepa21_cached_text_step10690_20260918_125937/checkpoint_for_eval
```

它是当前训练 checkpoint 的部署配置目录；其中的 `checkpoint_step_10690.safetensors` 链接到原始训练权重。可用 `--ckpt-dir`、`--ckpt-name` 指定其他已准备好的部署目录，目录中须有权重、`config.yaml` 和 `normalization_stats.npy`，并满足原生部署加载器的要求。

默认传输使用 harness 自带的 raw 图像格式：保留图像像素和 state，省去 PNG 编解码。launcher 的评测子进程设置 `MSGPACK_PUREPYTHON=1`，使用已安装 msgpack 的 Python 实现；不替换环境中的包。可用 `--image-format png` 选择另一种无损格式。

LIBERO 与 Plus 默认只保存每次试验的 outcome、任务身份、步数及耗时，用于成功率统计、完整性检查和续跑；不保存逐步模型轨迹、视频或导出逐步 JSONL。需要调试轨迹时显式加 `--export-step-jsonl`，它同时开启逐步记录和导出。正式 LIBERO 仍是每任务连续 50 次、共 2,000 次，seed、初始状态、动作和测评协议不变。

2026-09-20 完整验证：上述 32 workers / batch8 配置完成 2,000 / 2,000 次，零基础设施错误，总成功率 93.60%。模型加载后从首个测评客户端连接到最后一次 episode 结束共 74 分 6.7 秒（不含模型加载及事后汇总；纯 EPISODE_START→END 窗口为 73 分 33.2 秒），cgroup 内存采样峰值 705.395 GiB（限额 800 GiB，含已有文件缓存）。详见 [优化及验证记录](../../OPTIMIZATION_LOG.md)；不把短测速当作完整用时或全局最优结论。

这些参数已设为默认值，可直接运行 `python -m benchmarks.vla_harness.run_eval`。此默认布局针对 H100_gpu2-4gpus；其他机器应显式选择有效的 GPU 编号。

每张 GPU 一个模型实例，默认每个实例连接 8 个环境 worker；`max-batch-size=8` 是动态 batch 上限，实际 batch 取决于等待推理的请求数。GPU 0–3 都运行模型，GPU 3 同时承担全部环境的 EGL 渲染。不使用 DDP/DeepSpeed。降低 workers 或 batch 上限可以减少资源占用；这两个参数分别控制环境并发数和一次模型 forward 的样本数。

2026-09-20 的 40 worker 长程运行未完成，现默认改为 32 worker / batch8，以增加内存余量。四个模型现在并行加载，全部健康后才启动环境；加载后调用 GC 和 glibc malloc_trim，释放已不用的 CPU 分配。评测子进程使用 MALLOC_ARENA_MAX=2。每 15 秒写入 runtime_memory.jsonl（cgroup 总内存/匿名内存/文件缓存、事件计数、进程 RSS/PSS），意外退出时先保存 process_failure.json 的 PID、退出码和信号，再清理其他进程。启用 PYTHONFAULTHANDLER 记录 Python 支持的原生致命信号堆栈。此监控仅用于评测，不进入训练入口或模型 forward。

模型的加载和推理固定在一个专用线程。harness 的 AnyIO 限流器保证调用不重叠，但原来的真实通信压测中，推理会在 8 个共享线程间轮换；专用线程消除这项线程迁移，保留上游队列、动态组批和通信/推理重叠。该处理不修改模型计算，不能仅凭短测将稀有类型异常的底层根因认定为线程迁移。

监控还会检查记录库：遇到模型、通信或环境异常，保存 episode_failure.json 并停止该轮；正常执行到上限但 success=false 仍是有效测评。不会把程序报错当作模型失败或跳过该 trial。

只查看启动计划，不加载模型、不创建环境：

```bash
/workspace/basics/miniconda3/envs/openwam/bin/python -m benchmarks.vla_harness.run_eval --dry-run
```

短程联调（两个 task，各一个 trial，最多 20 个模型动作；不生成正式成功率）：

```bash
/workspace/basics/miniconda3/envs/openwam/bin/python -m benchmarks.vla_harness.run_eval \
  --gpus 0 --workers 2 --max-batch-size 2 --suites libero_spatial --smoke --render cpu
```

`--render gpu` 是默认值，使用 EGL，并默认选择已验证的物理 GPU 3。2026-09-20 在四卡空闲后，轮流使用 GPU 0/1/2/3 渲染的探针仍在 GPU 0 原生中止；统一使用 GPU 3 的参数测试正常。该设置绕开当前机器不稳定的 EGL 设备，不代表底层驱动问题已修复。`--render cpu` 通过已有的隔离 OSMesa 库运行环境渲染，模型仍在 GPU 上。两种 renderer 的像素可能存在差异，CPU 渲染的成功率不能称为与 EGL 逐像素等价。正式对比应固定 renderer，并记录它。

可用 `--render-gpus` 显式覆盖物理渲染设备；只在 `--render gpu` 下有效。CPU 模式仍可单独使用 `--render cpu`，不必指定 `--render-gpus`。不要把 GPU 逻辑编号与物理 EGL device 编号混淆。

输出默认在：

```
outputs/vla_harness/YYYY-MM-DD_HH-MM-SS/
```

启动后打印实际路径。终端运行时 Ctrl-C 会清理该 launcher 创建的模型和 worker 进程组；不会终止已有训练。每次运行创建新目录，不混用旧结果，也不做跨 trial 的自动续跑。

运行中的 SQLite 数据库放在 `/workspace/code/OpenWAM/vlaharness_tmp/run-*/`。这个入口是指向本机 `/tmp` 下 XFS 目录的软链接：本机 `/workspace` 属于 DPC 分布式存储，直接在其上使用多进程 SQLite WAL 曾导致 `disk I/O error`。每轮 outputs 目录中的数据库链接到该临时位置；所有 worker 退出后，launcher 用 SQLite backup API 备份、检查完整性，并将 outputs 中的链接替换为独立数据库文件。正常结束或 Ctrl-C 均执行备份；临时原件保留以便恢复。进程被 SIGKILL 或机器断电时仍需从临时原件恢复，因此运行中不要清理 `vlaharness_tmp`。

## 2026-09-20 参数短测（早先的 40/10 方案）

使用原 step10690 checkpoint、现有 openwam 环境；没有安装或替换包。短测每个 task 的 trial 0 最多 120 个控制动作，成功提前结束，另保留官方 30 步 settle。模型已预热，表中速率为整个 rollout 窗口的总动作步数/墙钟时间，不包括模型加载和环境初始化，也不是全量评测完成时间。

| 推理布局 | 环境数 | batch 上限 | 等待 | renderer | 动作步/秒 |
|---|---:|---:|---:|---|---:|
| 四卡 | 16 | 4 | 10 ms | CPU | 31.47 |
| 四卡 | 32 | 8 | 10 ms | CPU | 56.24 |
| 四卡 | 32 | 8 | 10 ms | GPU 3 | 134.51 |
| 四卡 | 40 | 4 | 10 ms | GPU 3 | 138.85 |
| 四卡 | 40 | 10 | 0 ms | GPU 3 | 150.64 |
| 四卡 | 40 | 10 | 10 ms | GPU 3 | 154.60 |
| 四卡 | 40 | 10 | 50 ms | GPU 3 | 140.51 |
| GPU 0/1/2 推理 | 40 | 14 | 10 ms | GPU 3 专用 | 140.47 |

上表使用原 PNG + C msgpack 路径。长程启动时曾出现服务端数组解码 `KeyError: shape`，该轮已停止并保留异常结果，不能用于正式成功率。三组各 16,000 次独立编解码压测均通过，尚未复现底层根因。现通过依赖自带的 Python 实现规避原生解码路径，并补测无损 raw 传输：

| 推理布局 / renderer | 环境数 | batch 上限 | 等待 | 传输 / msgpack | 动作步/秒 |
|---|---:|---:|---:|---|---:|
| 四卡 / GPU3 | 40 | 4 | 10 ms | raw / Python | 184.62 |
| 四卡 / GPU3 | 40 | 10 | 0 ms | raw / Python | 182.01 |
| **四卡 / GPU3** | **40** | **10** | **10 ms** | **raw / Python** | **193.60** |

此前短测选择了第三行；长程失败后，当前默认改为上文的 32/8 内存余量方案。同一 40 任务短测中，该组合相对 PNG/C 的 154.60 步/s 提升约 25.2%；这是组合切换的结果，不单独归因于某一库或某一个设置。23 项回归测试通过，其中 raw/PNG 经过真实序列化后与官方图像像素、state、canvas 和 action 对照一致。这个兼容措施不代表已确认或修复第三方底层 bug。

40 环境各行使用相同 40 个任务及正式 launcher 的模型路由。16/32 环境行是覆盖四个 suite 的子集，用于环境并发和 renderer 筛选；不同任务子集之间的速率不能作为严格的加速倍数。相同 32 任务中，EGL 比 CPU 渲染约快 2.39 倍；相同 40 任务中，batch 上限 10 比 4 约快 11.3%。不同 batch shape 和 renderer 可能使闭环轨迹略有差异，因此完成动作总数并非完全相同。

这是本机已测配置中的选择，不宣称遍历全部配置或能复现其他模型的“18 分钟”。0 ms 与 10 ms 的单轮差距较小，保留原 10 ms 默认；50 ms 和较小 batch 上限在本轮测试中更慢。任务提前完成后，并发数和实际 batch 会下降，完整 50 trials/task 的耗时需以正式运行日志为准。

最终 raw/Python 方案分段结果（40 环境 / batch 10 / 10 ms）：观测转换中位 4.10 ms，环境 step 中位 39.07 ms；含缓存和新推理的 RPC 平均 124.81 ms。每 10 步触发的新预测 RPC 中位 1.147 s，剩余缓存动作 RPC 中位 3.71 ms；原 PNG/C 的缓存动作 RPC 中位为 74.73 ms。RPC 包括传输、编码、排队和模型计算，不能直接等同于纯 forward 时间。正式入口没有加入这些逐步计时；计时仅位于独立 `profile_output/vla_harness_tuning_20260920/` 脚本。

完整 episode 复核与正式运行状态见 `OPTIMIZATION_LOG.md` 最新记录。原始参数、逐步数据、失败探针和统计在 `profile_output/vla_harness_tuning_20260920/`。

## 调用关系和数据

```text
run_eval.main → make_plan → launch
  ├─ 每张卡：model_server → harness run_server/serve_async
  │    └─ OpenWAMModelServer
  │         ├─ load_native_server → build_server_from_config → _init_policy
  │         └─ harness PredictModelServer 请求队列 / 动态 batch / session 动作缓存
  │              └─ predict_batch(obs_batch, ctx_batch)
  │                   ├─ 原生 ObsPreprocessor.preprocess
  │                   ├─ BatchedOpenWAMEngine.generate_batch
  │                   └─ 原生 native_eef10_to_libero7d
  └─ 每个 worker：worker.main → harness CLI run → Orchestrator
       └─ OpenWAMLIBEROBenchmark + SyncEpisodeRunner
            ├─ get_tasks / shard_work_items
            ├─ start_episode → reset
            ├─ get_observation → make_obs
            ├─ Connection.act → WebSocket 模型请求
            ├─ apply_action → step → LIBERO env.step
            └─ check_done / get_step_result → recorder / SQLite
  最后：harness merge_eval → summarize.write_summary
```

| 入口 | 输入 | 输出 / 行为 |
|---|---|---|
| `get_tasks()` | 官方四个 suite，或显式选择 | task 字典含 suite、task_id、language、原始 task、init_states |
| `reset(task)` | task 及 episode_idx | 原始 LIBERO obs；每个 task 建一次 env、seed(42)，trial 顺序执行 reset → set_init_state → 30 步 settle |
| `make_obs(raw_obs, task)` | agentview、腕部图像，EEF pose、gripper qpos | lossless RGB arrays、prompt、achieved EEF10 state |
| `predict_batch(obs_batch, ctx_batch)` | B 个观测及各自 session/episode 标识 | B 个 (10,7) action chunk；harness 每次只向相应环境发送一条 |
| `_prepare(observations)` | 原生预处理后的 PIL canvas、prompt、state | B 维文本、参考帧 latent、视频/动作初始噪声、proprio，以及官方 schedule |
| `generate_batch` | B 个准备好的观测 | B 个反归一化后的模型动作序列，各为 (T,10)；再由官方转换函数变为 LIBERO 7D |
| `step(action)` | 一条 float32 (7,) action | 原始 obs、reward、done、info；不额外二值化 gripper |
| `get_step_result` | 最后一次 StepResult | success=env.done；达到步数上限本身不算成功 |

双视角顺序、旋转、缩放和 L-shape canvas 均沿用官方代码。head 为 `agentview_image`，left wrist 为 `robot0_eye_in_hand_image`，right wrist 为空并在原生预处理中补黑。state 是 achieved EEF10；输出动作是 native delta EEF10，经官方函数转为 OSC 7D。gripper 保留官方连续值、符号转换与裁剪规则。

## harness 加速如何保留

- 保留上游 WebSocket/msgpack 和图像编解码、请求队列、动态 batch、收集请求与推理重叠、session 隔离、动作 chunk buffer、环境 worker、记录与合并。
- 每个 GPU server 真正合并模型输入：V-JEPA reference 编码及 WAM 联合 forward 都有 B 维。不会用 B 次单样本生成冒充 batch。
- 官方每次生成都重置 seed=42。batch 中每个样本复用各自对应的官方初始噪声，不能直接一次生成 (B,T,D) 而改变后续样本随机数。
- DiT velocity cache 按样本独立判断。某一步只有部分样本需要重算时，将这些样本合成一个子 batch；其他样本使用各自缓存。
- 文本使用 checkpoint 中已有离线 embedding cache，仍通过官方加载和查找接口；不重新加载 text encoder。
- 使用 sync runner。上游 live / continuous inference / LAAS 实现完整保留在副本中，但不为此 benchmark 启用，因为按墙钟时间跳 action 或推进物理步会改变官方测评协议。
- 执行动作只在 harness 的 session buffer 中缓存；原生 WAMPolicy 作为加载/对照入口保留，但不用于处理实际批量请求，避免第二套共享动作队列。

## 唯一上游修改：按整个 task 分片

官方 `reseed_each_trial=false`，同一 task 的 trial 共享其环境随机数序列。harness 默认把 episode 打散后分片，这会打乱该序列。

在副本 `src/vla_eval/orchestrator.py` 原分片调用处新增可选 `benchmark.shard_work_items` 钩子。此 benchmark 将每个 task 的全部 50 个 trial 作为一组，调用上游确定性 shuffle/分片算法，再展开为 trial 0..49。其他 benchmark 未定义该钩子时仍走原有逻辑。

因此四个 suite 的 40 个 task 共同调度，最多有 40 个环境 worker。一个 task 只归一个 worker，任务较早完成后会有尾部负载不均衡；为了保留 RNG 语义，不在中途把剩余 trial 搬到其他 worker。发生基础设施错误的 task 应从 trial 0 重新运行，不把缺失 trial 当失败率填充。

## 结果

默认保留包含逐任务 outcome 的 SQLite 和 `OpenWAMLIBERO_aggregate.json`；只有显式启用 `--export-step-jsonl` 才保存逐步记录并导出 episode JSONL。另写：

- `run.json`：checkpoint、Python、LIBERO 路径、上游版本、官方 policy 配置哈希、GPU/worker/task 分配。
- `model_N.log`、`worker_N.log`。
- `summary.json`、`suite_success_rates.csv`、`RESULTS.md`。

每个 suite 必须有 task 0..9 × trial 0..49 的 500 条唯一有效记录，才输出该 suite 成功率。四个 suite 全部完成才计算 2,000 个 episode 的总平均；它也等于四个 suite 成功率的算术平均。缺失、重复、环境/通信错误和 smoke run 不输出正式总分。

进程停止后可重新汇总已有 SQLite：

```bash
# EVAL_ID 取自本次 run.json；OUTPUT_DIR 是本次实际输出目录
/workspace/basics/miniconda3/envs/openwam/bin/python -m vla_eval.cli.main merge \
  --config OUTPUT_DIR/config.yaml --eval-id EVAL_ID
/workspace/basics/miniconda3/envs/openwam/bin/python -m benchmarks.vla_harness.summarize OUTPUT_DIR
```

## 验证与范围

CPU 回归测试：

```bash
CUDA_VISIBLE_DEVICES='' /workspace/basics/miniconda3/envs/openwam/bin/python -m pytest \
  tests/test_vla_harness_openwam.py tests/test_vla_harness_batch_engine.py -q
```

包含官方像素/state/action 对照、真实 harness WebSocket 多 session chunk 隔离、动态 batch、task 分片、suite 分母与错误处理，以及使用原生 `BaseWAMArchitecture.generate` 作为 oracle 的去噪/cache/噪声/归一化对照。真实 checkpoint、环境联调结果与速度数据见仓库根目录 `OPTIMIZATION_LOG.md` 最新记录，证据在 `profile_output/vla_harness_batch_20260919/`。

本机存储兼容检查为 `tests/test_vla_harness_recording.py`：40 个真实子进程通过上游 RecordingStore 写入 2,000 条明确标记为 fixture 的记录，验证 WAL/SHM 留在本地、备份完整性及上游合并。该测试不运行模型，其合成成功率不属于评测结果。

2026-09-19 验证结果：66 项自动测试通过；真实 checkpoint 的 batch=1 与官方输出完全一致，batch=2 的两请求模型吞吐样本提升 1.73×；四卡/16 环境/CPU 渲染短程联调完成 16 条各 20 步的有效 episode，零基础设施错误。正式全量成功率尚未测量。

理论等价指相同的观测和动作定义、去噪规则、cache 判断规则、环境步进与成功率协议。GPU batch shape 改变可能带来浮点舍入差异；不承诺闭环轨迹或最终成功率逐位一致。训练入口、官方单样本 generate、官方 LIBERO 测评源码均未因本次集成而修改。

## 新旧 V-JEPA 模型统一入口

```bash
bash scripts/eval_libero_vjepa.sh CKPT_DIR CKPT_NAME --smoke
bash scripts/eval_libero_vjepa.sh CKPT_DIR CKPT_NAME --benchmark libero-plus --smoke
```

去掉 `--smoke` 后执行正式测评；`--dry-run` 查看计划，不加载 GPU 模型。Plus 的准备步骤仍会生成独立部署配置和资产链接，原训练配置不修改。`OPENWAM_PYTHON` 可指定模型环境解释器；基准源码默认使用本仓库同级 `benchmarks/LIBERO` 或 `benchmarks/LIBERO-plus`，Plus 解释器默认使用模型环境同级的 `libero-plus` 环境，可用 `--libero-path` / `--libero-plus-python` 覆盖。

启动器读取 checkpoint 中的 `model.video_backbone.encoder.seperate_encode_view`。缺失或 false 时保留原来的 LANCZOS 主视角/腕部预缩放；true 时传输旋转后的原始相机图，由服务端分别直接 bilinear resize 到 224×224。history 图像走同一路径。原生 LIBERO / LIBERO-Plus WebSocket 客户端从服务端 PONG 自动读取此开关；旧服务端没有该字段时回退到原处理。直接使用 harness YAML 时，应在 benchmark params 中设置与 checkpoint 一致的 `seperate_encode_view`；通用启动器自动填写，不需要手动设置。

Plus 自动检测 raw V-JEPA / S-VAE，仅为 S-VAE 链接其 sidecar。根据训练 config 的 `text_encoder_type` / `vlm_text_embedding_method` 自动选择 `data/{umt5,cosmos3,qwen3p5}-embedding/libero-plus/` 下的缓存；VLM 再按提取方式分目录。Plus 默认 `correct_instruction_suffix=True`，使用 `corrected_text`；显式 `--correct-instruction-suffix False` 使用 `original_text`。标准 LIBERO 默认仍为 False。完整配置与预编码命令见 [text_conditioning.md](../../docs/text_conditioning.md)。`TEXT_EMBEDDING_DIR` / `PLUS_DEPLOY_DIR` 可覆盖缓存和部署目录。旧的 `prepare_libero_plus_svae` 模块和 S-VAE 快捷脚本仍然可用。正式启动前保留已有文本覆盖校验，缺失 prompt 不会回退到训练缓存。

## LIBERO 最小记录实测（2026-10-10）

默认最小记录同时适用于 LIBERO 和 LIBERO-Plus，与 checkpoint 的模型配置无关。GPU2 上使用新版 bottleneck final10690，对全部 40 个官方 LIBERO task 各运行前两个连续 trial，保留官方最大步数、初始状态、10 步同步推理和关闭 velocity cache：32 workers/batch8 的原记录方式两次端到端耗时为 398.3 秒和 364.8 秒，最小记录为 324.9 秒；实际控制步吞吐从 47.1 / 50.8 提升到 53.7 步/秒。40 workers/batch16 仅再增加约 0.45% 控制步吞吐，却多用约 15 GiB PSS，因此保留 32 workers/batch8 默认值。

这是 80-trial 子集实测，包含模型加载和结果写出，并非完整 2,000-trial 耗时保证。闭环轨迹有数值波动，不能要求各次控制步数或逐 trial 成败完全一致。证据位于测试 checkpoint 的 `libero_acceleration_20261010/comparison.json`。
