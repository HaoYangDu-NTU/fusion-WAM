# RoboCasa-GR1 自有 checkpoint 接入、评测与训练参考

已有权重可以复用 GR1 的 Unix socket 传输和 rollout，在模型自己的 Python 环境加载 checkpoint。模拟器接收 29 维绝对关节动作，模型架构、文本编码器和历史长度由自己的适配器决定。下方 Long-WAM 的初始化和 P384 训练配方仅适用于选择该配方的实验。

## 1. 在自己的模型包中提供加载函数

例如实现 `my_policy.gr1:load_policy(checkpoint_path, stats_path)`，返回下表中的策略对象。服务端沿用项目已有的 dispatcher；一次加载权重可服务多个 episode，历史按 `episode_id` 隔离。[传输契约](../../.benchmark_runtime/robocasa_gr1/LongLive/Long-WAM/src/longwam/benchmarks/robocasa_gr1/contract.py) · [dispatcher 和返回类型](../../.benchmark_runtime/robocasa_gr1/LongLive/Long-WAM/src/longwam/benchmarks/robocasa_gr1/policy.py)

| 方法 | 自己的适配器负责什么 |
|---|---|
| `reset(episode_id)`、`release(episode_id)` | 分别初始化与释放该回合的缓存。单帧模型可以实现为空操作。 |
| `observe(*, episode_id, image)` | 接收执行 chunk 期间的中间 ego RGB 帧，供需要 20 Hz 历史的模型缓存。 |
| `infer(*, episode_id, image, state, instruction)` | 当前帧为 uint8 HWC 256×256，状态为 29 维，顺序为左臂 7、右臂 7、左手 6、右手 6、腰 3。根据自己训练时的配置做预处理并推理。 |
| 返回 | `policy.PolicyInference(actions, input_image_sha256, policy_inference_seconds)`。动作需为有限的 CPU 数组 `[16,29]`，使用上述字段顺序，已还原为绝对关节控制量。用 `contract.array_sha256(image)` 计算原始输入指纹，延迟填实测正秒数。评测器一次执行 16 步。 |

如果模型输出 EEF33，需要先做配套的坐标转换和 IK。已有的 EEF33 客户端可作为实现参考，适配前先核对模型的末端姿态、夹爪和腰部定义。[EEF33 转换实现](openwam2robocasa_gr1_interface.py)

## 2. 启动自有策略服务

先完成下文的仿真资产自检。在同一 Bash 终端执行以下步骤，替换权重、统计量、工厂与模型解释器路径。模型环境须能导入自己的包、torch 和下列 Long-WAM 传输模块。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/robocasa_gr1"
export PYTHONPATH="$RT/LongLive/Long-WAM/src:/absolute/path/to/model-package-parent${PYTHONPATH:+:$PYTHONPATH}"
export MY_CKPT=/absolute/path/to/checkpoint MY_STATS=/absolute/path/to/training-stats.json
export FACTORY=my_policy.gr1:load_policy
MODEL_PY=/absolute/path/to/model-env/bin/python
SIM_PY="$RT/sim-env/bin/python"
WORK=$(mktemp -d /tmp/gr1-eval.XXXXXXXX)
export GR1_SOCKET="$WORK/policy.sock"
export GR1_OUT="$PWD/outputs/gr1-$(basename "$WORK")"
mkdir -p "$GR1_OUT"

"$MODEL_PY" -u - >"$GR1_OUT/server.log" 2>&1 <<'PYMODEL' &
import importlib
import os
from pathlib import Path
from longwam.benchmarks.robocasa_gr1.policy import PolicyRequestDispatcher, UnixPolicyServer

module, name = os.environ["FACTORY"].split(":")
load = getattr(importlib.import_module(module), name)
policy = load(os.environ["MY_CKPT"], os.environ["MY_STATS"])
UnixPolicyServer(Path(os.environ["GR1_SOCKET"]), PolicyRequestDispatcher(policy)).serve_forever()
PYMODEL
SERVER_PID=$!
```

## 3. 先验一条轨迹，再按任务全量评测

下面的薄入口直接调用已有的 `evaluate_task`，沿用官方环境和上游关节控制映射。将它写成文件是为了让仿真子进程能用 spawn 启动。第一个命令选一项任务、一个 episode、32 个控制步，足够跨两次推理；全量命令评当前 checkpoint 的 24 项任务，每项 100 局，每局最多 720 步。每次调用使用新的输出文件，以免覆盖记录。

```bash
cat > "$WORK/eval_gr1.py" <<'PYROLL'
import json
import os
import sys
from pathlib import Path
from longwam.benchmarks.robocasa_gr1 import rollout as r

def main(mode):
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    smoke = mode == "smoke"
    tasks = r.resolve_tasks([])[:1] if smoke else r.resolve_tasks([])
    output = Path(os.environ["GR1_OUT"])
    rows = []
    with (output / (mode + ".jsonl")).open("x") as log:
        with r.UnixPolicyClient(Path(os.environ["GR1_SOCKET"])) as client:
            for task in tasks:
                episodes = r.evaluate_task(task=task, client=client, env_factory=r.SpawnedGymEnv,
                                           episode_count=1 if smoke else 100, n_envs=1 if smoke else 5,
                                           max_control_steps=32 if smoke else 720)
                for row in episodes:
                    print(json.dumps(row), file=log, flush=True)
                rows.extend(episodes)
    assert len(rows) == len(tasks) * (1 if smoke else 100)
    summary = json.dumps(r.summarize(rows), indent=2)
    (output / (mode + "-summary.json")).write_text(summary + "\n")
    print(summary)

if __name__ == "__main__":
    main(sys.argv[1])
PYROLL
set -o pipefail
"$SIM_PY" "$WORK/eval_gr1.py" smoke | tee "$GR1_OUT/smoke.log"
# 确认策略请求、真实动作执行与逐局日志后再运行
"$SIM_PY" "$WORK/eval_gr1.py" full | tee "$GR1_OUT/full.log"
kill "$SERVER_PID"
```

每个待评 checkpoint 单独加载、运行并保留权重标识及结果。仿真资产已在真实抽屉任务中完成自检。接入自己的 checkpoint 后，还需用上面的 32 步 smoke 检查策略请求、动作方向和历史缓存。

**策略链路实测**：使用从本地测试 checkpoint 加载的保持关节位置策略，在 CupToDrawer 完成 32 个控制步。Unix socket 记录到一次 reset、第 0 和 16 步两次 infer、30 次中间帧 observe，以及最后一次 release。每个动作块为 16×29；rollout 耗时 5.35 秒，模拟器命令墙钟 30.77 秒。此测试验证了调度、传输和动作执行，自有模型成功率需替换权重后测量。[逐局记录](../../outputs/gr1-policy-smoke-0vNtxYhE/smoke.jsonl) · [策略事件](../../outputs/gr1-policy-smoke-0vNtxYhE/server.log) · [验收结果](../../outputs/gr1-policy-smoke-0vNtxYhE/verification.json)

## 可选训练参考：Long-WAM 四卡配方

以下命令在项目根目录执行。当前机器已具备数据，可以直接复核配置；在新机器准备数据时，使用 [prepare_teleop.sh](prepare_teleop.sh) 下载固定 revision 并重新校验。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/robocasa_gr1"
export PATH=/data/fcy/basics/miniconda3/envs/openwam/bin:$PATH
PYTHON="$(command -v python)" bash benchmarks/robocasa_gr1/longwam_4gpu.sh --dry-run
```

### 参考模型的数据与动作

| 项目 | 参考配置 |
|---|---|
| 训练数据 | NVIDIA GR00T Teleop-Sim，revision `09c6de8af50168090e7e9cc01e1ec3bce788de24`。24 项任务，每项 1,000 条示范，共 5,820,277 帧。数据在 `$RT/teleop/LeRobot`。 |
| 观测 | 20 Hz，单目第一视角 RGB，从 256×256 处理至 224×224。指令取自 episode remarks。44 维原始状态选取 29 维关节，再编码成 58 维 sin/cos。 |
| 历史与预测 | P384 覆盖 19.2 秒历史。每个训练窗口有 417 个原始时间点，采样 105 个视频时间点，预测 2 个未来 latent 和 16 步动作。 |
| 动作与部署 | 从原始 44 维动作中选 29 维绝对关节目标，用训练数据的全局 min-max 归一化，部署使用同一份统计量还原。该 checkpoint 配套 Long-WAM 关节策略客户端；项目已有的 OpenWAM EEF33 客户端使用另一套动作表示。 |
| 初始化 | 视频专家使用 LongLive Robot-S，动作专家使用对应的映射权重。运行还依赖 Wan2.2-TI2V-5B 资源和 GR1 文本缓存。 |

### 参考模型的优化、启动与恢复

AdamW 学习率为 3e-5，β 为 (0.95, 0.999)，ε 为 1e-8，权重衰减 1e-5，梯度裁剪 1.0。使用 bf16、seed 42，每卡 8 个数据 worker、预取深度 2。前 1,500 次更新 warmup，随后使用 cosine_hf。每 5,000 步保存模型与恢复状态。30k 步按全局 batch 480 计算，共抽取 1,440 万个训练窗口。[训练参数与初始化路径](../../.benchmark_runtime/robocasa_gr1/four_gpu_composed.yaml)

配置好 Long-WAM 训练环境、基础权重和文本缓存后，首次训练与中断恢复分别使用下面两条命令。续训时将路径替换成同一实验实际保存的状态目录；只指定权重文件会重新开始优化器计步。[上游训练与恢复说明](../../.benchmark_runtime/robocasa_gr1/LongLive/Long-WAM/docs/TRAINING.md)

```bash
# 首次训练
bash benchmarks/robocasa_gr1/longwam_4gpu.sh

# 中断恢复时单独执行
bash benchmarks/robocasa_gr1/longwam_4gpu.sh resume=/absolute/path/to/checkpoints/state/step_NNNNNN
```

## 仿真资产与评测口径

模拟器固定在 `4840e671596f93ca03651524b9f72ffb1aadfeff`，使用 MuJoCo 3.2.6 和 robosuite 1.5.1。资产根目录为 `$ROBOCASA_GR1_PATH/robocasa/models/assets`。这里的运行时资产用于 reset、渲染和物理交互，训练示范另存于 `$RT/teleop/LeRobot`。

| 资产根目录下的路径 | 来源及内容 |
|---|---|
| `scenes/tabletop_layouts/`、`scenes/kitchen_layouts/`、`scenes/kitchen_styles/`、`arenas/` | 随固定模拟器源码提供。6 份 tabletop 布局含普通与杂乱桌面、cotrain、柜体、抽屉和微波炉布局；场景与 arena 共 33 个文件已与源码包按字节比对。[SHA-256 清单](../../outputs/gr1-assets-0lLVMxoq/scene-sha256.json) |
| `fixtures/`、`textures/`、`generative_textures/` | 固定模拟器指定的三个 Box 包，分别有 782、219、450 个有效载荷文件。 |
| `objects/objaverse/`、`objects/aigen_objs/` | 同版模拟器指定的两个 Box 物体库，分别有 33,199 和 58,733 个文件。 |
| `objects/sketchfab/`、`objects/lightwheel/` | NVIDIA PhysicalAI-DigitalCousin-Assets 固定 revision `1b018839a6da865dffecd3185fe054211bc71270`，分别有 1,998 和 7,327 个文件。 |

七组下载资产共 102,708 个有效文件，解压后 22.02 GB（十进制，不计归档目录与 macOS 元数据）。本机按 GR1 官方 ZIP 的文件名、未压缩字节数和 CRC32 核对；缓存中匹配的内容可复用，其余从对应的固定源取得。各包 URL、归档大小及中央目录哈希见[官方包清单](../../outputs/gr1-assets-0lLVMxoq/official-inventory.json)，落盘校验见[逐包验收记录](../../outputs/gr1-assets-0lLVMxoq/verified-assets.json)。

在项目根目录设置路径。只有全新安装或重建资产时才需要执行注释中的官方下载命令；日常检查直接运行环境 smoke。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/robocasa_gr1"
export ROBOCASA_GR1_PATH="$RT/robocasa-gr1-tabletop-tasks-4840e671596f93ca03651524b9f72ffb1aadfeff"
export ROBOCASA_GR1_PYTHON="$RT/sim-env/bin/python"
# 新机器：在固定模拟器源码及仿真依赖安装完毕后运行
# (cd "$ROBOCASA_GR1_PATH" && "$ROBOCASA_GR1_PYTHON" robocasa/scripts/download_tabletop_assets.py -y)
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash benchmarks/robocasa_gr1/run_smoke.sh env
```

本机使用 OSMesa 跑通了上述环境入口。此前因缺少 Objaverse 而中止的 PnPCupToDrawerClose 已完成 reset、渲染和一步环境动作，返回第一视角观测与完整的 29 维动作字段。[默认入口日志](../../outputs/gr1-assets-0lLVMxoq/default-smoke.log)

2026-11-20 实测 **24/24 个任务通过**。每项使用 seed 0，reset 后读取 29 维关节状态，以当前状态作为绝对关节目标执行一步；前后两次都取得有效 RGB。已独立解码全部 24 张截图并核对像素统计。四进程并发时，单项从建环境到保存结果的耗时范围为 8.85–15.57 秒，中位数 11.34 秒。该检查验证资产和仿真交互，策略成功率需用自己的 checkpoint 另测。[逐任务结果](../../outputs/gr1-assets-0lLVMxoq/smoke-summary.json) · [抽屉任务截图](../../outputs/gr1-assets-0lLVMxoq/CupToDrawer.png)

如需重复 24 项任务的资产验收，可将本次使用的检查脚本复制到新输出目录。它从固定任务表取 Gym ID，每个任务保存一步交互后的第一视角 PNG 和观测统计。下面按顺序运行，便于定位失败的场景或物体。

```bash
CHECK=$(mktemp -d "$PWD/outputs/gr1-asset-recheck-XXXXXXXX")
cp outputs/gr1-assets-0lLVMxoq/check_env.py "$CHECK/"
(
  set -e
  export PYTHONPATH="$RT/LongLive/Long-WAM/src:$ROBOCASA_GR1_PATH${PYTHONPATH:+:$PYTHONPATH}"
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
  for task_index in $(seq 0 23); do
    "$ROBOCASA_GR1_PYTHON" "$CHECK/check_env.py" "$task_index" >"$CHECK/task-$task_index.log" 2>&1
  done
)
```

Long-WAM 默认每项 10 局，最多 720 个控制步。若采用本项目收录的 **VLA Evaluation Harness GR1** 口径，评最近 5 个 checkpoint，每项 100 局。按上述保存周期取 10k、15k、20k、25k、30k，共 12,000 局。每个 checkpoint 单独计算 24 项任务成功率的算术平均，选择平均值最高的一个，保留其分任务成绩及全部原始结果。这是该 Harness 的口径，独立的 GR1 官方榜单提交规则尚未核实。[Harness 规则](../../third_party/vla-evaluation-harness/leaderboard/benchmarks/robocasa.md)

```bash
# 仅针对 Long-WAM 原生 checkpoint bundle；其他架构使用本文开头的策略工厂
BUNDLE=/absolute/path/to/checkpoint_bundle
TEXT_CACHE=/absolute/path/to/gr1/text_embeds
RESULT=/absolute/path/to/eval/step_030000
(cd "$RT/LongLive/Long-WAM" && bash scripts/robocasa_gr1/eval.sh \
  expected_history=384 episodes=100 max_control_steps=720 \
  checkpoint="$BUNDLE/model.pt" model_config="$BUNDLE/config.yaml" \
  stats="$RT/global_action_minmax.json" text_cache="$TEXT_CACHE" \
  simulator_python="$ROBOCASA_GR1_PYTHON" output_dir="$RESULT")
```

## 验证记录与实现位置

真实 Parquet、第一视角 RGB 解码、关节变换和 CPU 线性头跑了 3 次更新，18 项协议测试通过。完整 Long-WAM 训练的最近一次离线验证在加载 Wan2.2-TI2V-5B snapshot 时中断。仿真资产与该训练权重独立，环境验收见上节。完整模型训练速度和策略成功率仍待实测。[Smoke 记录](../../outputs/benchmark-smoke-qKKuSId4/README.md)

| 环节 | 文件 |
|---|---|
| 数据验收与动作统计 | [校验脚本](verify_teleop.py)、[29 维 min-max](../../.benchmark_runtime/robocasa_gr1/global_action_minmax.json) |
| 训练与环境 | [四卡入口](longwam_4gpu.sh)、[环境自检](run_smoke.sh) |
| 评测与动作接口 | [Long-WAM 评测参数](../../.benchmark_runtime/robocasa_gr1/LongLive/Long-WAM/configs/eval/robocasa_gr1.yaml)、[OpenWAM EEF33 接口](openwam2robocasa_gr1_interface.py) |
