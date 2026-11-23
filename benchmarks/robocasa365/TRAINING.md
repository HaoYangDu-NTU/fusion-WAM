# RoboCasa365 自有 checkpoint 接入、训练与评测

已有策略可以直接通过 `--policy 包名:工厂名` 接入。评测器负责重置环境、收集三视角观测、执行动作并统计成功率。你需要提供**权重、匹配的模型定义与预处理配置，以及训练时使用的归一化统计**。若模型运行在独立环境中，工厂也可以返回连接现有推理服务的客户端。

## 1. 实现策略接口

在自己的 Python 包里实现一个**无参数**工厂，例如 `my_robot.robocasa365:make_policy`。评测器每个任务调用一次工厂，每个 episode 调用一次 `reset()`，每次重规划调用 `predict(obs)`。下例的 `load_policy` 是由你实现的模型加载函数；环境变量是本示例向无参数工厂传路径的方式。

```python
import os
from .model import load_policy

def make_policy():
    return load_policy(
        checkpoint=os.environ["MY_CKPT"],
        stats=os.environ["MY_STATS"],
    )
```

返回对象需要实现以下约定。模型内部的分辨率、表示转换、归一化和设备放置由适配器负责。[字段与顺序](infra/SCHEMA.md) · [rollout 实现](infra/evaluation.py)

| 接口 | 数据或行为 |
|---|---|
| `reset()` | 清除上一个 episode 的历史、动作缓存和循环网络状态。 |
| `obs["rgb"]` | `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand` 三个键，每个为 uint8 数组 `[H,256,256,3]`，按时间从旧到新排列。 |
| 状态和语言 | `state[H,16]` 按底盘位置、底盘四元数、相对末端位置、相对末端四元数、夹爪双关节位置排列，四元数顺序 XYZW。`language` 是任务指令，`timestamp` 和 `history_valid` 对应各历史帧。 |
| 历史 | `--history H` 提供最近 H 个连续控制步；回合开始用首帧补齐。如需偏移 −6、−4、−2、0，可传 H=7，再在适配器内取索引 0、2、4、6。 |
| `predict(obs)` | 返回有限的 CPU NumPy 数组 `[K,12]`，K≥1，范围为 [−1,1]。字段顺序为底盘及躯干 4、控制模式 1、末端位置增量 3、末端旋转增量 3、夹爪 1。模型负责恢复到该原生控制表示，官方控制器负责后续缩放。 |
| 重规划 | 评测器执行返回的整个 chunk，遇成功、终止或 horizon 提前结束。若模型预测 16 步、部署只执行 8 步，在适配器返回前 8 行。自有策略的 K 由返回数组决定。 |

## 2. 用自己的权重跑 smoke，再评 Target50

以下命令使用本项目的仿真环境。将四个路径或模块值替换为自己的设置；确保工厂及其依赖能在该解释器中导入。远程策略可把模型依赖放在服务端。

```bash
cd /data/fcy/code/openwam-exp
export MY_CKPT=/absolute/path/to/checkpoint
export MY_STATS=/absolute/path/to/training-stats.json
export PYTHONPATH="/absolute/path/to/model-package-parent${PYTHONPATH:+:$PYTHONPATH}"
FACTORY=my_robot.robocasa365:make_policy
RUN=benchmarks/robocasa365/infra/run.sh
OUT="$PWD/outputs/robocasa365-my-policy-$(date +%Y%m%d-%H%M%S)"
H=1  # 按训练时的历史需求修改

bash "$RUN" eval --policy "$FACTORY" --history "$H" --smoke --trials 1 \
  --smoke-steps 32 --tasks CloseBlenderLid --output "$OUT/smoke"
cat "$OUT/smoke/summary.json"

# 确认截图、动作方向、policy_calls 和 action_chunk_lengths 后启动正式评测
bash "$RUN" eval --policy "$FACTORY" --history "$H" --trials 50 --output "$OUT/target50"
```

RoboCasa 1.0.1，pretrain 厨房，沿用各任务官方 horizon 和成功判定。Atomic-Seen 18 项、Composite-Seen 16 项、Composite-Unseen 16 项，各 50 局，共 2,500 局。`episodes.jsonl` 记录逐局结果，`summary.json` 给出各任务、三个 split 和总成绩。为便于复现，随成绩保存实际权重哈希、适配器版本、模型配置、统计量及执行长度。

## 3. 需要训练时复用 Human300

Standard 训练数据使用 `pretrain_human300` 的 300 项任务、30,000 条示范。Target 数据只用于评测。自有训练循环可复用 `Human300Dataset` 和 `make_loader`，选择历史长度、chunk、batch 与 worker 数，并用 `action_valid` 屏蔽 episode 边界外的动作。优化器、冻结策略、初始化、保存与恢复由自己的模型决定。[Dataset](infra/dataset.py) · [训练与 checkpoint 接口](infra/training.py)

```bash
bash "$RUN" dry-run --manifest outputs/robocasa365_infra/human300_manifest.json \
  --history 4 --chunk 16 --batch-size 2 --batches 1 --output "$OUT/data-smoke"
```

这条命令使用 Dummy Policy 检查 batch 和 checkpoint 读写。已有实测覆盖三个 split 各一个任务、每项 8 步，耗时 121.24 秒；自有模型的时延和成功率需另测。[实测记录](../../outputs/benchmark-smoke-qKKuSId4/README.md)
