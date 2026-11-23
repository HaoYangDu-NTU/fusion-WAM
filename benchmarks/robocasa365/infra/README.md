# RoboCasa365 原始数据与模型无关基础设施

这里独立于已有 OpenWAM 模型代码，不修改模型、原始数据或原评测客户端。
所有命令从 `/data/fcy/code/openwam-exp` 执行；`run.sh` 会定位独立 Python。

## 官方依据与边界

- [Standard Leaderboard](https://robocasa.ai/leaderboard)：训练仅允许未改动的 `pretrain_human300`，不用 Target 演示、MimicGen、额外 subtask 标签。
- [多任务评测](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html)：`atomic_seen` / `composite_seen` / `composite_unseen`，**厨房 split 固定为 `pretrain`**。
- [评测说明](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html)：每任务 50 scenarios，horizon 取 v1.0.1 registry（已含 1.5× 更新）。
- [原始 LeRobot 格式](https://robocasa.ai/docs/build/html/datasets/using_datasets.html)。

50 个评测任务分组为 18/16/16。Leaderboard Overall 是 50 个任务成功率的均值，即按 18/16/16 加权的分组均值；`overall_split_mean` 仅是另行导出的诊断指标。标准模式禁止缩短 horizon、改厨房 split、减少任务/rollout；所有缩短运行明确标成 `smoke_non_benchmark`，`leaderboard_score=null`。

## 目录与环境

| 内容 | 实际路径 |
|---|---|
| 官方 RoboCasa | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa` |
| 官方 robosuite | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robosuite` |
| 隔离 Python 3.11 | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa/.venv/bin/python` |
| 原始数据 | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa/datasets/v1.0/pretrain/{atomic,composite}/<Task>/<date>/lerobot` |
| manifest / 审计 / logs | `/data/fcy/code/openwam-exp/outputs/robocasa365_infra` |zif

固定 RoboCasa `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` (1.0.1)、robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (1.5.2)、MuJoCo 3.3.1、NumPy 2.2.5、LeRobot 0.3.3。版本沿用项目已验证提交，调查时与官方 main 的 env_utils/Gym/registry 无差异。官方文档要求版本 1.0.1，未发布一个必须采用的唯一 git hash。`env` 检查包版本、实际导入路径、git commit、tracked diff；不匹配即失败。

PyTorch 使用 CPU wheel 2.7.1，负责接口与 Dummy smoke；不会占用现有模型训练环境。本机默认使用官方支持的 OSMesa CPU 渲染；NVIDIA 590.48.01/H100 的 EGL 路径在最小静态场景也出现不稳定黑帧，未修改官方源码来绕过。主机已有 `libOSMesa.so.8`；新主机需要提供该系统库（Ubuntu 包 `libosmesa6`）。如需自行验证 EGL，可显式设置 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0`，后端会记录进结果。后续大型模型可在独立模型环境运行并实现此处协议。

```bash
# 重建环境；锁文件生成后优先安装全量固定依赖。
CONDA_BIN=/data/haoyang/miniconda3/bin/conda \
  bash benchmarks/robocasa365/infra/setup.sh
# 新机器还需下载官方厨房资产（不属于训练 Target 数据）：
DOWNLOAD_ASSETS=1 CONDA_BIN=/path/to/conda \
  bash benchmarks/robocasa365/infra/setup.sh
bash benchmarks/robocasa365/infra/run.sh env
```

## 数据准备与检查

```bash
RUN=benchmarks/robocasa365/infra/run.sh
OUT=outputs/robocasa365_infra
bash "$RUN" manifest --output "$OUT/human300_manifest.json"
bash "$RUN" download --manifest "$OUT/human300_manifest.json" --workers 8
bash "$RUN" audit --manifest "$OUT/human300_manifest.json" \
  --output "$OUT/data_audit.json" --workers 4
# 可选逐帧完整解码（比默认检查慢得多）：
bash "$RUN" audit --manifest "$OUT/human300_manifest.json" \
  --output "$OUT/data_audit_deep.json" --workers 4 --deep
```

清单严格对照官方 `DATASET_SOUP_REGISTRY['pretrain_human300']`，包括路径、来源、filter 和官方下载 URL。不会调用 `--all`，不会下载 Target 演示。每包先下载到 `.part`，安全检查解包到 staging，完成后 rename；顶层官方 README 原样保留。`download.complete.json` 记录来源 URL、压缩包长度、SHA256。下载成功不等于审计成功：审计单独检查全部任务的 meta、每条 parquet 的长度/字段/finite/timestamps/episode IDs、三视角视频数量/尺寸/帧数及首帧解码，并计算**所有原文件 SHA256**。`--deep` 解码全部视频帧。官方没有提供独立 checksum 清单；这里的 SHA256 是本地下载与后续复核依据，不能声称已做官方 checksum 对比。

原始包的 Episode 数可能多于 100。磁盘保留全部，Dataset 精确复用官方 `get_subset_demos_filter_key` 的 seed=0 shuffle 和 `100_demos` 逻辑。未篡改、转换、重编码或删除 Episode。`extras` 原样保存但不进入模型接口。

## Observation / Action Schema

版本 `robocasa365.raw.v1`；切片逐项校验官方 `meta/modality.json`。不归一化、不 resize、不改变语言、不使用现有模型的 15D/19D 表示。

| 原始 state16 切片 | 含义 |
|---|---|
| 0:3 / 3:7 | 世界坐标 base position / quaternion xyzw |
| 7:10 / 10:14 | base 坐标系 EEF position / quaternion xyzw |
| 14:16 | 双指 gripper qpos |

| 原始 action12 切片 | Gym 字段 |
|---|---|
| 0:4 | `action.base_motion`：base vx/vy/yaw + torso |
| 4:5 | `action.control_mode` |
| 5:8 | `action.end_effector_position`：归一化 OSC delta |
| 8:11 | `action.end_effector_rotation`：归一化 axis-angle delta |
| 11:12 | `action.gripper_close` |

原始控制量是 controller 的归一化输入，不是 SI 单位下的绝对 pose。传给 Gym 后由官方 `PandaOmronKeyConverter` 做 gripper/control_mode 的 0.5 阈值转换；本层不再缩放、裁剪或改变旋转表示。非法维度、NaN/Inf、越界动作直接报错。

`Human300Dataset(manifest, history=H, chunk=K, tasks=None)` 返回：

```text
observation.rgb[camera] : uint8 [H,256,256,3] RGB，left/right/eye_in_hand
observation.state       : [H,16] 原始浮点值
observation.language    : 原始 annotation.human.task_description 对应的文本
observation.timestamp   : [H] 原始 Episode 内时间戳
observation.history_valid : bool [H]
action                  : [K,12] 原始浮点值
action_valid            : bool [K]
action_timestamp        : [K] 原始时间戳
episode_id / task / frame_index / episode_length / is_first / is_last
next_done               : 原始 next.done [K]
```

历史不足或 chunk 超过 Episode 末尾时重复边界，并用 mask 标出；不会跨 Episode。`is_last` 是物理 Episode 边界，与原始 `next.done` 分开。数据值保留原始精度（通常 float64）；dtype 转换由将来模型显式决定。视频调用官方 LeRobot decoder，解码结果从其 float CHW 无损还原为 uint8 HWC；不重新编码。DataLoader 默认平坦按帧索引、无任务加权，支持 batch/workers/shuffle。

```python
from benchmarks.robocasa365.infra.dataset import Human300Dataset, make_loader
train = Human300Dataset('outputs/robocasa365_infra/human300_manifest.json',
                        history=3, chunk=16)
loader = make_loader(train, batch_size=8, workers=4)
batch = next(iter(loader))
```

完整 Dataset 默认需要全部 300 个任务；`tasks=['OpenDrawer']` 只用于明确的局部 smoke。不会把缺失任务悄悄略过。

## 模型无关训练与 Checkpoint

`TrainConfig` 定义 H/K/batch/workers/schema，`config.json` 是无 Tricks 配置示例。未来模型实现 `compute_loss(batch) -> {'loss': scalar_tensor, ...}` 与 state_dict 接口；必须用 `action_valid` 排除 padding。`TrainingPipeline.step(..., update=False)` 仅 eval/no_grad，只有调用方显式传入 optimizer 并设置 `update=True` 才会 backward/step；当前 CLI 不提供训练开关。

Checkpoint 保存 schema、配置、step、model、可选 optimizer、Python/NumPy/Torch CPU RNG，atomic rename；加载拒绝 schema 不匹配。当前为 CPU smoke 框架，不宣称支持分布式/AMP/CUDA RNG 恢复。仅加载受信任的本地 PyTorch checkpoint。

```bash
bash benchmarks/robocasa365/infra/run.sh dry-run \
  --manifest outputs/robocasa365_infra/human300_manifest.json \
  --tasks OpenDrawer --history 3 --chunk 4 --batch-size 2 --batches 2 \
  --output outputs/robocasa365_infra/train_smoke
```

Dummy 没有可训练参数；此命令验证真数据、Batch、forward/loss、mask、checkpoint roundtrip，不做梯度更新。

## 闭环评测

Policy 实现 `reset()` 和 `predict(observation) -> ndarray[K,12]`，每 Episode reset。模型只能接收上述 RGB/state/language/time/history，不能收到成功信号、MuJoCo state 或任务内部信息。输出 chunk 逐步执行，每步更新历史；没有平滑、重试、集成或额外控制器。官方 success 信号只供 harness 统计与终止。

```bash
# 每个类别自动选首个任务，1 trial、4 steps，生成三张实际渲染图：
bash benchmarks/robocasa365/infra/run.sh eval --policy dummy \
  --smoke --trials 1 --smoke-steps 4 --history 3 --chunk 2 \
  --output outputs/robocasa365_infra/eval_smoke

# 将来标准完整评测：50 tasks × 50 rollouts，原始官方 horizon。
# factory 是零参数函数，返回实现 Policy 协议的模型适配器。
bash benchmarks/robocasa365/infra/run.sh eval --policy my_adapter:make_policy \
  --output outputs/my_model_standard
```

每次使用新 output 目录，避免混合结果。输出 `run.json`（版本/模式）、`episodes.jsonl`（实时逐条持久化）、`episodes.csv`、`summary.json`（每任务、每 split、50 任务平均成功率）。标准结果只有完整成功执行后才导出 `leaderboard_score`；不自动向榜单投稿。Scenario seed 显式记录，构造与 reset 都使用官方 RNG 接口，不读取训练演示作为评测初始状态。

## 验证结果

实际运行状态见同目录 `STATUS.md`；原始证据位于 `outputs/robocasa365_infra`。

```bash
ROBOCASA365_MANIFEST=outputs/robocasa365_infra/human300_manifest.json \
OMP_NUM_THREADS=1 /data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa/.venv/bin/python -m pytest \
  benchmarks/robocasa365/infra -q
```

当前实际输出可对照 `environment.json`、`protocol_snapshot.json`、`schema.json`、
`manifest_verification.json`、`data_audit.json`、`train_smoke/dry_run.json`、
`eval_smoke_final/summary.json`。旧的 `eval_smoke` 与
`eval_smoke_egl_render_rejected` 是保留的 EGL 故障诊断，不是通过的渲染验证。

官方 Gym 对 state 的空间声明统一使用 [-1,1]，实际世界坐标可能超出，因而
`passive_env_checker` 会提示 observation space 警告。诊断已确认是
`state.base_position`（如 [4.896,-3.310,0.700]）；接口保留真实坐标。
Torchvision 的解码弃用提示来自固定版本 LeRobot 使用的官方 decoder，
实际解码通过。MimicGen、GR1 IK 等可选机器人扩展未安装；PandaOmron
benchmark 不依赖这些扩展。
