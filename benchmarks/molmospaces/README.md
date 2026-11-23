# MolmoSpaces DROID 基础设施

从自己的 checkpoint 开始评测，请看[逐步评测指南及本机 smoke 记录](CHECKPOINT_EVAL.md)。

在 `/data/fcy/code/openwam-exp` 执行下列命令。源码与依赖、渲染环境、下载缓存、资产和探测数据集中在 `.benchmark_runtime/molmospaces/`，由 Git 忽略。RoboCasa365 已搬到同级 `.benchmark_runtime/robocasa365/`。

以 [MolmoSpaces v0.2.9](https://github.com/allenai/molmospaces/tree/v0.2.9) 固定仿真和评测源码。对应提交为 `735138876887eb64650742631b8dda841dd264bf`。WebSocket 使用官方参考 [OpenPI 分支](https://github.com/omarrayyann/openpi/tree/711487f019e5f03b254d427d4523b1f0805a4814) 的协议实现。模型工厂、权重、训练配置里的 `training_factory` 留给后续接入。

## 仅准备评测环境

在项目根目录运行以下命令。仅评测模式安装 Classic、Filament、WebSocket 客户端和官方运行资产。可直接使用自己的策略服务，训练集、训练环境和参考策略权重均可省略。

```bash
bash benchmarks/molmospaces/setup.sh --eval-only
RT="$PWD/.benchmark_runtime/molmospaces"
"$RT/molmospaces/.venv-classic/bin/python" -m benchmarks.molmospaces.smoke_env \
  --output "outputs/molmo-env-smoke-$(date +%Y%m%d-%H%M%S)"
```

环境自检核对 11 个评测条目，使用官方 Dummy Policy 分别在 MS-Pick（Classic）和 MB-Pick-Filament 取第一个 episode，最多执行两步。输出包括 inventory.json 和两份 rollout 日志。正式评测使用下方 evaluate 入口及官方 horizon；Dummy Policy 的结果仅用于验证环境。

## 环境和资产

```bash
MOLMO_PIP_INDEX=https://mirrors.sustech.edu.cn/pypi/web/simple \
  bash benchmarks/molmospaces/setup.sh
```

脚本在独立的 Classic、Filament、DROID 环境安装依赖，校验官方 Filament wheel 的 Git blob，并记录版本清单。Classic 使用 MuJoCo 3.5.0 与同版 Warp，Filament 使用官方 MuJoCo 3.7.1 wheel 与 Warp 3.7.0。MolmoSpaces 的资源管理器决定组件版本和按需下载，资产目录仅供评测。由于上游 Decord wheel 内部标为 cp36，安装仅对这一条平台告警允许继续，并要求 Decord 在 Python 3.11 实际导入成功。

## DROID 数据与训练接口

训练数据来源固定为官方 `gs://gresearch/robotics/droid/1.0.1` 的 train split。完整 RLDS 为约 1.87 TB，放在指定根目录下的 `droid/1.0.1/`。MolmoSpaces/MolmoBot 同分布示范及其评测 rollout 均排除在训练输入之外。此来源边界遵循 [Not Using MolmoSpaces Data 榜单](https://molmospaces.allen.ai/leaderboard)。

`droid.py` 流式读取 RLDS，使用 OpenPI 发布的 keep ranges 选取成功轨迹中的 anchor，保留两个外部视角和腕部 RGB，选第一条非空原始指令。状态和动作均为 7 个 Franka 关节加 1 个 DROID gripper 值。动作直接使用 RLDS 的绝对关节位置目标，chunk 沿原轨迹，终点重复并提供 `action_valid`。训练统计只用所选训练帧。后续模型用 `TrainablePolicy.compute_loss(batch)` 接收归一化结果和 mask，用自身的 state_dict 接口管理权重。训练代码无需导入仿真器。

```bash
RT="$PWD/.benchmark_runtime/molmospaces"
ART="$RT/molmospaces-artifacts"
# 下载完整训练集前准备约 1.87 TB 空间。
mkdir -p "$ART/droid/droid/1.0.1"
gcloud storage cp --recursive 'gs://gresearch/robotics/droid/1.0.1/*' "$ART/droid/droid/1.0.1/"
"$ART/.venv-droid/bin/python" -m benchmarks.molmospaces.droid \
  --root "$ART/droid" --keep-ranges "$ART/droid_sample_ranges_v1_0_1.json" \
  --stats "$ART/train-statistics.json"
```

DROID 公开实现参考 [OpenPI](https://github.com/Physical-Intelligence/openpi)、[MolmoAct2-DROID](https://huggingface.co/allenai/MolmoAct2-DROID)、[Cosmos3 DROID](https://nvidia.github.io/cosmos-framework/action_policy_droid_posttrain/) 和 [PRTS-Droid](https://github.com/TeleHuman/PRTS)。本接口使用明确的 OpenPI keep-range 规则与独立的均值/标准差统计，不能与各模型 checkpoint 自带的统计互换。

## 后续接入与评测

给 `serve` 传入自己的 `module:factory`，工厂接收 checkpoint 路径并返回具有 `predict(observation)` 的策略。官方模拟端使用一台外部相机与一台腕部相机，并执行每次请求前 8 条动作，约 15 Hz。模型接入时须明确训练与部署的视角匹配，第二台 DROID 外部相机在这个客户端不会自动生成。服务端只做一次反归一化，采用官方客户端的关节顺序与 gripper 控制映射。

```bash
# 用模型自己的 Python 环境启动，安装 openpi-client 和 websockets>=13。
python -m benchmarks.molmospaces.provenance /path/to/checkpoint \
  --output outputs/my-model-checkpoint.json
python -m benchmarks.molmospaces.serve --factory my_module:load_model \
  --checkpoint /path/to/checkpoint --stats "$ART/train-statistics.json"
# 在另一终端启动评测。--smoke 只选第一个官方 Episode。
"$RT/molmospaces/.venv-classic/bin/python" -m benchmarks.molmospaces.evaluate \
  --policy-name my-model --checkpoint-manifest outputs/my-model-checkpoint.json \
  --suite MS-Pick --smoke --output outputs/molmo-smoke
# 完整评测使用新输出目录，省略 --smoke 和 --suite。
```

`protocol.json` 列出 [官方 MS](https://allenai.github.io/molmospaces/ms-bench/) 4 项和 [官方 MB](https://allenai.github.io/molmospaces/mb-bench/) 7 项，包括官方 RandCam 相机选择。沿用官方 Episode、默认 horizon 与任务 success，并记录其默认成功即结束设定。逐项目保存 rollout 日志与原始输出，调用官方 `eval_to_csv.py --success-condition both` 导出 Oracle SR 与 End SR，通过 Episode 计数门槛才汇总至 `metrics.jsonl`。Smoke 的 `provenance.json` 单独标记。

当前模型评测留空。安装过程与测试结果在 `.benchmark_runtime/molmospaces/molmospaces-artifacts/`。完整数据未下载，官方真实数据探测只取 shard 0 的文件开头，探测文件不能作为完整训练集。
