# RoboCasa365 基础设施交付记录（2026-11-17）

已完成官方环境、全量 Human300 准备、原始结构映射、模型无关数据/训练/评测接口与 Dummy smoke。没有训练任何实际模型，没有修改现有模型代码，没有数据增强、额外监督或推理 Tricks。

## 环境与路径

实际工作区是 `/data/fcy/code`，系统没有 `/code`。

| 项目 | 路径或版本 |
|---|---|
| RoboCasa | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa`，1.0.1，commit `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` |
| robosuite | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robosuite`，1.5.2，commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| 独立环境 | `/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa/.venv`，Python 3.11.18 |
| 核心依赖 | MuJoCo 3.3.1、NumPy 2.2.5、LeRobot 0.3.3、PyTorch 2.7.1+cpu |
| 默认渲染 | OSMesa / Mesa 23.2.1 llvmpipe，使用主机 `libOSMesa.so.8` |
| 新代码 | `/data/fcy/code/openwam-exp/benchmarks/robocasa365/infra` |
| 日志与结果 | `/data/fcy/code/openwam-exp/outputs/robocasa365_infra` |

`requirements.lock.txt` 固定 pip 依赖，`conda-explicit-linux-64.txt` 固定 Python 与 conda 包；pip/packaging 由 conda explicit 文件固定，避免 pip freeze 导出的构建机私有路径。锁文件 `pip install --dry-run --no-deps` 通过，`pip check` 无冲突。官方两个仓库 tracked diff 为空。`.pth` 显式加入源码根目录，解决从 code 父目录启动时 PEP660 editable 与同名 namespace 目录冲突的问题。私有宏按官方脚本创建。

调查时当前官方 main 与所选 RoboCasa 提交的 Gym wrapper、env_utils、dataset_registry 一致；版本 1.0.1 的官方 horizon 更新已包含，不额外乘 1.5。

## 全量数据结果

原始路径：

```text
/data/fcy/code/openwam-exp/.benchmark_runtime/robocasa365/robocasa/datasets/v1.0/pretrain/{atomic,composite}/<Task>/<date>/lerobot
```

| 检查项 | 实际结果 |
|---|---:|
| 官方 Human300 任务 | 300/300，65 Atomic + 235 Composite |
| 原始 Episode | 32,043，全部保留 |
| 原始状态/动作帧 | 29,106,226 |
| 官方 `100_demos` / seed=0 筛选后 | 30,000 Episode，27,815,381 帧 |
| 原始三视角视频 | 96,129 |
| 已记录 SHA-256 的原始文件 | 226,701 |
| 已下载原始 tar 字节总计 | 113,154,340,352（约 105.38 GiB） |
| 失败任务 | 0 |
| Target 演示目录 | 无 |

下载清单所有 300 项的 task/path/split/source/filter_key/URL，以及评测任务分组，均与安装后的官方 registry 完全一致。只使用 `pretrain_human300`，没有 `--all`、MimicGen、Target 演示或额外 subtask 标签。

检查覆盖：所有原始 Episode 的元数据计数、唯一 Episode ID、Parquet 行数、state16/action12 维度、有限值、帧索引、20 Hz 时间戳、语言索引、三视角文件与视频帧数；所有视频检查头信息并解码首帧；所有原始文件完整读取计算 SHA-256。六组官方环境资产下载后通过 ZIP CRC 检查，并记录 archive SHA-256。

原始包有超过 100 个 Episode 的任务，这是官方文件内容；保留原文件，通过与官方相同的 `100_demos` 算法选择训练视图，不写回修改数据。

证据：`human300_manifest.json`、`human300_official_manifest.json`、`manifest_verification.json`、`data_audit.json`、`data_summary.json`、`task_audits/`、`assets_manifest.json`。

## 统一接口

- `schema.py` / [SCHEMA.md](SCHEMA.md)：原始 state16、action12、三视角 RGB、原始语言；四元数 XYZW，不转 rotation6d；动作按官方 modality 切片映射到 Gym，由官方 wrapper 分发控制器。
- `dataset.py`：只读 LeRobot v2.1 + 官方视频 decoder，H 历史帧、K Action Chunk、batch/workers；边界重复并标注 mask，不跨 Episode；保留 float64 原始数据与时间戳，不做归一化或增强。
- `training.py`：`TrainConfig`、`TrainablePolicy.compute_loss`、`TrainingPipeline`、versioned checkpoint；当前 CLI 只有 no_grad dry run，Dummy 无可训练参数。未来模型显式提供 optimizer/update 才可训练。
- `evaluation.py`：注入 `Policy.reset/predict`；只提供 RGB/本体状态/语言/时间，不传成功信号或仿真内部状态给模型；chunk 顺序执行，无平滑、重试或其他策略组件。

正式评测固定 pretrain 厨房，50 个任务 × 每任务 50 次；三组为 Atomic-Seen 18、Composite-Seen 16、Composite-Unseen 16。Horizon 由官方 `get_task_horizon` 提供。Overall 是 50 个任务 SR 的平均值；三 split 简单平均另记为诊断指标，不冒充榜单得分。缩短测试均标记 `smoke_non_benchmark`，`leaderboard_score=null`。

## 实际验证

| 验证 | 结果 |
|---|---|
| 固定包版本、实际导入源码、tracked diff、官方任务分组检查 | 通过 |
| `pip check` / 可移植 lock dry-run | 通过 |
| 新增契约 + 真数据集成测试 | **10 passed**；仅官方 decoder 的 torchvision 弃用提示 |
| 现有 RoboCasa365 interface/config/debug 测试 | **20 passed** |
| 全量 Dataset 构造 | 300 任务，30,000 Episode，27,815,381 帧 |
| 全量首/中/末样本 RGB/state/action 解码 | AdjustToasterOvenTemperature / OrganizeCoffeeCondiments / YogurtDelightPrep，均通过 |
| Batch + Dummy forward/loss | H=3、K=4、batch=2、workers=2，2 batches，16 个有效 action steps |
| Checkpoint | 保存/恢复通过，step=0，trainable_parameters=0，updates=0 |
| 实际仿真 | 三类别各 1 Episode、4 steps，OSMesa 三视角渲染 + 原生动作执行 + 结果导出通过 |

实际 Batch：state `[2,3,16]`、action `[2,4,12]`、每视角 RGB `[2,3,256,256,3]`。首/中/末样本检查包括末帧 action mask `[true,false,false,false]`。真数据测试额外核对 Parquet 原值与实际官方控制器分发。

仿真任务为 CloseBlenderLid / DeliverStraw / ArrangeBreadBasket，官方 horizon 分别 900 / 2550 / 4350。本次只执行每任务 4 步，Dummy SR=0/3，**不是模型性能或正式榜单结果**。

证据：`tests_final.log`、`existing_tests.log`、`full_dataset_smoke.json`、`train_smoke_full/dry_run.json`、`train_smoke_full/dummy.pt`、`eval_smoke_final/{run.json,episodes.jsonl,episodes.csv,summary.json,*.png}`。`protocol_snapshot.json` 导出了全部 50 个任务、horizon 与 2,500 rollout 计划。

## 运行命令

从 `/data/fcy/code/openwam-exp` 执行，输出目录使用新名字：

```bash
RUN=benchmarks/robocasa365/infra/run.sh
MANIFEST=outputs/robocasa365_infra/human300_manifest.json

bash "$RUN" env

# 不提供 --tasks 时验证完整 300 任务 Dataset；不训练。
bash "$RUN" dry-run --manifest "$MANIFEST" \
  --history 3 --chunk 4 --batch-size 2 --workers 2 --batches 2 \
  --output outputs/rc365_data_check_new

# 官方三类别各一个任务的实际闭环与渲染测试。
bash "$RUN" eval --policy dummy --smoke --trials 1 --smoke-steps 4 \
  --history 3 --chunk 2 --output outputs/rc365_eval_check_new

# 复查全部原始数据（可选 --deep 解码全部视频帧，耗时显著增加）。
bash "$RUN" audit --manifest "$MANIFEST" --workers 8 \
  --output outputs/rc365_audit_new.json

# 将来接入模型：零参数工厂返回实现 Policy 协议的适配器。
# 默认完整 50×50，不缩短官方 horizon；当前未执行此性能评测。
bash "$RUN" eval --policy my_adapter:make_policy --output outputs/rc365_standard_new
```

全新安装/重新下载的命令见 [README.md](README.md)。

## 尚未解决与验证范围

1. 本机 NVIDIA H100 / driver 590.48.01 的 EGL 路径会产生黑帧。最小静态场景也可复现，GL error 为 0；具体驱动层根因未解决。OSMesa 连续渲染稳定并已完成实际厨房验证，当前默认使用它；没有修改官方渲染源码、相机或物理参数。EGL 故障目录/日志保留作诊断，不能当作已通过的渲染验证。
2. 官方未发布上游数据校验和；此处记录的是下载 archive 与本地原始文件的 SHA-256。视频已检查全量文件/头信息/帧数并逐视频解码首帧，**未穷举解码每个视频的所有帧**；可通过 `audit --deep` 补充。
3. 官方 Gym 把 state observation space 声明为 [-1,1]，实际世界位置超出时会发 warning。诊断定位 `state.base_position`，保留原始坐标，没有裁剪或归一化来消除警告。
4. 按本次范围，没有实际模型训练、正式 2,500 rollout 性能测试或榜单投稿。Checkpoint 当前明确为 CPU PyTorch 接口，未实现分布式/AMP/CUDA RNG 恢复。中断评测保留 JSONL，但不自动续跑；中断解包的无 receipt 目录需要人工检查后再恢复，默认拒绝覆盖。
