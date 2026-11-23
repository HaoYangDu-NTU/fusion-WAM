# RoboLab-120 自有 DROID checkpoint 评测

**本机可以读取 DROID 训练数据、检查评测资产、运行策略通信 smoke。真实相机 rollout 需要 RTX 节点。** 在四卡 H100 机器上实跑官方 BananaInBowlTask 时，Isaac Sim 在渲染器启动阶段报告 Vulkan `ERROR_DEVICE_LOST`，进程退出码 139，未得到可计分 episode。模型推理和模拟器可部署在不同机器，H100 可继续用于模型服务。[本次 Isaac 日志](../../outputs/robolab-smoke-y0ekTMIh/isaac-verified.log) · [Isaac Sim 5.1 硬件要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

已有 DROID 训练权重时，直接从[接入自己的 checkpoint](#接入自己的-checkpoint)开始。使用训练时保存的预处理配置和归一化统计启动策略服务，在 RTX 节点先运行单任务 smoke，通过后再运行完整评测。评测无需读取 DROID 训练集。

## 配置与目录

| 项目 | 本次选择 |
|---|---|
| Benchmark | NVlabs/RoboLab v0.3.3，commit `dc3383888b7279b4256312268a35a5f9046c44d3`。120 个任务，simple / moderate / complex 分别 64 / 39 / 17。 |
| 模拟器 | Python 3.11、Isaac Sim 5.1.0.0、Isaac Lab 2.3.2.post1。本机使用上游支持的 `isaac51` 组合，复用共享 Isaac Sim 安装。上游默认组合为 Isaac Sim 5.0.0 / Isaac Lab 2.2.0，对比成绩时应披露版本。 |
| 项目布局 | 官方源码、物体、场景和机器人在 `.benchmark_runtime/robolab/RoboLab/`，本机 Python 在 `.benchmark_runtime/robolab/sim-env/`，入口和校验脚本在 [benchmarks/robolab](.)。 |
| 数据来源 | 本次选 DROID 真机数据作为后训练来源，复用共享盘 LeRobot v3 转换。RoboLab-120 的任务、录制回放和 smoke 输出用于评测。若模型使用过其他仿真训练数据，按实际情况披露榜单训练类别。 |

官方介绍和仓库中可查评测设计、支持的安装组合及训练来源分类。[项目](https://research.nvidia.com/labs/srl/projects/robolab/) · [固定版本仓库](https://github.com/NVlabs/RoboLab/tree/dc3383888b7279b4256312268a35a5f9046c44d3) · [榜单](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html) · [环境依赖](../../outputs/robolab-smoke-y0ekTMIh/sim-packages.txt) · [数据与锁文件哈希](../../outputs/robolab-smoke-y0ekTMIh/provenance.json)

本机共享环境的 `pip check` 仍报告 9 项版本约束冲突，包括 Isaac Sim 声明的 filelock、fsspec、networkx 和 websockets 版本；其中 RoboLab 上游对 websockets 有显式覆盖。其余冲突来自共享工具依赖。迁移到独立 RTX 环境时使用下方官方冻结安装命令。[依赖检查原文](../../outputs/robolab-smoke-y0ekTMIh/pip-check.txt)

## 训练数据与读取验证

从项目根目录使用 `.benchmark_runtime/robolab/data/DROID`。它指向共享数据 `/data/rhodes_team/EmbodiedDataFoundation/PRTS_WAM/lerobot/DROID`。该转换的 train split 为 **74,344 条 episode、17,909,675 帧、15 Hz**。其产物包括数据 Parquet、episode 索引、477 个视频文件，以及字段与统计元数据。使用关节位置策略时，`observation.state` 与 `action` 均为 7 维关节 + 1 维夹爪；另有具名的末端、速度及关节字段，选择时应对照训练配置。[字段清单](../../.benchmark_runtime/robolab/data/DROID/meta/info.json) · [转换契约](../../.benchmark_runtime/robolab/data/DROID/meta/contract.json) · [DROID 官方来源](https://droid-dataset.github.io/)

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/robolab"
DATA_PY=/data/fcy/basics/miniconda3/envs/openwam/bin/python
"$DATA_PY" benchmarks/robolab/smoke_droid.py "$RT/data/DROID"
```

实测校验了全部 episode 的边界和总帧数，并读出 episode 0 的 16 个连续控制步。三路 RGB（腕部、外部一、外部二）各为 `[16,180,320,3]`，状态及动作各为 `[16,8]`，语言为 “Put the blue block in the green bowl”。跨 episode 抽窗口前应尊重索引边界，模型归一化应与其训练统计一致。媒体清点覆盖全库，逐帧解码覆盖上述窗口。[实测结果](../../outputs/robolab-smoke-y0ekTMIh/droid-smoke.json)

## 环境与 smoke

已获取固定 commit 的全部 1,646 个 LFS 文件路径（1,127 个唯一对象），通过 `git lfs fsck HEAD`。离线解析覆盖 120 个任务引用的 47 个不同场景 USD 和 DROID 机器人 USD。[LFS 完整性](../../outputs/robolab-smoke-y0ekTMIh/lfs-fsck.log) · [资产清点](../../outputs/robolab-smoke-y0ekTMIh/lfs-summary.json) · [场景解析结果](../../outputs/robolab-smoke-y0ekTMIh/asset-smoke.json)

在本机或挂载同一环境的 RTX 节点，可用以下命令复核。第一条解析离线 USD，第二条通过官方 OpenPI 客户端在本机回环地址实际交换图像、状态和动作。第三条是官方仿真验收，要求真实 reset 并执行五步。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/robolab"
"$RT/tools/venv/bin/python" benchmarks/robolab/check_assets.py "$RT/RoboLab"
bash benchmarks/robolab/run.sh "$PWD/benchmarks/robolab/smoke_policy.py"

# 在 RTX 节点运行
bash benchmarks/robolab/run.sh -m pytest -q -s tests/test_run_empty.py --task BananaInBowlTask
```

本机 `sim-env` 的基础解释器依赖共享的 `/data/junsong/envs/sonic-isaaclab-2.3.2`。对于无法挂载该目录的 RTX 主机，安装系统 ffmpeg 和 Git LFS，使用相同上游版本创建独立环境。下例假定 RTX 主机已能访问项目源码及其资产；新 checkout 需要执行 `git lfs pull`。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/robolab"
(cd "$RT/RoboLab" && uv sync --python 3.11 --frozen --extra isaac51 --extra test)
export ROBOLAB_PYTHON="$RT/RoboLab/.venv/bin/python"
# 安装策略传输客户端。本项目另一个 benchmark 已保存该客户端源码。
uv pip install --python "$ROBOLAB_PYTHON" \
  .benchmark_runtime/molmospaces/openpi-molmo-jointpos/packages/openpi-client
bash benchmarks/robolab/run.sh -m pytest -q -s tests/test_run_empty.py --task BananaInBowlTask
```

## 接入自己的 checkpoint

用模型自己的环境启动 OpenPI WebSocket 兼容服务，加载 checkpoint 和训练时的图像处理器、状态及动作统计。官方关节客户端每次推理发送两张补边至 224×224 的 RGB（左肩外部相机和腕部）、7 维关节状态、夹爪和 `prompt`。响应键为 `actions`，值为 `[K,8]`，前 7 维为绝对关节目标，最后 1 维由客户端按 0.5 阈值转为二值夹爪。模型服务负责反归一化。对需要逐控制步历史、不同相机或其他动作空间的权重，需按训练语义编写客户端适配。[官方客户端源码](../../.benchmark_runtime/robolab/RoboLab/policies/pi0_family/client.py)

如果 checkpoint 使用逐维 mean/std 归一化，可以复用项目中的通用 DROID 服务。实现可导入的 `load_policy(checkpoint_path)` 返回策略对象，其 `predict(sample)` 接收 `images`、归一化的 `state` 和 `language`，输出归一化的 `[K,8]`。统计文件格式为 `droid.absolute-joint-position.v1`；分位数归一化或模型内部反归一化的策略需要按自身定义适配。[服务接口](../molmospaces/serve.py)

```bash
# 模型节点，从项目根目录执行；替换成自己的路径、加载函数和推理步数。
MODEL_PY=/absolute/path/to/model-env/bin/python
CKPT=/absolute/path/to/checkpoint
STATS=/absolute/path/to/training-stats.json
"$MODEL_PY" -m benchmarks.molmospaces.serve --factory my_robot.policy:load_policy \
  --checkpoint "$CKPT" --stats "$STATS" --chunk 8 --port 8000
```

这个通用服务默认只监听 `127.0.0.1`。两端跨节点时，在 RTX 节点用 `ssh -N -L 8000:127.0.0.1:8000 <模型节点>` 转发到自己的 H100 服务，再将模拟端的 `POLICY_HOST` 设为 `127.0.0.1`。使用已有可达的兼容服务时可直接指定其地址。

下例假定自己的模型服务已启动，返回至少 8 步，部署选择执行 8 步后重规划。`--policy pi05` 选择上游客户端预设；实际权重由服务端加载。保存结果时记录真实模型名、权重哈希、训练来源和 chunk 参数。

```bash
cd /data/fcy/code/openwam-exp
POLICY_HOST=127.0.0.1  # 若推理服务在 H100 节点，填写 RTX 节点可达的服务地址
POLICY_PORT=8000
bash benchmarks/robolab/run.sh policies/pi0_family/run.py --headless --policy pi05 \
  --remote-host "$POLICY_HOST" --remote-port "$POLICY_PORT" --open-loop-horizon 8 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 --output-folder-name my-model-smoke
```

待单任务运行成功后，去掉 `--task` 即可覆盖默认评测任务。每项 episode 数为 `--num-envs × --num-runs`。保留官方任务时限、默认布局、相机和成功条件，记录指令变体、每项样本数及版本。结果在官方源码的 `output/<实验名>/`，含 `episode_results.jsonl`、任务子目录、HDF5 与视频；成功率和子任务分数应分别报告。[结果格式](../../.benchmark_runtime/robolab/RoboLab/docs/data.md)

## 本机实测边界

策略通信使用合成观测和保持关节位置的测试服务。16 个控制调用触发两次真实 WebSocket 请求，客户端 reset 后的新调用再次请求；检查了图像大小、关节输出、夹爪二值化和 chunk 边界。该测试独立于物理仿真。本机未产生真实相机闭环成绩、评测吞吐或自有模型成功率。[通信日志](../../outputs/robolab-smoke-y0ekTMIh/policy-smoke.log) · [仿真退出码](../../outputs/robolab-smoke-y0ekTMIh/isaac-verified.exit)
