# MolmoSpaces 自有模型评测

当前目标是将自己的 checkpoint 接入 **MolmoSpaces v0.2.9**，按 **Not Using MolmoSpaces Data** 组别评测 MS 的 4 项和 MB 的 7 项。评测只需仿真资产和策略权重，无需下载 DROID 训练集或 π0.5 权重。该组别要求核对模型训练数据来源，MolmoSpaces / MolmoBot 同分布示范和官方 benchmark rollout 均排除在训练集之外。当前实测覆盖Classic 和 Filament 各一个完整官方 episode，自有模型接入后再测正式成绩。[基础设施说明](README.md) · [逐步接入与 smoke 复现](CHECKPOINT_EVAL.md)

## 配置、观测与动作

| 项目 | 当前评测接口 |
|---|---|
| 环境 | 运行目录为 `.benchmark_runtime/molmospaces/`。Classic 使用 MuJoCo 3.5.0，Filament 使用对应的 3.7.1 wheel；每项 renderer、episode 路径和相机由协议文件选择。 |
| 观测 | 单帧外部 RGB + 腕部 RGB，各为 224×224×3，按比例缩放并补边。RandCam 使用套件指定的随机外部相机。语言输入为转小写的官方任务指令。 |
| 状态与统计 | 按顺序输入 Franka 7 个关节角和夹爪，共 8 维。使用 checkpoint 配套的训练统计，schema 为 `droid.absolute-joint-position.v1`；`state` 和 `actions` 均须含 8 维 mean、std。 |
| 模型接口 | 工厂接收 checkpoint 路径并返回实现 `predict(sample)` 的策略。`sample` 含 `images`、`state`、`language`。模型返回有限的归一化动作数组 `[K,8]`，服务端做一次反归一化，得到 7 个绝对关节角目标（弧度）和 1 个夹爪目标。 |
| 执行 | 控制频率约 15 Hz，每次请求执行前 8 个动作再读取新观测。若模型预测 16 步，设置 `K=16`，执行长度仍为 8。夹爪由官方客户端按默认 0.5 阈值执行。 |

**先核对自己 checkpoint 的相机、历史与动作语义。** 如果权重需要三视角、控制步偏移 −6/−4/−2/0 的历史，或输出末端位姿增量，就需要额外的观测或动作适配。当前客户端只发送每次重规划时的两张图；缓存四次请求会得到间隔 8 步的图像。单纯改变输出维度也无法把末端位姿增量转换成绝对关节角目标。

## 实现自己的加载入口

在你的模型包中提供可导入的函数，例如 `my_policy.molmo:load_policy`。服务端调用 **`load_policy(checkpoint_path)`** 一次；函数应恢复模型和训练时的图像处理器，将模型切到推理模式，并返回实现 `predict(sample)` 的对象。路径既可指向单文件，也可指向由你的加载器识别的目录。

一次推理读取 `sample["images"]["exterior_image_1_left"]`、`sample["images"]["wrist_image_left"]`、`sample["state"]` 和 `sample["language"]`，返回 CPU NumPy 数组。服务端已经标准化状态，随后会反归一化动作；若模型自带这两项处理，应在适配器中对齐转换。返回的行数必须等于启动服务时的 `--chunk`。当前服务接口按无状态请求设计；依赖逐控制步观测或 episode 重置信号的模型需要扩展客户端协议。

## 配置入口与 checkpoint 记录

在同一个 Bash 终端依次执行以下步骤。替换下方四个与模型相关的变量。模型适配模块须已安装在模型环境或可从 PYTHONPATH 导入。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/molmospaces"
EVAL_PY="$RT/molmospaces/.venv-classic/bin/python"
# 首次部署或新机器运行
if [[ ! -x "$EVAL_PY" || ! -x "$RT/molmospaces/.venv-filament/bin/python" ]]; then
    bash benchmarks/molmospaces/setup.sh --eval-only
fi

CKPT=/path/to/my/checkpoint
STATS=/path/to/my/training-stats.json
FACTORY=my_policy.molmo:load_policy
MODEL_PY=/path/to/my/model-env/bin/python

RUN="$PWD/outputs/molmo-my-model-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
"$EVAL_PY" -m benchmarks.molmospaces.provenance "$CKPT" --output "$RUN/checkpoint.json"
sha256sum "$STATS" > "$RUN/stats.sha256"
```

## 启动策略服务

模型环境须能导入 openpi_client 和 websockets。缺依赖时可在该环境安装 `"$MODEL_PY" -m pip install -e "$RT/openpi-molmo-jointpos/packages/openpi-client" 'websockets>=13'`。预留一个未被使用的端口。模型实际返回的步数必须与 K 一致。

```bash
K=8
PORT=8080
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
"$MODEL_PY" -u -m benchmarks.molmospaces.serve \
  --factory "$FACTORY" --checkpoint "$CKPT" --stats "$STATS" \
  --chunk "$K" --port "$PORT" > "$RUN/server.log" 2>&1 &
SERVER_PID=$!
```

权重加载完成后执行 `curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/healthz"`。如果进程退出，先看 `$RUN/server.log`。健康检查仅验证服务监听，下一步才会发送真实图像和动作。

## 单条轨迹与全量评测

```bash
EVAL=("$EVAL_PY" -m benchmarks.molmospaces.evaluate
      --policy-name my-model --port "$PORT" --workers 1
      --checkpoint-manifest "$RUN/checkpoint.json")

# 仅让仿真进程使用本机验证过的软件 OpenGL；模型服务仍可用 GPU
CLASSIC=(env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=8)
"${CLASSIC[@]}" "${EVAL[@]}" --suite MS-Pick --smoke --output "$RUN/smoke"
cat "$RUN/smoke/metrics.jsonl"
cat "$RUN/smoke/MS-Pick/official.csv"
```

单套件全量评测可沿用上一条命令，去掉 `--smoke` 并更换输出目录。本机两种渲染器使用不同的驱动设置，跑全榜时可按下列两条命令分别启动，模型服务保持在线。每个套件仍使用其官方 renderer 和 episode。

```bash
"${CLASSIC[@]}" "${EVAL[@]}" --suite MS-Open --suite MS-Close --suite MS-Pick \
    --suite MS-PnP --suite MB-Pick-v1.5 --suite MB-Pick-Classic --output "$RUN/full-classic"

ICD=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
env VK_ICD_FILENAMES="$ICD" VK_DRIVER_FILES="$ICD" LP_NUM_THREADS=4 \
    "${EVAL[@]}" --suite MB-Pick-Filament --suite MB-Pick-RandCam --suite MB-PnP \
    --suite MB-PnP-NextTo --suite MB-PnP-Color --output "$RUN/full-filament"

# 所有评测结束后关闭本次启动的服务
kill "$SERVER_PID"
```

每次使用新的输出目录。`--smoke` 选第 0 个官方 episode，保留官方 horizon 和成功判定。对已验证默认驱动稳定的机器，可省略所有 suite 过滤，以一次调用覆盖 11 项。本机软件 Vulkan 已验证 RandCam 单条完整闭环，硬件 Vulkan 及全套稳定性尚待验证。各条目的 benchmark 路径、renderer 和相机选择见 [protocol.json](protocol.json)。

## 指标与结果

每个套件的 `official.csv` 由官方导出器生成。`success_rate_pct` 是结束时的 End SR，`oracle_rate_pct` 是轨迹中曾达到成功条件的 Oracle SR。`OVERALL` 是该套件总体，其他行按官方类别分组。`rollout.log` 记录运行过程，`rollouts/` 保留原始轨迹，`command.json` 与 `packages.txt` 记录仿真命令及依赖。入口在核对 CSV 的总 episode 数后才向顶层 `metrics.jsonl` 写入该套件的结果。确认全部 11 项均完成，再使用正式输出整理榜单材料。

## 本机验证与耗时

| 套件 | 官方 horizon | 策略请求 | 评测入口墙钟耗时 | 官方结果 |
|---|---:|---:|---:|---|
| MS-Pick，Classic + OSMesa | 455 | 57 | 124 秒 | [CSV](../../outputs/molmo-recheck-KF5tHVgu/MS-Pick/MS-Pick/official.csv) |
| MB-Pick-RandCam，Filament + 软件 Vulkan | 303 | 38 | 208 秒 | [CSV](../../outputs/molmo-recheck-KF5tHVgu/MB-Pick-RandCam/MB-Pick-RandCam/official.csv) |

两套件并行运行，每套件 1 个 worker，均使用从磁盘加载的保持关节位置策略，Oracle SR 和 End SR 均为 0%。耗时包括启动、仿真和导出。已检查 HDF5 轨迹、相机视频、官方 CSV 和 metrics.jsonl。11 个套件的 episode 列表与各自元数据计数一致，其余 9 个套件本轮只检查清单。[套件清点](../../outputs/molmo-recheck-KF5tHVgu/suites.json)

本机两个仿真环境原先依赖容器内的临时 Python 路径。本次将 CPython 3.11 放到项目运行目录，并调整安装脚本，让新部署也在项目内保存解释器。修复了 checkpoint 哈希在 Python 3.10 下的兼容性，4 项接口测试通过。[测试记录](../../outputs/molmo-recheck-KF5tHVgu/contract-recheck.log)

官方 v0.2.9 评测器接受本机 Objaverse 版本 `20260131`，并提示其资产可能与原始版本 `20251016_from_20250610` 有差异，正式结果应记录所用版本。Classic 默认 EGL 此前发生过中途退出，复测使用 OSMesa；硬件 Vulkan 及全榜稳定性仍待验证。[评测器版本提示](../../outputs/molmo-recheck-KF5tHVgu/MB-Pick-RandCam.log) · [此前 EGL 错误栈](../../outputs/molmo-ckpt-smoke-wLsvLB9m/retry/MS-Pick/rollout.log)

## 实现位置

| 环节 | 文件 |
|---|---|
| 版本、资产与套件 | [安装入口](setup.sh)、[11 项套件协议](protocol.json) |
| 模型接入与控制 | [策略服务](serve.py)、[状态和动作归一化](droid.py)、[相机与策略客户端](sim_config.py) |
| 评测与追溯 | [权重哈希清单](provenance.py)、[rollout、CSV 和 episode 数量核对](evaluate.py) |
