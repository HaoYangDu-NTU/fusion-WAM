# 从自己的 checkpoint 评测 MolmoSpaces

在 `/data/fcy/code/openwam-exp` 运行。模型进程负责加载权重和预测动作，评测进程负责相机、物理仿真、任务结束判定及官方 CSV。只做评测时需要官方场景和 episode 资产，无需下载 DROID 示范或 π0.5 权重。选择 Not Using MolmoSpaces Data 组别还需核实模型的训练数据来源，benchmark rollout 只用于评测。[官方榜单](https://molmospaces.allen.ai/leaderboard) · [固定评测源码 v0.2.9](https://github.com/allenai/molmospaces/tree/v0.2.9)

## 1. 核对 checkpoint 与控制接口

| 项目 | 当前服务交给模型或要求模型返回的内容 |
|---|---|
| RGB | `sample["images"]` 含 `exterior_image_1_left` 和 `wrist_image_left`，每张为 224×224×3 HWC RGB。官方客户端按比例缩放并补边。RandCam 使用该 suite 指定的随机外部相机。 |
| 语言 | `sample["language"]` 为本 episode 的官方指令，已转小写。 |
| 本体状态 | `sample["state"]` 为 8 维 float32，顺序是 Franka 的 7 个关节角和夹爪。官方客户端先把夹爪位置除以 0.824033 并裁至 [0,1]，服务端再按训练统计标准化。 |
| 动作 | `predict(sample)` 返回有限的 float32 数组 `[K,8]`。反归一化后前 7 维是按官方顺序的绝对关节角目标（弧度），最后一维是 DROID 夹爪目标，默认由官方客户端按 0.5 阈值执行二值控制。 |
| 执行 | 约 15 Hz，每次请求执行前 8 步。模型若返回 16 步，就设 `--chunk 16`，第 9 个控制步会用新观测重新推理。 |
| 历史 | 当前请求只包含当下两路图像。需要四帧历史、第三视角或不同动作空间的 checkpoint，须先明确对应的观测和控制适配；服务端缓存最近四次请求取到的是间隔 8 步的观测。 |

先看权重对应的训练配置和动作头。特别是 LIBERO / RoboCasa 的末端位姿增量与 Franka 绝对关节目标语义不同，即使补齐到 8 维也不能直接用于这个接口。

在自己的模型包里提供一个可导入的工厂，例如 `my_policy.molmo:load_policy`。工厂接收一个 checkpoint 路径字符串，加载权重、设置推理模式，返回暴露 `predict(sample)` 的对象。模型特有的 tokenizer、图像处理、设备放置、推理和结果转为 CPU NumPy 均由工厂或该对象处理。服务端实现见 [serve.py](serve.py)。

本接口的统计文件包含 `schema="droid.absolute-joint-position.v1"`，以及 `state.mean`、`state.std`、`actions.mean`、`actions.std` 四个长度为 8 的数组。使用此 checkpoint 训练时匹配的统计量，标准差必须大于零。若模型使用分位数、delta 动作或已经在内部反归一化，先核对或适配服务端的转换，避免重复缩放。

## 2. 准备环境并记录权重

在同一个 Bash 终端依次执行以下步骤。替换下方四个与模型相关的变量。模型适配模块须已安装在模型环境或可从 PYTHONPATH 导入。

```bash
cd /data/fcy/code/openwam-exp
RT="$PWD/.benchmark_runtime/molmospaces"
EVAL_PY="$RT/molmospaces/.venv-classic/bin/python"
# 首次部署或新机器运行
test -x "$EVAL_PY" || bash benchmarks/molmospaces/setup.sh --eval-only

CKPT=/path/to/my/checkpoint
STATS=/path/to/my/training-stats.json
FACTORY=my_policy.molmo:load_policy
MODEL_PY=/path/to/my/model-env/bin/python

RUN="$PWD/outputs/molmo-my-model-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
"$EVAL_PY" -m benchmarks.molmospaces.provenance "$CKPT" --output "$RUN/checkpoint.json"
sha256sum "$STATS" > "$RUN/stats.sha256"
```

## 3. 启动本地策略服务

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

## 4. 先测一条官方轨迹，再跑完整套件

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

每次使用新的输出目录。`--smoke` 选第 0 个官方 episode，保留官方 horizon 和成功判定。对已验证默认驱动稳定的机器，可省略所有 suite 过滤，以一次调用覆盖 11 项。本机软件 Vulkan 已验证短程闭环，硬件 Vulkan 及全套稳定性尚待验证。各条目的 benchmark 路径、renderer 和相机选择见 [protocol.json](protocol.json)。

## 5. 读取结果

每个套件的 `official.csv` 由官方导出器生成。`success_rate_pct` 是结束时的 End SR，`oracle_rate_pct` 是轨迹中曾达到成功条件的 Oracle SR。`OVERALL` 是该套件总体，其他行按官方类别分组。`rollout.log` 记录运行过程，`rollouts/` 保留原始轨迹，`command.json` 与 `packages.txt` 记录仿真命令及依赖。入口在核对 CSV 的总 episode 数后才向顶层 `metrics.jsonl` 写入该套件的结果。确认全部 11 项均完成，再使用正式输出整理榜单材料。

## 本机 smoke 与复现

可先用保存在项目输出目录的保持关节位置策略复现上述命令。在第 2 步设置完 RT 和 EVAL_PY 后，将四个模型变量替换为下列值，再从创建 RUN 开始执行。该策略从磁盘加载 8 个系数，实际读取两张图的形状、语言和状态，返回 8 步动作。测试专用的单位统计量只与这个策略配套。

```bash
DEMO="$PWD/outputs/molmo-ckpt-smoke-wLsvLB9m"
export PYTHONPATH="$DEMO:$PWD${PYTHONPATH:+:$PYTHONPATH}"
CKPT="$DEMO/hold.npy"
STATS="$DEMO/stats.json"
FACTORY=hold_policy:load
MODEL_PY="$EVAL_PY"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

本机使用 v0.2.9 评测器认可的 Objaverse 资产版本 20260131。启动时上游会提示该版本可能与生成原始 benchmark 时使用的物体资产存在差异，具体版本可在 rollout 日志中核对。

| 验证项 | 结果 | 证据 |
|---|---|---|
| MS-Pick，Classic + OSMesa，episode 0 | 官方 horizon 455 步，57 次策略请求，命令墙钟 91.82 秒。1 条轨迹完成，Oracle / End 均为 0%。 | [官方 CSV](../../outputs/molmo-ckpt-smoke-wLsvLB9m/cpu/MS-Pick/official.csv) · [rollout 日志](../../outputs/molmo-ckpt-smoke-wLsvLB9m/cpu/MS-Pick/rollout.log) |
| MB-Pick-RandCam，Filament + 软件 Vulkan，episode 0 | 显式截到 9 步，2 次策略请求，耗时记录 57.35 秒。Oracle / End 均为 0%，已生成 HDF5、相机视频和 CSV。 | [官方导出器 CSV](../../outputs/molmo-ckpt-smoke-wLsvLB9m/randcam.csv) · [日志](../../outputs/molmo-ckpt-smoke-wLsvLB9m/randcam.log) |
| MS-Pick，默认 EGL | 两次在客户端已连接后 SIGABRT，分别收到 44、47 次策略请求；第二次栈顶在上游 OpenGL 的 mjr_readPixels。 | [错误栈](../../outputs/molmo-ckpt-smoke-wLsvLB9m/retry/MS-Pick/rollout.log) |

以上是测试策略的链路耗时和成绩。自有模型的推理时延、成功率及全量评测耗时需在接入实际权重后测量。
