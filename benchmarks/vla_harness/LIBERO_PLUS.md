# OpenWAM LIBERO-Plus on harness

使用 OpenWAM 官方 `benchmarks/libero-plus/` 的任务、环境、观测、动作和评分规则；外层沿用本目录已验证的 harness 调度、动态批量推理、session 动作缓存、无损 raw 传输、固定模型线程及本机 SQLite WAL。标准 LIBERO 默认入口保持不变。

## 2026-09-29 等价优化验证更新

已检查10030任务目录与七类扰动，目录/协议保持一致。harness仅对glass41–50与zoom21–30启用通过逐位对照的局部加速；原逐step噪声、双视角、seed/初始化、30步settle、600/700步上限和成功定义均保留。Numba在既有libero-plus环境中使用，未升级依赖。

Plus benchmark在成功完成step后上报watchdog进度，避免健康长episode被当成停滞；卡住的step不会被定时线程掩盖。离线文本Wan/V-JEPA部署使用strict checkpoint恢复的empty shell减少CPU初始化，在线tokenizer路径仍可用。

七类13个真实环境的固定动作轨迹、噪声52项测试、1.3B/5B batch1和8的完整预测及动作逐位一致。四卡18个选定任务的完整episode已完成，0基础设施错误。这是局部工程验证，不能代替10030任务完整成功率；历史章节中的未验证描述应按其日期理解。详细前后实现、测速及覆盖限制见 [OPTIMIZATION_LOG.md](../../OPTIMIZATION_LOG.md) 最前面的2026-09-29条目和 `profile_output/plus_reliability_perf_20260929/REPORT.md`。

## 运行条件与验证范围

模型服务与 harness worker 使用 `/workspace/basics/miniconda3/envs/openwam/bin/python`。每个 Plus 任务的仿真子进程使用官方 README 配好的 `/workspace/basics/miniconda3/envs/libero-plus/bin/python`，源码为 `/workspace/code/benchmarks/LIBERO-plus`。不向这两个环境安装或替换包。

初版实现时四卡在训练，当时没有加载 GPU 模型、启动 EGL 测评或计算 T5 embedding；2026-09-22 的双模式文本编码见下文。代码对照及 CPU/OSMesa 仿真验证记录在 `profile_output/vla_harness_libero_plus_20260920/`。CPU 对照使用相同脚本动作检查输入与仿真行为；不将它当作真实模型闭环成功率，也不声称 Plus 已达到某个加速比。真实模型的 batch=1 对照、动态 batch 闭环及完整 10,030 次成绩仍需 GPU 空闲后验证。

已通过 45 项自动测试（18 项 Plus + 27 项原 LIBERO 回归）。10 个真实代表任务覆盖四个 suite、七类扰动，官方与适配器在 30 步 settle 后连续三步的观测、状态、准确语言、预处理图像和动作逐项一致。另以 CPU 脚本模型和两个真实 Plus 环境跑通实际 launcher、worker、WebSocket、session 缓冲、SQLite、merge 与 Plus summary；20 步短程各产生一条有效记录，未产生正式成功率。

## V-JEPA 2.1 + S-VAE：沿用已完成的 LIBERO 部署方式

2026-09-22 的 S-VAE LIBERO harness 测评使用 `outputs/libero_vjepa21_cached_text_svae/2026-09-21_10-23-40/checkpoint_step_10690.safetensors`，成绩为 **1920/2000（96.00%）**。该成绩仅属于标准 LIBERO。Plus 使用同一 checkpoint、相同模型服务和 batch engine；S-VAE 仍通过 `video_encoder.batch_encode → _apply_svae → encode_mean` 执行，1408→48，不另写推理分支。

新增 `prepare_libero_plus_svae.py` 将这次成功运行的部署准备步骤固定下来：

- 读取 checkpoint 头部，核对 S-VAE encoder、输入统计、feature_norm 和 DiT 输入通道；不加载权重或分配 GPU。
- 在独立 config 中补上缺失的 `attr: vae / source: state_dict / VJEPA21VideoEncoder` 结构标记，和成功的 LIBERO 部署一致；不改原训练 config。
- 链接本模型的 checkpoint、`manifest.json`、**`svae_config.json`** 和 `normalization_stats.npy`。S-VAE 权重已包含在主 checkpoint 中，不需要部署时额外加载训练用的 `svae.pt`。
- config 的 `text_embedding_dir` 与部署目录的 `text_embedding` 链接都指向 Plus 缓存；缓存尚未生成时明确报错，不回退到旧 LIBERO 缓存。重复准备同一部署可安全复用；配置或链接不同则拒绝覆盖。

默认独立部署目录为：

`/workspace/code/OpenWAM/outputs/libero-plus/vjepa21_svae_step10690_harness_original_text/checkpoint_for_eval`

启用修正时使用独立的 `vjepa21_svae_step10690_harness_corrected_text/checkpoint_for_eval`，避免两种运行互相改写缓存链接。

不使用 GPU 的准备与计划检查：

```bash
cd /workspace/code/OpenWAM
CUDA_VISIBLE_DEVICES='' bash scripts/eval_libero_plus_vjepa21_svae.sh --dry-run
```

**下文提供双模式 T5 缓存生成命令；正式测评仍需单独启动：**

```bash
cd /workspace/code/OpenWAM
# 短程联调，不生成正式成绩
bash scripts/eval_libero_plus_vjepa21_svae.sh --smoke
# 全量 10030 个任务，各 trial0 一次
bash scripts/eval_libero_plus_vjepa21_svae.sh
```

启动脚本默认同一已验证的 S-VAE step10690，四个模型、32 workers、batch8、等待10ms、GPU3 EGL、raw 传输；支持追加原 launcher 参数。换另一份训练产物时，通过 `SOURCE_DIR`、`CKPT_NAME`、`DEPLOY_DIR` 指定对应来源与新目录，`TEXT_EMBEDDING_DIR` 可指定已有的准确 Plus 缓存。原生官方 scheduler 也可使用准备好的 `--ckpt-dir` 和 `--ckpt-name`，不需要改变环境协议。

本段为 S-VAE 部署初次验证记录，当时没有启动 GPU 编码或 Plus 正式测评，因此没有 S-VAE Plus 成功率和速度数字。准备目录不代表文本缓存已生成。**61 项 CPU 测试通过**；在 meta device 上按实际新部署配置构建完整架构，与最终 checkpoint 的 **2229 个 tensor 名称及形状全部匹配**，确认 S-VAE latent=48、CUDA 未初始化。该结构检查仅模拟文本缓存的文件读取，不加载文本 tensor 或模型权重，不代表文本缓存已就绪。记录位于 `profile_output/libero_plus_svae_20260922/`，原模型和 benchmark 核心逻辑未修改。

## 与标准 LIBERO 不同的官方规则

| 项目 | LIBERO-Plus 官方规则 |
|---|---|
| 任务数 | Spatial 2402 / Object 2518 / Goal 2591 / Long 2519，共 10030 |
| 每个任务 | 仅 trial 0，1 次；新建环境和进程 |
| 随机数 | `single_eval._control_seed(10000)` 后首次导入 benchmark；不调用 `env.seed` |
| 初始状态 | 原生 `get_task_init_states(task_id)`，选择第 0 项 |
| settle | 30 步 `[0,0,0,0,0,0,-1]` |
| 步数上限 | Spatial/Object/Goal 600，Long 700 |
| 观测 | 官方双视角、rotate_180、L-shape 插槽缩放、achieved EEF10 |
| 语言 | 默认 `correct_instruction_suffix: true`，仅移除非语言扰动的文件名后缀；显式 false 时精确使用 `task.language` |
| 动作/成功 | 原生 EEF10→7D 连续命令；`env.step` 返回 done 才成功 |
| 汇总 | 官方 category/difficulty 元数据；总成功数 / 总 trials，非七类别的简单平均 |

Plus 的 `libero/libero/benchmark/__init__.py:105–111` 在导入时执行 `random.shuffle`，会消耗全局 Python RNG。官方每个任务是新进程；在常驻进程中只重新设 seed 会跳过后续任务的首次导入消耗。因此每个 harness worker 内部为每个任务创建一个官方环境子进程，在子进程中保留 seed→导入→建环境的顺序。父进程和模型服务不能消耗该仿真进程的 RNG。Plus 的 seed=10000 不覆盖模型侧原生 deploy 的生成设置。

## 可选指令后缀修正与两组离线文本

`correct_instruction_suffix` 默认 **True**。显式设为 **False** 时保留上游 `task.language`。使用 **True** 时，`benchmarks/libero_plus_instructions.py::task_instruction` 只删除非 `_language_` 任务的 `view/initstate/noise/table/tb/light/add/level/moved` 元数据后缀。保持 filename 中原任务措辞，不改读可能措辞不同的 BDDL。任务 ID、环境扰动、RNG、图像、动作、成功判定均不变。

已核对完整 10030 个任务：8493 个非语言扰动任务修正后严格对应标准 LIBERO 的 40 条指令；1537 个语言扰动任务文本逐字不变。两组分别有 10002 / 1549 条唯一文本。共享文本只编码一次，独立目录内的同名 tensor 文件用硬链接复用。

| 参数 | 缓存目录（相对于 OpenWAM） | 唯一文本数 |
|---|---|---:|
| False | `data/libero-plus-embedding/original_text` | 10002 |
| True（默认） | `data/libero-plus-embedding/corrected_text` | 1549 |

每组有 `tasks.json`（10030 条任务到 prompt 映射）、`manifest.json`（精确 prompt 到文件映射）、BF16 `[512,4096]` context 与 int64 `seq_len`。训练用的 `data/text_embedding/manifest.json` 和原 40 个 tensor 不变。旧 Plus 缓存路径 `data/text_embedding/libero_plus` 作为指向 `original_text` 的兼容软链，供之前的默认部署继续使用。

S-VAE 启动脚本会根据参数自动选择独立部署目录和匹配缓存：

```bash
cd /workspace/code/OpenWAM
# 默认保留上游行为
bash scripts/eval_libero_plus_vjepa21_svae.sh --dry-run
# 仅修正指令后缀；去掉 --dry-run 才会真正启动测评
bash scripts/eval_libero_plus_vjepa21_svae.sh --correct_instruction_suffix True --dry-run
```

也支持 `--correct-instruction-suffix True` 和环境变量 `CORRECT_INSTRUCTION_SUFFIX=True`。通用 harness `run_eval` 支持同名参数，但需要 `--ckpt-dir` 已配置相应缓存；启动前会验证其完整覆盖当前模式的所有 prompt。官方 `single_eval.py` 支持同名 CLI 参数和 `policy_config.yml` 的 `correct_instruction_suffix: false/true`；官方 scheduler 可通过 policy config 传入。同一个纯函数用于官方入口、harness 子进程、任务文本导出和离线编码，避免各路径清洗不一致。

四卡一次编码两组（仅文本 encoder，不加载 policy）：

```bash
cd /workspace/code/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /workspace/basics/miniconda3/envs/openwam/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/precompute_libero_plus_text.py \
  --text-encoder /workspace/model/wan2_2_openwam_native_from_diffusers/models_t5_umt5-xxl-enc-bf16.safetensors \
  --tokenizer /workspace/model/wan2_2_openwam_native_from_diffusers/google/umt5-xxl \
  --output-dir /workspace/code/OpenWAM/data/libero-plus-embedding \
  --batch-size 8 --sleep-seconds 0
```

复用训练时同一冻结 Wan T5、tokenizer 和 `encode_text`（512 token padding、尾部归零），按唯一文本分片。每 GPU 进程默认限制 PyTorch 分配器最多使用该卡总显存的 25%；这不是性能隔离，四卡编码会与训练争用算力。重跑会检查并跳过已经写完的 tensor；只有所有分片完成后才发布两组 manifest。若更换 encoder/tokenizer 权重，需要另选新输出目录，不应复用旧 tensor。

`libero_plus_env --describe /path/tasks.json --correct_instruction_suffix True` 可以仅在 CPU 上导出修正后的文本。修正模式是默认值，原始模式需显式传入 False。

上游问题：[issue #64](https://github.com/sylvestf/LIBERO-plus/issues/64)、[issue #65](https://github.com/sylvestf/LIBERO-plus/issues/65)。本实现选择“只移除后缀”，与 [PR #71](https://github.com/sylvestf/LIBERO-plus/pull/71) 改用 BDDL 文本的方案有意区别。两种模式的测评结果必须标明参数，不混为同一测评协议。此开关不修复上游语言改写文本本身的其他质量问题。

## 2026-09-22 双模式缓存完成记录

两组四卡编码均已完成；启动到 manifest 发布 447.12 秒，包含权重加载的编码阶段 327.21 秒。每卡 tensor 峰值 12.30 GiB。10042 个不同 tensor 文件逐一通过 dtype、形状、有限值、长度及 padding 校验；两组实际预检查均 missing=0。修正组的 40 条基础任务 embedding 与原 LIBERO 逐 prompt 缓存全部 bitwise 相同。GPU 编码进程已退出，原训练继续运行。验证证据：profile_output/libero_plus_instruction_20260922/ 下的 cache_validation.json、live_preflight.json、encoding_metrics.json。这些是文本缓存验证，不是 Plus 闭环成功率验证。

## 四卡启动

只查看计划，不启动模型或仿真环境：

```bash
cd /workspace/code/OpenWAM
CUDA_VISIBLE_DEVICES='' /workspace/basics/miniconda3/envs/openwam/bin/python \
  -m benchmarks.vla_harness.run_eval --benchmark libero-plus --dry-run
```

Plus 文本缓存生成且 GPU 空闲后，先运行短程联调；`--smoke` 仅限制两个 task IDs、每任务 20 步，不出正式成功率：

```bash
cd /workspace/code/OpenWAM
/workspace/basics/miniconda3/envs/openwam/bin/python -m benchmarks.vla_harness.run_eval \
  --benchmark libero-plus \
  --ckpt-dir outputs/libero-plus/vjepa21_cached_text_step10690_harness/checkpoint_for_eval \
  --ckpt-name checkpoint_step_10690.safetensors \
  --libero-path /workspace/code/benchmarks/LIBERO-plus \
  --libero-plus-python /workspace/basics/miniconda3/envs/libero-plus/bin/python \
  --gpus 0,1,2,3 --workers 32 --max-batch-size 8 --max-wait-time 0.01 \
  --render gpu --render-gpus 3 --smoke
```

完整测评使用同一条命令去掉 `--smoke`。32 workers / batch8 是从标准 LIBERO 已完成长程验证的配置沿用，尚未在 Plus 上独立测速。`--categories` 按扰动类别选择子集，覆盖对应类别在所有底层 suite 中的任务。例如：

```bash
bash scripts/eval_libero_vjepa.sh CKPT_DIR CKPT_NAME --benchmark libero-plus \
  --correct-instruction-suffix true --categories "Sensor Noise,Camera Viewpoints" --dry-run
```

七个可选名称为 `Objects Layout`、`Camera Viewpoints`、`Robot Initial States`、`Language Instructions`、`Light Conditions`、`Background Textures`、`Sensor Noise`。省略 `--categories` 评测全部七类；名称必须完整且不重复。`--suites` 只用于标准 LIBERO；Plus 内部仍以 suite/task_id 定位官方任务和保留去重记录。`--task-ids` 为 0-based 索引，可进一步与类别取交集。

四卡各一个模型服务；公共入口默认每卡 16 个常驻 worker，环境进程随任务创建。环境子进程使用 Python Pipe 传送原始 ndarray；父进程动作以 Python list 跨越 NumPy 2.x→1.x 边界，子进程恢复为官方使用的 float32，不改变数值。模型批处理路径减少重复准备和拷贝；去噪、DiT cache 判据和动作缓冲语义保持不变。

启动前会先在 CPU 上读取官方任务语言，检查当前 checkpoint 的离线缓存能覆盖所选任务。缺失时在加载任何 GPU 模型前报错；不会静默替换指令。原生加载 T5 的 checkpoint 不需要这项离线缓存准备。

## 调用逻辑

1. `run_eval.main → make_plan`：选择 `--benchmark libero-plus`，由 `task_catalog` 对照官方 `task_classification.json` 与 `libero_suite_task_map.py` 生成任务列表。输入为扰动类别子集（可进一步限制 task ID），输出为模型服务和 worker 分片计划；每任务 1 trial。
2. `run_eval.launch`：验证环境/文本缓存，读取 `configs/libero_plus.yaml`，沿用原 launcher、RuntimeMonitor、SQLite WAL 和四卡模型服务。额外给 worker 传递 Plus 仓库、解释器与私有 config 路径。
3. `OpenWAMLIBEROPlusBenchmark.get_tasks/reset`：读取轻量任务目录；每次 reset 启动 `OfficialPlusEnvironment`。不会在常驻 worker 中导入实际 Plus 仿真库或缓存全部初始状态。
4. `libero_plus_env.serve → single_eval._control_seed → benchmark.get_task/get_task_init_states → single_eval._make_env_with_randomization_retries`：使用官方解释器，在新子进程中导入官方模块、建立环境、reset、set_init_state、settle。向父进程返回原始 obs、准确 language、task_name。
5. `make_obs → make_observation → harness Connection`：将两幅图像和 achieved EEF10 转成已验证的 harness 观测。`OpenWAMModelServer.predict_batch → BatchedOpenWAMEngine.generate_batch` 沿用标准 LIBERO 实现。
6. `StepBenchmark.step → OfficialPlusEnvironment.step → env.step`：接收模型的 7D 动作，返回原始 obs/reward/done/info；沿用原 `check_done/get_step_result` 执行 600/700 步上限和 done 成功判断。
7. 上游 merge 后 `summarize_plus`：以保存的官方元数据汇总七类扰动和逐任务明细及逐任务 difficulty；检查缺失、重复、异常和非 trial0 记录，完整 10030 条才输出全 benchmark 总成功率。

结果报告为 `summary.json`、`summary.csv` 和 `RESULTS.md`，包含所选扰动类别及总成功率；完整运行共七类加 Overall。内部保留任务目录、最小逐任务完成记录和异常日志用于续跑与正确计数。子集可报告所选集合的结果，`complete` 与 `total_average_success_rate` 区分是否覆盖完整 benchmark；smoke 不报告成功率。环境异常会传回 worker 并进入已有失败检测，不能当作正常模型失败；子进程继承本 worker 的进程组，关闭 Pipe/清理 worker 时一并退出。


## 公共吞吐与归档优化（2026-10-09）

所有 LIBERO-Plus 配置共用以下路径，原生协议、图像、history、动作与去噪步数不变：

- 不传 `--workers` 时，每张推理 GPU 配 16 个环境 worker；默认 EGL 渲染使用最后一张所选 GPU；通过 `--render-gpus` 可显式均分到多张已验证的渲染卡。`--workers`、`--render-gpus` 可显式覆盖，适配 CPU/显存较小的机器。
- `--max-batch-size` 仍是显存保护上限（默认 8）；应先用目标模型实测，再提高到 16/32 等。不能保证一个固定大 batch 对所有 checkpoint 都安全。
- OpenWAM 批量推理复用同一批内相同 seed/grid 的初始视频噪声，省去重复 CPU 随机数生成和拷贝；全批计算不再 gather/scatter 整批输入，最终动作一次传回 CPU。缓存的每样本判定、x/v 预测形式、归一化、独立视角及历史帧语义保持原样。
- 测评结束后在节点本地制作并校验 SQLite 快照，再顺序复制到结果目录；从本地 DB 合并结果，避免共享文件系统上的大量 SQLite 随机读写。默认 SQLite 仅保留任务标识、成功/失败及异常等续跑/计数记录，不保存环境逐步数据、推理轨迹、视频或派生 JSONL；默认报告七类成功率和全 Plus 总成功率。需要逐步记录与导出时在启动时显式传 `--export-step-jsonl`。以前运行已保存的逐步数据仍可从旧 DB 导出。

CLI 示例中的旧 `--workers 32 --render-gpus 3` 是显式历史参数；删除它们才能采用新的公共默认策略。批大小和 worker 数改变浮点批处理分组，性能应在目标模型与实际任务上复核。

### 启动与吞吐检查

公共 launcher 并行加载每张所选 GPU 的模型副本，全部 `/health` 就绪后才启动环境 worker。初始化期间仍检查所有所属进程，保留失败原因并清理本 launcher 创建的进程组。CPU 内存需能容纳并发构建的副本；不改变 checkpoint、精度或测评协议。

模型日志每30秒记录一次批大小计数、请求数、前处理与生成耗时，不含观测或动作轨迹。评估并发调整时，以完成相同工作量的墙钟耗时为准，把启动、环境重置、渲染和结果处理均计入；GPU利用率只辅助定位瓶颈。不同任务的步数差异会影响短窗口的“任务/分钟”，不能仅用它或GPU占用推断完整测评加速比。
