# 公共评测运行时：第一批实现与验证

2026-11-05。对应 `ROBOSCOPE_EVALUATION_SPEED_PLAN.md` 中 F0/F1 的初始实现，适用于整套框架，不依赖 VPP2。

## 已实现

| 文件 | 作用 |
|---|---|
| `benchmarks/vla_harness/execution.py` | 执行长度、已知预测长度、历史帧偏移、原生 batch/逐步 observe/状态能力声明；公共阶段计时 |
| `benchmarks/vla_harness/chunk_server.py` | 复用现有原生合批服务，增加可选完整动作块返回、episode token 和控制步校验 |
| `benchmarks/vla_harness/client.py` | worker 消费动作队列，按实际控制步维护历史，在重规划时上传所需帧 |
| `benchmarks/vla_harness/performance.py` | 汇总各进程最后一个累计统计快照，输出请求比和真实 batch 分布 |
| `scripts/profiling/profile_evaluation_transport.py` | 无 GPU 的真实 WebSocket 配对通信探针，显式标记替身策略和测量范围 |

OpenWAM/M29、FM-VJEPA、VPP2 都继承公共服务并声明自身能力；没有为每个模型复制客户端队列实现。M29 历史配置由 Encoder 读取；FM 原执行长度 16、历史偏移 -4；VPP2 原执行长度 10。预测长度与执行长度分开。

`run_eval` 和 `run_resumable` 已支持 `--client-action-queue`。未启用时使用原每步请求路径；旧续跑任务不会因为新增的默认 false 字段改变身份。开启后作为明确配置写入新任务身份及 run manifest，不能与未开启的旧 attempt 无记录混跑。

没有修改依赖源码或安装包。worker 入口在本进程内选择本地 Connection 子类，继续使用现有 harness runner。每个真实控制步的动作执行、done 检查、超时计数、录像与结果记录仍由原 runner/benchmark 完成。

安全边界：

- 历史按每个控制步更新，不按请求次数更新；episode 前几帧按原规则补首帧。
- 对历史图像保留独立副本，防止模拟器重用可变 buffer。
- reset/end/close 清空动作和历史；响应必须匹配当前 episode token 和控制步。
- 块级模式下，中途断线使当前 episode 报错，禁止自动把请求重放到丢失状态的新会话。
- 声明必须逐步 observe 的 adapter 不能启用跳过观测；无原生 batch 的 adapter 不能配置大 batch。
- worker 仍绑定同一模型实例，未引入跨实例迁移会话状态。

## 如何启用与查看统计

在原启动命令上增加：

```text
--client-action-queue
```

查看单个 attempt：

```bash
python -m benchmarks.vla_harness.performance /path/to/attempt
```

输出 `performance/*.jsonl` 按角色分为 client、model、environment。统计包括请求往返、模型调用、实际 batch、环境初始化/step、观测处理；RuntimeMonitor 增加 cgroup CPU 配额和 CPU/内存/IO pressure。各阶段是 CPU wall 时间，存在重叠，不可简单相加，也不是 CUDA kernel 计时。

不同模型的 VAE/CLIP/denoise 等详细 GPU 分段、自动并发调优与性能 profile 持久化仍待后续实现。

## 验证范围与结果

测试使用现有 openwam Python 环境，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MSGPACK_PUREPYTHON=1`。不加载真实模型权重，不启动模拟器，不占用 GPU。

主要覆盖：

1. 真实 WebSocket 下多会话合批，原路径与新路径的预测输入和动作一致。
2. M29 独立双视角/旧拼图、历史 0/1/2 tubelets；FM/VPP2 实际 adapter 类通过替身策略接入公共服务。
3. 预测块未耗尽时终止 episode、同连接重启、剩余动作清除、不同会话隔离。
4. 真实 SyncEpisodeRunner 的控制步上限与块中途 done，不把一个动作块当一个控制步。
5. 重放步号、旧 episode token、断线重放、图像 buffer 复用和能力不兼容检查。
6. 新旧续跑身份、各模型 CLI 布尔参数解析、启动等待/失败清理与 manifest。
7. OOM 拆批顺序、GPU owner thread 串行化等既有回归。

最终针对性回归：**158 passed，2 deselected，0 failures/errors**，约 39 秒。机器可读结果：`outputs/evaluation_runtime_validation/targeted.xml`。三个模型服务的 `--print_config` 均确认新布尔参数可解析；`compileall` 和 `git diff --check` 通过。

较大范围检查识别到两个与本批实现无关的已有记录接口测试不兼容，最终针对性检查显式排除它们，没有修改记录模块行为：

- `test_summary_only_export_preserves_steps_and_matches_full_export`：当前依赖的 `merge_db` 没有 `export_step_jsonl` 参数。
- `test_episode_only_recording_keeps_result_without_step_rows`：当前依赖的 recorder 在 `record_step=False` 时仍返回非空 `db_path`；该测试也调用前述缺失参数。

启动测试原先同样要求给 `merge_db` 传入已经移除的参数，已将其 mock 对齐实际 API，同时增加开关开/关覆盖。旧任务身份测试明确指定旧协议的 `--correct-instruction-suffix false`，避免继承新默认值而把协议变化误判为本批配置变化；新增测试还覆盖当前协议下开关开/关的续跑身份。

### 通信专项结果

配置：4 个会话、各 60 个控制步、两路 256×256 RGB、执行长度 10、raw 图像传输、Python MessagePack；每模式重复 3 次、交替先后次序。使用替身策略和真实 WebSocket，不含模拟器、真实模型与启动耗时。

| 条件 | 原请求数 → 新请求数 | 原图像数组 payload → 新 payload | 原路径中位时间 → 新路径 |
|---|---|---|---|
| 单帧模型 | 240 → 24 | 90 MiB → 9 MiB | 约 0.230s → 0.049s |
| 历史偏移 [-4,0] | 240 → 24 | 90 MiB → 27 MiB | 约 0.241s → 0.092s |

两组中模型预测次数均为 24，预测输入和逐步动作一致。历史模式需要传额外条件帧，因此 payload 不会像单帧模式一样减少十倍；当前协议还会重复包含条件帧中的当前帧。

上述 payload 是接收观测中图像数组的未压缩字节数，不包含协议开销。时间只是此机器当时通信探针的结果，不能作为完整评测的加速比例，也不能证明真实闭环成功率不变。

完整数据：

- `outputs/evaluation_runtime_validation/transport_single_frame.json`
- `outputs/evaluation_runtime_validation/transport_history.json`

## 尚未验证与下一阶段

四张 H100 当前均在运行训练，因此没有启动真实模型的 GPU 闭环或吞吐测试。新路径默认保持显式开启，待空闲资源下进行同任务、同 renderer、同 batch 的真实模型配对验证。

尚未实现自动并发搜索、模拟子进程 `step_chunk`、跨任务常驻进程池和 MPS 配置。当前节省的是 worker→模型服务的逐步观测请求；Plus 模拟子进程→worker 的逐步图像 Pipe 传输仍存在。

后续先用 M29 和一个无历史模型跑真实配对子集，检查输入、动作误差、任务成功差异和端到端时间；再开展 F2 并发/batch 搜索，最后引入子进程通信优化。全过程保持各模型和 benchmark 的既有 seed、settle、horizon 与动作协议。
