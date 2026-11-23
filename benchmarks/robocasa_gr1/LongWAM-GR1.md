# RoboCasa-GR1 Tabletop 四卡训练与评测

此配置复用 Long-WAM 的 GR1 关节动作训练器及配套 socket 评测器。代码位于本目录，数据和环境位于项目根目录下的 `.benchmark_runtime/robocasa_gr1`。既有 OpenWAM GR1 客户端使用 EEF33。这份参考配置的 29 维绝对关节输出需要对应的关节动作客户端。后续接入 M29 时，须对齐 reader、动作头、状态编码、反归一化和控制器，并做闭环验证。

| 项目 | 推荐配置 |
|---|---|
| 数据 | NVIDIA Teleop-Sim，固定 revision `09c6de8af50168090e7e9cc01e1ec3bce788de24`。使用 24 个官方任务，每项 1,000 条示范，20 Hz，共 5,820,277 帧；指令取自 episode remarks。 |
| 输入与输出 | 单目 ego RGB，原视频 256×256，模型输入 224×224。原始状态和动作均为 44 维，按官方字段选 29 维。状态编码为 sin/cos 共 58 维，29 维绝对关节动作用官方任务统计计算全局 min-max；执行时还原绝对命令。 |
| 时序 | P384，即 19.2 秒历史。每窗口 417 个原始时间点，采样 105 个视频时间点，2 个未来 latent，每次预测 16 个动作。 |
| 训练 | 30,000 次优化器更新，全局 batch 480。上游 16×30×1，本机建议 4×5×24，每卡 8 个数据 worker，prefetch 2。该分解已通过配置合成，仍需实际显存和吞吐测试。 |
| 优化器 | AdamW，LR 3e-5，β=(0.95,0.999)，ε=1e-8，weight decay 1e-5，梯度裁剪 1，bf16，seed 42。前 1,500 次更新 warmup，随后 cosine_hf。 |
| 权重与保存 | Long-WAM 官方参考配方以 Robot-S 初始化视频专家，使用对应的映射动作权重。每 5k 步保存权重和恢复状态。复用此优化预算训练自有模型时，单独披露初始化来源。 |

## 运行

从 openwam-exp 根目录执行。数据下载支持断点续传，最后逐任务校验 parquet 和视频配对、帧数，并生成动作统计。

```bash
export PATH=/data/fcy/basics/miniconda3/envs/openwam/bin:$PATH
source /data/fcy/code/proxy.sh
proxy_global
export HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1
bash benchmarks/robocasa_gr1/prepare_teleop.sh
PYTHON=$(command -v python) bash benchmarks/robocasa_gr1/longwam_4gpu.sh --dry-run
```

训练时使用安装了 Long-WAM 训练依赖的解释器，配置好官方 Robot-S、动作初始化权重和文本缓存后运行同一入口并去掉 `--dry-run`。本次只准备数据、环境和配置。复现用 LongLive commit 为 `31c70d9071cd7851d82939806dc9fac645fcabe5`。[上游训练说明](https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/Long-WAM/docs/TRAINING.md) · [数据与缓存说明](https://github.com/NVlabs/LongLive/blob/31c70d9071cd7851d82939806dc9fac645fcabe5/Long-WAM/docs/DATA.md)

## 评测口径

官方 GR1 仓库提供 24 项任务和每项 10 局、720 控制步的演示命令。Long-WAM 默认同样每项 10 局。这里“榜单”采用本项目收录的 VLA Evaluation Harness GR1 口径：每项 100 局，评最近 5 个 checkpoint。按当前周期预先固定 10k、15k、20k、25k、30k，合计 12,000 局。每个 checkpoint 计算 24 项成功率的算术平均，以最高均值对应的单一 checkpoint 填 `suite_scores.gr1`，其各项分数以 `_gr1` 结尾，`overall_score=null`。保存所有 checkpoint 和分任务原始成绩。没有核实到独立的官方 GR1 提交审核规则。[官方模拟器与任务](https://github.com/robocasa/robocasa-gr1-tabletop-tasks/tree/4840e671596f93ca03651524b9f72ffb1aadfeff) · [本地收录规则](../../third_party/vla-evaluation-harness/leaderboard/benchmarks/robocasa.md)

模拟器锁定上面 commit，MuJoCo 3.2.6、robosuite 1.5.1。完成依赖和 assets 安装后，用现有入口验证 reset、渲染和一步交互，再运行上游 Long-WAM 关节策略评测器。

```bash
runtime="$PWD/.benchmark_runtime/robocasa_gr1"
export ROBOCASA_GR1_PATH="$runtime/robocasa-gr1-tabletop-tasks-4840e671596f93ca03651524b9f72ffb1aadfeff"
export ROBOCASA_GR1_PYTHON="$runtime/sim-env/bin/python"
bash benchmarks/robocasa_gr1/run_smoke.sh env
```

对每个待评权重执行下列命令。`BUNDLE` 指向包含匹配的 model.pt 和 config.yaml 的目录，`TEXT_CACHE` 为该策略的文本缓存，`RESULT` 为该 checkpoint 独立的输出目录；这三个变量均使用绝对路径。

```bash
cd "$runtime/LongLive/Long-WAM"
bash scripts/robocasa_gr1/eval.sh expected_history=384 episodes=100 max_control_steps=720 \
  checkpoint="$BUNDLE/model.pt" model_config="$BUNDLE/config.yaml" \
  stats="$runtime/global_action_minmax.json" text_cache="$TEXT_CACHE" \
  simulator_python="$ROBOCASA_GR1_PYTHON" output_dir="$RESULT"
```

本地验证记录包括[四卡配置合成](../../.benchmark_runtime/robocasa_gr1/verified-dry-run.log)、[数据与模拟 rollout 协议测试](../../.benchmark_runtime/robocasa_gr1/contract-tests.log)和[29 维训练集动作统计](../../.benchmark_runtime/robocasa_gr1/global_action_minmax.json)。模拟 rollout 使用测试环境替身。全量数据下载日志在 `download.log`，下载后配对及帧数验收在 `verify-download.log`，两者均位于运行目录。
