# 接入新的原生 encoder

公共宿主已有 VideoEncoder 扩展点；实验框架沿用它来挂载模型和配置，不增加统一的时序适配器。新 encoder 的输入预处理、历史帧接口、时间压缩、归一化、输出通道及训练范围由自身实现决定。

1. 在 `openwam/model/video_backbone/encoder/<name>.py` 实现并用 `@register_video_encoder("<name>")` 注册；从包 `__init__.py` 导入，或在 Hydra encoder 配置填写 `implementation_module: your_package.encoder`。工厂只消费 name/model_path/implementation_module，其他非 null 参数直接交给原生 from_pretrained。
2. 在 `configs/model/video_backbone/encoder/<name>.yaml` 写独立配置，不将 DINO/I-JEPA 的字段强加给新实现。现有 properties 描述 z_dim、spatial/temporal compression、dit_patch_size、num_condition_latents；这些是宿主计算尺寸所需的元信息，不要求它们相同。
3. 保存原生部署资产并实现 from_skeleton。saved components 记录类路径；重建使用 checkpoint 本地文件及 YAML，不依赖训练机器上的预训练路径。必要的权重必须包含在策略 checkpoint 中。
4. 在 `assets/registry.json` 增加独立权重项；`configs/experiments/E03.yaml` 写资产名、freeze_visual、geometry 和原生 Hydra overrides，再登记到 `configs/experiments.yaml`。相邻 `.yaml.example` 仅展示 profile 结构，不是可运行模型。
5. 如有可训练预处理，确认它在 autograd 区域运行。I-JEPA 在冻结特征提取后单独延迟执行原生适配器，就是一个明确的专用分支。新 encoder 可以有自己的桥接，不必套入 I-JEPA 的 `_adapt_frame_grid`。
6. 推理/批评测中显式处理该 encoder 自己的历史输入。若其帧语义不同，扩展 `wan_backbone.py`、`deploy/policy.py`、`benchmarks/vla_harness/batch_engine.py` 与 `model_server.py` 的专用分支。不要通过篡改其他 encoder 的 temporal_compression 来套用。
7. 在 `checkpoint_contract.py` 添加该模型独立的 sidecar/形状校验，在 `prepare_libero_plus_svae.py` 和 `run_train_eval.py` 添加原生资产清单；如评测流程不同，可增加单独 preparer。通用名字 `eval_libero_vjepa.sh` 是保留的历史文件名，当前支持 E00/E01/E02。
8. 检查 native 时间几何、训练梯度与 freeze、clean mask、checkpoint 严格重建、普通/批推理和 Plus 资产重定位，然后单独做 GPU smoke 和恢复验收。

现有实现可分别参考 `vjepa21.py`（视频 tubelet）、`dinov3.py`（framewise）、`ijepa.py`（可训练历史差分），每个都保留自己的配置和部署 sidecar。测试入口为 `test_exp_dinov3_native.py`、`test_experiment_infra.py`。
