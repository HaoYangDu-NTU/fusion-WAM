"""Real E05 encoder + actual LIBERO reader + reduced M29 backward, CPU only.

Reports measured CPU costs explicitly; never substitutes them for full-size GPU
training costs and never starts a formal training run.
"""
import argparse
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.model import build_architecture, resolve_architecture_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-policy", action="store_true", help="Keep the actual 30-layer M29 and Action Expert")
    parser.add_argument("--steps", type=int, default=2, choices=range(1, 4))
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cfg, profile, _ = resolve_experiment("E05")
    dataset = build_dataset(cfg.dataloader, split="train")
    sample = dataset._getitem_impl(64)
    original = dict(video_dim=cfg.model.video_backbone.dit_config.dim,
                    video_layers=cfg.model.video_backbone.dit_config.num_layers,
                    action_dim=cfg.model.action_backbone.dim)
    if not args.full_policy:
        for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
            cfg.model.video_backbone.dit_config[key] = value
        cfg.model.action_backbone.dim = 48
        cfg.model.action_backbone.ffn_dim = 96
    started = time.monotonic()
    spec = resolve_architecture_config(cfg.model)
    model = build_architecture(spec.registry_name, spec.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.freeze_modules(cfg.model.freeze)
    model.init_training_schedulers()
    model.train()
    load_seconds = time.monotonic() - started
    encoder = model.video_backbone.video_encoder
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(not m.training for m in encoder.modules())
    started = time.monotonic()
    inputs = model.prepare_inputs([sample])
    encode_seconds = time.monotonic() - started
    features = inputs["input_latents"]
    assert features.shape == (1, 2048, 10, 14, 28)
    assert inputs["num_clean_condition_latents"] == 2
    assert torch.isfinite(features).all() and not features.requires_grad
    raw = features.clone()
    # Validate leakage with the real native pretrained aggregator, not just a mock.
    images = [sample["clean_condition_images"][0], *sample["video"]]
    pixels = encoder.preprocess_video(images)
    original_z = encoder.batch_encode(pixels)
    pixels[:, :, 2:] = torch.rand_like(pixels[:, :, 2:])
    altered_z = encoder.batch_encode(pixels)
    torch.testing.assert_close(original_z[:, :, :2], altered_z[:, :, :2], rtol=0, atol=0)
    assert not torch.equal(original_z[:, :, 2:], altered_z[:, :, 2:])
    projection = model.video_backbone.dit.patch_embedding
    with torch.no_grad():
        grid = projection(features, num_clean_condition_latents=2)
    assert grid.shape == (1, cfg.model.video_backbone.dit_config.dim, 10, 7, 14)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.training.learning_rate, betas=tuple(cfg.training.adam_betas),
                                 weight_decay=cfg.training.weight_decay)
    before = projection.clean_proj.weight.detach().clone()
    target_checks = []
    original_loss = model._compute_video_loss
    def inspect_target(prediction, target, *args, **kwargs):
        assert not target.requires_grad
        target_checks.append(dict(shape=list(target.shape), requires_grad=target.requires_grad))
        return original_loss(prediction, target, *args, **kwargs)
    model._compute_video_loss = inspect_target
    rows = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        losses = model.compute_loss(**inputs)
        forward = time.monotonic() - started
        assert all(torch.isfinite(v) for v in losses.values())
        started = time.monotonic()
        losses["loss"].backward()
        backward = time.monotonic() - started
        grad = {name: float(module.weight.grad.float().norm()) for name, module in (
            ("clean_proj", projection.clean_proj), ("future_proj1", projection.proj1),
            ("future_proj2", projection.proj2), ("output_head", model.video_backbone.dit.head.head))}
        assert all(value > 0 for value in grad.values())
        block_gradient = model.video_backbone.dit.blocks[0].self_attn.q.weight.grad
        assert block_gradient is not None and torch.isfinite(block_gradient).all()
        assert block_gradient.float().norm() > 0
        grad["m29_first_block_q"] = float(block_gradient.float().norm())
        assert all(p.grad is None for p in encoder.parameters())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.training.max_grad_norm)
        optimizer.step()
        rows.append(dict(step=step+1, forward_seconds=forward, backward_seconds=backward,
                         losses={k: float(v.detach()) for k, v in losses.items()}, gradient_norms=grad))
    assert not torch.equal(before, projection.clean_proj.weight)
    torch.testing.assert_close(raw, inputs["input_latents"], rtol=0, atol=0)
    reload_checks = None
    if args.full_policy:
        import numpy as np
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        from safetensors.torch import save_file
        from openwam.train.utils.checkpointing import save_config
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from benchmarks.vla_harness.checkpoint_contract import validate_vggt_omega_checkpoint
        from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="E05_strict_reload_") as temporary:
            directory = Path(temporary)
            model.video_backbone.save_deploy_assets(str(directory), cfg)
            save_config(str(directory), cfg)
            shutil.copyfile(dataset.normalization_stats_path, directory / "normalization_stats.npy")
            save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
                      str(directory / "checkpoint.safetensors"))
            contract = validate_vggt_omega_checkpoint(directory, "checkpoint.safetensors", OmegaConf.to_container(cfg))
            _, restored = load_from_checkpoint_dir(str(directory), "cpu", "checkpoint.safetensors")
            assert all(not p.requires_grad for p in restored.video_backbone.video_encoder.parameters())
            restored_z = restored.prepare_inputs([sample])["input_latents"]
            torch.testing.assert_close(raw, restored_z, rtol=0, atol=0)
            native = SimpleNamespace(architecture=restored, _cfg_scale=1., _decode_video=False,
                _dit_cache=None, _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {
                    "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 33,
                    "video_num_frames": 9, "height": 224, "width": 448}}))
            engine = BatchedOpenWAMEngine(native)
            observations = [dict(prompt=sample["prompt"], image=sample["video"][0], history_image=sample["clean_condition_images"][0], state=np.zeros(10, np.float32))]
            batch, _, _ = engine._prepare(observations)
            ordinary = restored.video_backbone.preprocess_input_for_inference(prompt=sample["prompt"],
                clean_condition_images=sample["clean_condition_images"], num_frames=9, height=224, width=448,
                seed=42, num_inference_steps=2)
            torch.testing.assert_close(batch["clean_condition_latents"], ordinary["clean_condition_latents"], rtol=0, atol=0)
            assert batch["latents"].shape == (1, 2048, 10, 14, 28)
            chunks = engine.generate_batch(observations)
            assert chunks[0].shape == (32, 10) and np.isfinite(chunks[0]).all()
            reload_checks = dict(full_real_checkpoint=True, strict_reload=True, contract=contract,
                frozen_features_max_abs_difference=0., train_eval_clean_max_abs_difference=0.,
                action_chunk_shape=list(chunks[0].shape), actions_finite=True,
                temporary_checkpoint_deleted=True, elapsed_seconds=time.monotonic()-started)
    report = dict(passed=True, real_pretrained_encoder=True, real_libero_reader=True,
                  device="cpu", threads=4, batch_size=1,
                  policy_size="full 30-layer architecture" if args.full_policy else "reduced for CPU smoke only",
                  original_policy=original, smoke_policy=dict(video_dim=cfg.model.video_backbone.dit_config.dim,
                      video_layers=cfg.model.video_backbone.dit_config.num_layers, action_dim=cfg.model.action_backbone.dim),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  encoder_parameters=sum(p.numel() for p in encoder.parameters()), encoder_trainable=0,
                  load_seconds=load_seconds, encode_10_times_seconds=encode_seconds,
                  input_rgb_frames=len(images), per_view_resolution=[224,224],
                  raw_feature_shape=list(features.shape), projected_shape=list(grid.shape),
                  tokens=grid.shape[2:].numel(), clean_latents=2, future_latents=8,
                  video_padding_mask_shape=list(inputs["video_is_pad"].shape),
                  target_checks=target_checks, future_leakage_max_abs=0.0,
                  frozen_targets_unchanged=True, projection_updated=True, steps=rows,
                  reload_checks=reload_checks,
                  peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  full_size_gpu_timing_measured=False)
    destination = "E05_full_policy_cpu.json" if args.full_policy else "E05_real_pipeline_cpu.json"
    (ROOT / "reports" / destination).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
