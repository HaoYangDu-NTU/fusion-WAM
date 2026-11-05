"""E00+ acceptance with real V-JEPA weights/data and optional full-size M29.

Default is a reduced two-layer policy on CPU. This is not a benchmark or a
formal run. Saves a report, including strict production-loader round-trip.
"""
import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from openwam.dataloader.registry import build_dataset
from openwam.experiments import preflight, resolve_experiment
from openwam.model import build_architecture, resolve_architecture_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/E00_plus/acceptance.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    cfg, profile, _ = resolve_experiment("E00+")
    cache = preflight(cfg, profile)
    if not args.full_policy:
        for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=2, freq_dim=32).items():
            cfg.model.video_backbone.dit_config[key] = value
        cfg.model.action_backbone.dim = 48
        cfg.model.action_backbone.ffn_dim = 96
    started = time.monotonic()
    dataset = build_dataset(cfg.dataloader, split="train")
    sample = dataset._getitem_impl(64)
    spec = resolve_architecture_config(cfg.model)
    model = build_architecture(spec.registry_name, spec.params)
    model.set_dtype_device(torch.bfloat16, device)
    model.freeze_modules(cfg.model.freeze)
    model.init_training_schedulers()
    model.set_training_runtime(noise_schedule_distribution="logit_normal")
    model.train()
    print("Loaded actual V-JEPA and M29 policy", flush=True)
    inputs = model.prepare_inputs([sample])
    inputs["use_gradient_checkpointing"] = True
    assert inputs["input_latents"].shape == (1, 1408, 5, 14, 28)
    assert inputs["num_clean_condition_latents"] == 1
    assert not inputs["input_latents"].requires_grad
    encoder = model.video_backbone.video_encoder
    assert all(not p.requires_grad for p in encoder.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    rows = []
    # Explicitly exercise warmup and active sparsity without changing formal config.
    for number, optimizer_step in enumerate((0, 5000)):
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        result = model.compute_loss(**inputs, optimizer_step=optimizer_step)
        result["loss"].backward()
        for module in (model.action_router.q, model.action_router.k, model.action_router.v,
                       model.action_router.out, model.video_backbone.dit.blocks[0].self_attn.q):
            assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
            assert module.weight.grad.float().norm() > 0
        assert all(p.grad is None for p in encoder.parameters())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        assert all(torch.isfinite(v) for v in result.values())
        rows.append(dict(step=number, optimizer_step=optimizer_step, seconds=time.monotonic()-tick,
                         metrics={key: float(value.detach()) for key, value in result.items()}))
        print(f"Backward/step passed, optimizer_step={optimizer_step}", flush=True)
    assert rows[0]["metrics"]["loss_router_sparse"] == 0
    assert rows[1]["metrics"]["loss_router_sparse"] > 0

    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.train.utils.checkpointing import save_config
    forward = {k: v for k, v in inputs.items()
               if k not in ("actions", "action_is_pad", "video_is_pad", "proprio", "proprio_mask",
                            "use_gradient_checkpointing", "use_gradient_checkpointing_offload")}
    forward["latents"] = forward.pop("input_latents")
    forward["timestep"] = torch.tensor([500.], device=device, dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        expected = model(inputs["actions"], forward["timestep"], proprio=inputs["proprio"], **forward)
    with tempfile.TemporaryDirectory(prefix="E00_plus_reload_") as temporary:
        directory = Path(temporary)
        model.video_backbone.save_deploy_assets(temporary, cfg)
        save_config(temporary, cfg)
        shutil.copyfile(dataset.normalization_stats_path, directory / "normalization_stats.npy")
        model.save_checkpoint(str(directory / "checkpoint.safetensors"))
        # Release optimizer/master weights before constructing the deployment shell.
        del optimizer, model, encoder, result
        gc.collect()
        _, restored = load_from_checkpoint_dir(temporary, str(device), "checkpoint.safetensors")
        assert restored.action_router is not None
        with torch.no_grad():
            actual = restored(inputs["actions"], forward["timestep"], proprio=inputs["proprio"], **forward)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(actual[1]).all() and actual[1].shape == (1, 32, 80)
        print("Production loader strict reload and action forward passed", flush=True)
        # Exercise the benchmark's real multistep sampler, not just one forward.
        import numpy as np
        from omegaconf import OmegaConf
        from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
        native = SimpleNamespace(architecture=restored, _cfg_scale=1., _decode_video=False,
            _dit_cache=None, _prompt_embed_cache={}, cfg=OmegaConf.create({"inference": {
                "denoise_mode": "sync", "denoise_steps": 2, "num_frames": 33,
                "video_num_frames": 9, "height": 224, "width": 448}}))
        engine = BatchedOpenWAMEngine(native)
        observations = [dict(prompt=sample["prompt"], image=sample["video"][0],
            history_image=sample["clean_condition_images"][0],
            clean_condition_images=sample["clean_condition_images"], state=np.zeros(10, np.float32))]
        chunks = engine.generate_batch(observations)
        assert chunks[0].shape == (32, 10) and np.isfinite(chunks[0]).all()
        print("Two-step benchmark sampler produced finite [32,10] actions", flush=True)

    report = dict(passed=True, experiment="E00+", device=str(device),
                  real_pretrained_encoder=True, real_libero_reader=True,
                  full_policy=args.full_policy, policy_layers=cfg.model.video_backbone.dit_config.num_layers,
                  text_cache=cache, raw_latent_shape=list(inputs["input_latents"].shape),
                  strict_production_reload=True, reload_prediction_max_abs_difference=0.,
                  action_prediction_shape=list(actual[1].shape), rows=rows,
                  benchmark_sampler_steps=2, benchmark_action_chunk_shape=list(chunks[0].shape),
                  elapsed_seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
