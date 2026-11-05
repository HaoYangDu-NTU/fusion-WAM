"""Bounded E01/E02 real-weight CPU smoke; never starts full training or a full cache.

The policy is reduced to one layer for gradient checks. CPU timings are not
estimates of the 30-layer four-GPU training throughput.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.image_frame_cache import (
    CachedImageDataset, ImageFrameCache, ImageFrameCacheCollator,
)
from openwam.dataloader.spatial_frame_cache_specs import frame_cache_identity
from openwam.model import build_architecture, resolve_architecture_config
from scripts.precompute_flux2_features import atomic_json
from scripts.precompute_wan_latents import write_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("E01", "E02", "E07", "E08"))
    parser.add_argument("--sample-index", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", help="Report directory, separate from production caches")
    args = parser.parse_args()
    if not 1 <= args.steps <= 4:
        parser.error("Smoke steps must be 1..4")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = Path(args.output or ROOT / "reports/image_frame_cache" / args.experiment)
    out.mkdir(parents=True, exist_ok=True)
    cfg, profile, _ = resolve_experiment(args.experiment)
    reader = build_dataset(cfg.dataloader, split="train")
    sample = reader._getitem_impl(args.sample_index)
    identity = frame_cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, reader)
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        cfg.model.video_backbone.dit_config[key] = value
    cfg.model.action_backbone.dim = 48
    cfg.model.action_backbone.ffn_dim = 96
    started = time.monotonic()
    spec = resolve_architecture_config(cfg.model)
    model = build_architecture(spec.registry_name, spec.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.train()
    model.freeze_modules(list(cfg.model.freeze))
    model.init_training_schedulers()
    encoder = model.video_backbone.video_encoder
    load_seconds = time.monotonic() - started
    print(f"{args.experiment}: loaded real encoder, starting online preparation", flush=True)
    started = time.monotonic()
    online = model.prepare_inputs([copy.deepcopy(sample)])
    online_prepare_seconds = time.monotonic() - started
    clip = [sample["clean_condition_images"][0], *sample["video"]]
    with torch.no_grad():
        pixels = encoder.preprocess_video(clip)
        raw = encoder._encode_frames(pixels)[0].cpu().contiguous()
        # Offline batching can change BF16 kernel rounding. Measure it explicitly.
        individually_encoded = encoder._encode_frames(pixels[:, :, :1])[0, :, 0]
        difference = (individually_encoded.float() - raw[:, 0].float()).abs()
        # This audit is separate from exact cache serialization/loss parity below.
        # CPU BF16 kernels can depend on flattened image batch geometry; do not
        # declare a production GPU cache equivalent on the strength of CPU tests.
        geometry_cosine = torch.nn.functional.cosine_similarity(
            individually_encoded.float(), raw[:, 0].float(), dim=0).mean()
    directory = out / "bounded_cache"
    directory.mkdir(exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Smoke cache identity changed; choose a new --output directory")
    atomic_json(manifest, identity)
    cache = ImageFrameCache(directory)
    metadata_dataset = CachedImageDataset(reader, cache)
    # Verify action/proprio/text/padding equality and bypass the decoder at boundaries.
    metadata_checks = []
    for index in (0, args.sample_index, len(reader)-1):
        expected = reader._getitem_impl(index)
        with patch.object(reader, "_decode_window_video", side_effect=AssertionError("RGB decoded on a cache hit")):
            actual = metadata_dataset[index]
        for key in ("action", "action_mask", "video_mask", "proprio", "proprio_mask"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual["prompt"] == expected["prompt"]
        metadata_checks.append(index)
    cached_sample = metadata_dataset[args.sample_index]
    keys = cached_sample["image_frame_indices"]
    shards = {}
    for time_index, key in enumerate(keys):
        shards.setdefault(key // identity["shard_size"], {})[str(key)] = raw[:, time_index].contiguous()
    for shard, values in shards.items():
        write_shard(directory / f"shard_{shard:06d}.safetensors", values)
    started = time.monotonic()
    cached_sample = ImageFrameCacheCollator(directory)([cached_sample])[0]
    read_seconds = time.monotonic() - started
    with patch.object(encoder._m, "forward", side_effect=AssertionError("Encoder ran on a cache hit")):
        started = time.monotonic()
        cached = model.prepare_inputs([cached_sample])
        cached_prepare_seconds = time.monotonic() - started
    feature_key = "ijepa_frame_features" if args.experiment == "E02" else "input_latents"
    torch.testing.assert_close(cached[feature_key], online[feature_key], rtol=0, atol=0)
    # The cached clean prefix must remain independent of future observations.
    if args.experiment == "E02":
        with torch.no_grad():
            encoder.delta_scale.fill_(0.2)  # exercise both learned history parameters
        live = encoder._normalize(encoder._adapt_frame_grid(raw.unsqueeze(0)))
        changed = raw.unsqueeze(0).clone()
        changed[:, :, 2:] += 1
        altered = encoder._normalize(encoder._adapt_frame_grid(changed))
        torch.testing.assert_close(live[:, :, :1], altered[:, :, :1], rtol=0, atol=0)
    projection = model.video_backbone.dit.patch_embedding
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.adam_betas), weight_decay=cfg.training.weight_decay)
    phase_rows = []
    # Same noise seed and weights: compare full loss and representative gradients.
    parity = []
    for prepared in (online, cached):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        losses = model.compute_loss(**copy.deepcopy(prepared))
        losses["loss"].backward()
        parity.append((float(losses["loss"]), projection.proj1.weight.grad.detach().clone()))
    assert parity[0][0] == parity[1][0]
    torch.testing.assert_close(parity[0][1], parity[1][1], rtol=0, atol=0)
    before = projection.proj1.weight.detach().clone()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        losses = model.compute_loss(**copy.deepcopy(cached))
        forward = time.monotonic() - started
        assert torch.isfinite(losses["loss"])
        started = time.monotonic()
        losses["loss"].backward()
        backward = time.monotonic() - started
        assert all(p.grad is None for p in encoder._m.parameters())
        if args.experiment == "E02":
            for p in (encoder.delta_scale, encoder._delta_proj.weight, encoder.feature_norm.weight):
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        assert model.video_backbone.dit.blocks[0].self_attn.q.weight.grad.abs().sum() > 0
        optimizer.step()
        phase_rows.append(dict(step=step, loss=float(losses["loss"]), forward_seconds=forward,
                               backward_seconds=backward))
    assert not torch.equal(before, projection.proj1.weight)
    # State schema is unchanged; the cache is an external data artifact.
    model.load_state_dict(model.state_dict(), strict=True)
    report = dict(passed=True, experiment=args.experiment, device="cpu", real_pretrained_encoder=True,
                  real_libero_reader=True, reduced_policy_layers=1, batch_size=1,
                  raw_shape=list(raw.shape), tokens=profile["geometry"]["tokens"],
                  cached_online_features_bitwise_equal=True, loss_equal=True, projection_gradient_equal=True,
                  frozen_backbone_no_grad=True, projection_updated=True,
                  ijepa_adapter_gradients=args.experiment == "E02", metadata_indices=metadata_checks,
                  encoder_load_seconds=load_seconds, online_prepare_seconds=online_prepare_seconds,
                  cache_read_seconds=read_seconds, cached_prepare_seconds=cached_prepare_seconds,
                  batch_geometry_max_abs=float(difference.max()), batch_geometry_mean_abs=float(difference.mean()),
                  batch_geometry_mean_cosine=float(geometry_cosine),
                  production_gpu_numerics_validated=False,
                  estimated_full_cache_gib=len(reader)*raw[:, 0].numel()*raw.element_size()/2**30,
                  steps=phase_rows, full_training_started=False,
                  gpu_steady_step_time="not measured: existing four-GPU decoder job left running")
    atomic_json(out / "smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
