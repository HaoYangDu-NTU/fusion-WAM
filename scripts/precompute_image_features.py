"""Estimate or explicitly build E01/E02 frame caches. No training is launched.

Each real stereo timestamp is encoded once; CPU prefetch and bounded asynchronous
atomic shard writers overlap with encoding. Use torchrun for independent GPUs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from torch.utils.data import DataLoader

from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.spatial_frame_cache_specs import frame_cache_identity
from openwam.model.video_backbone.encoder.registry import build_video_encoder
from scripts.feature_cache.frames import CurrentStereoFrames, atomic_json
from scripts.precompute_wan_latents import write_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E01", "E02", "E07", "E08"), required=True)
    parser.add_argument("--build", action="store_true", help="Without this flag, estimate only")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-frames", type=int, help="Bounded smoke cache, not valid for full training")
    parser.add_argument("--batch-size", type=int, default=8, help="Stereo timestamps per encoder call")
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--write-inflight", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.loader_workers < 0 or args.write_inflight < 1 or (args.limit_frames is not None and args.limit_frames < 1):
        parser.error("Invalid batch, worker, inflight or frame limit")
    torch.set_num_threads(4 if args.device == "cpu" else 1)
    cfg, _, _ = resolve_experiment(args.experiment + "-Cache")
    output = Path(args.output or cfg.dataloader.frame_feature_cache_dir)
    cfg.dataloader.frame_feature_cache_dir = None
    reader = build_dataset(cfg.dataloader, split="train")
    frames = CurrentStereoFrames(reader)
    identity = frame_cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, reader)
    row_bytes = 2 * torch.tensor(identity["shape"]).prod().item()
    estimate = dict(experiment=args.experiment, frames=len(frames), shape=identity["shape"],
                    dtype=identity["dtype"], estimated_gib=row_bytes * len(frames) / 2**30,
                    naive_window_gib=row_bytes * 10 * len(frames) / 2**30,
                    output=str(output), estimate_only=not args.build)
    print(json.dumps(estimate), flush=True)
    if not args.build:
        return
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    # Same lock+atomic manifest publication as the existing Wan builder.
    import fcntl
    with (output / "manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing cache identity differs; use a new directory")
        atomic_json(manifest, identity)
    encoder = build_video_encoder(cfg.model.video_backbone.encoder).to(device=device, dtype=torch.bfloat16)
    encoder.requires_grad_(False).eval()
    stop = min(len(frames), args.limit_frames or len(frames))
    size = identity["shard_size"]
    indices = []
    for shard_start in range(0, stop, size):
        if (shard_start // size) % world != rank:
            continue
        path = output / f"shard_{shard_start // size:06d}.safetensors"
        existing = set()
        if path.exists():
            with safe_open(path, framework="pt") as handle:
                existing = set(handle.keys())
        indices.extend(index for index in range(shard_start, min(shard_start + size, stop)) if str(index) not in existing)
    loader = DataLoader(frames, batch_size=args.batch_size, sampler=indices,
                        num_workers=args.loader_workers, pin_memory=device.type == "cuda",
                        **(dict(prefetch_factor=2, multiprocessing_context="spawn") if args.loader_workers else {}))
    started = time.monotonic()
    pending, additions, current_shard = [], {}, None
    processed = 0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-cache-write") as writer:
        def submit(shard, values):
            if values:
                if len(pending) >= args.write_inflight:
                    pending.pop(0).result()
                pending.append(writer.submit(write_shard, output / f"shard_{shard:06d}.safetensors", values))
        with torch.no_grad():
            for batch in loader:
                # Use the exact existing PIL/ImageNet preprocessing, including BF16 rounding.
                from PIL import Image
                images = [Image.fromarray(p.permute(1, 2, 0).numpy()) for p in batch["pixels"]]
                pixels = torch.cat([encoder.preprocess_video([image]) for image in images])
                raw = encoder._encode_frames(pixels).squeeze(2).cpu().contiguous()
                if list(raw.shape[1:]) != identity["shape"] or raw.dtype != torch.bfloat16 or not torch.isfinite(raw).all():
                    raise ValueError("Unexpected/nonfinite frozen frame grids")
                for index, value in zip(batch["index"].tolist(), raw):
                    shard = index // size
                    if current_shard is not None and shard != current_shard:
                        submit(current_shard, additions)
                        additions = {}
                    current_shard = shard
                    additions[str(index)] = value.clone()
                    processed += 1
                if processed % 128 == 0:
                    print(json.dumps(dict(rank=rank, processed=processed, remaining=len(indices)-processed,
                                          seconds=time.monotonic()-started)), flush=True)
        submit(current_shard, additions)
        for future in pending:
            future.result()
    report = dict(**estimate, rank=rank, world_size=world, built=processed, seconds=time.monotonic()-started,
                  full_cache_requested=stop == len(frames), complete_local_writes=True,
                  cache_identity_checked=True, numerical_parity="Run scripts/smoke_image_frame_cache.py")
    atomic_json(output / f"build_rank{rank}.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
