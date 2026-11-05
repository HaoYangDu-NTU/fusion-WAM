"""Offline deterministic Wan2.1/2.2 window cache. torchrun supported; atomic shard writes.

Default is estimate-only. --build explicitly encodes all windows; --smoke-steps N
builds only the exact epoch-0 sampler prefix plus DataLoader prefetch for testing.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.wan_latent_cache import cache_identity
from openwam.model.video_backbone.encoder.wan_vae import WanVAEFeatureEncoder


class WindowPrefetchDataset(torch.utils.data.Dataset):
    """Spawn-safe CPU readers; never construct or copy a CUDA encoder in workers."""

    def __init__(self, config, indices):
        self.config, self.indices, self.dataset = config, indices, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.dataset is None:
            self.dataset = build_dataset(OmegaConf.create(self.config), split="train")
        index = self.indices[position]
        # Strict access is essential: fallback to another index would corrupt the cache.
        return index, self.dataset._getitem_impl(index)["video"]


def write_shard(path, additions):
    """Merge under the existing inter-process lock, then publish atomically."""
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        values = {}
        if path.exists():
            with safe_open(path, framework="pt") as handle:
                values = {key: handle.get_tensor(key).clone() for key in handle.keys()}
        for key, value in additions.items():
            if key in values:
                torch.testing.assert_close(values[key], value, rtol=0, atol=0)
            values[key] = value
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        try:
            save_file(values, str(temporary))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E03", "E04"), default="E03")
    parser.add_argument("--output")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--verify-count", type=int, default=4)
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Independent single-view encoder processes per GPU; preserves microbatch=1 numerics")
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--write-inflight", type=int, default=2)
    parser.add_argument("--limit-windows", type=int, help="Focused validation only: cache first N windows")
    args = parser.parse_args()
    if args.loader_workers < 0 or args.prefetch_factor < 1 or args.write_inflight < 1:
        parser.error("loader-workers must be nonnegative; prefetch-factor/write-inflight must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    from openwam.train.utils.seeding import seed_everything
    seed_everything(42, rank=rank)
    cfg, _, _ = resolve_experiment(args.experiment)
    configured_output = cfg.dataloader.get("latent_cache_dir")
    # Generation always reads original RGB; never attach an incomplete cache to its reader.
    cfg.dataloader.latent_cache_dir = None
    ds = build_dataset(cfg.dataloader, split="train")
    identity = cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, len(ds))
    output = Path(args.output or configured_output or "cache/wan21_libero_windows").resolve()
    import math
    estimate = len(ds) * math.prod(identity["shape"]) * 2
    print(json.dumps(dict(rank=rank, experiment=args.experiment, shape=identity["shape"],
                          windows=len(ds), raw_bytes=estimate, gib=estimate / 2**30)), flush=True)
    if not args.build and args.smoke_steps is None:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Existing cache identity differs; use a new output directory")
        else:
            temporary = manifest.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps(identity, indent=2) + "\n")
            os.replace(temporary, manifest)
    indices = list(range(len(ds)))
    if args.limit_windows is not None:
        if args.limit_windows < 1 or args.smoke_steps is not None:
            parser.error("limit-windows must be positive and cannot be combined with smoke-steps")
        indices = indices[:args.limit_windows]
    if args.smoke_steps is not None:
        # Accelerate shards the same EpochRandomSampler permutation into full batches.
        extra = int(cfg.training.dataset_num_workers) * 2 + 2
        count = min(len(ds), (args.smoke_steps + extra) * int(cfg.training.batch_size) * 4)
        indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:count].tolist()
    groups = {}
    for index in indices:
        groups.setdefault(index // identity["shard_size"], []).append(index)
    # Read headers only: completed windows are never decoded again except parity checks.
    existing, tasks, verify_indices = set(), [], set()
    local_groups = {s: w for s, w in sorted(groups.items()) if s % world == rank}
    verify_existing, verify_new = 0, 0
    for shard, wanted in local_groups.items():
        path = output / f"shard_{shard:06d}.safetensors"
        keys = set()
        if path.exists():
            with safe_open(path, framework="pt") as handle:
                keys = set(handle.keys())
        for index in wanted:
            cached = str(index) in keys
            if cached:
                existing.add(index)
            verify = (verify_existing if cached else verify_new) < args.verify_count
            if verify:
                verify_indices.add(index)
                if cached:
                    verify_existing += 1
                else:
                    verify_new += 1
            if not cached or verify:
                tasks.append(index)
    loader_options = dict(batch_size=None, num_workers=args.loader_workers)
    if args.loader_workers:
        loader_options.update(prefetch_factor=args.prefetch_factor, persistent_workers=True,
                              multiprocessing_context="spawn")
    loader = torch.utils.data.DataLoader(
        WindowPrefetchDataset(OmegaConf.to_container(cfg.dataloader), tasks), **loader_options)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.workers_per_gpu < 1 or world % args.workers_per_gpu:
        raise ValueError("WORLD_SIZE must be divisible by positive --workers-per-gpu")
    torch.cuda.set_device(local_rank // args.workers_per_gpu)
    encoder = WanVAEFeatureEncoder.from_pretrained(
        **{k: v for k, v in OmegaConf.to_container(cfg.model.video_backbone.encoder).items() if k != "name"}
    ).to("cuda")
    started = time.monotonic()
    made, verified, pending = 0, [], []
    current_shard, additions = None, {}
    wait_seconds, encode_seconds, write_wait_seconds = 0.0, 0.0, 0.0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-cache-write") as writer:
        def submit(shard, values):
            nonlocal write_wait_seconds
            if not values:
                return
            if len(pending) >= args.write_inflight:
                before = time.monotonic()
                pending.pop(0).result()  # Bounded memory and propagated writer failures.
                write_wait_seconds += time.monotonic() - before
            pending.append(writer.submit(write_shard, output / f"shard_{shard:06d}.safetensors", values))

        previous = time.monotonic()
        for index, video in loader:
            wait_seconds += time.monotonic() - previous
            shard, key = index // identity["shard_size"], str(index)
            if shard != current_shard:
                submit(current_shard, additions)
                current_shard, additions = shard, {}
            before = time.monotonic()
            z = encoder.batch_encode(encoder.preprocess_video(video))[0].cpu().contiguous()
            if list(z.shape) != identity["shape"] or z.dtype != torch.bfloat16 or not torch.isfinite(z).all():
                raise ValueError(f"Invalid {args.experiment} latent at index {index}: {z.shape}, {z.dtype}")
            encode_seconds += time.monotonic() - before
            if index not in existing:
                additions[key] = z
                made += 1
            if index in verify_indices:
                if index in existing:
                    with safe_open(output / f"shard_{shard:06d}.safetensors", framework="pt") as handle:
                        reference = handle.get_tensor(key)
                else:
                    # Decode synchronously as the original implementation did, independent of workers.
                    sample = ds._getitem_impl(index)
                    reference = encoder.batch_encode(encoder.preprocess_video(sample["video"]))[0].cpu()
                torch.testing.assert_close(z, reference, rtol=0, atol=0)
                verified.append(dict(index=index, max_abs_error=0.0, bitwise_equal=True,
                                     existing=index in existing))
            if made and made % 256 == 0:
                print(json.dumps(dict(rank=rank, encoded=made, seconds=time.monotonic()-started,
                                      loader_wait_seconds=wait_seconds, encode_seconds=encode_seconds,
                                      write_wait_seconds=write_wait_seconds, verified=verified)), flush=True)
            previous = time.monotonic()
        submit(current_shard, additions)
        for future in pending:
            future.result()  # Complete every write before announcing successful generation.
    result = dict(rank=rank, encoded=made, seconds=time.monotonic()-started, verified=verified,
                  workers_per_gpu=args.workers_per_gpu,
                  loader_workers=args.loader_workers, prefetch_factor=args.prefetch_factor,
                  write_inflight=args.write_inflight, loader_wait_seconds=wait_seconds,
                  encode_seconds=encode_seconds, write_wait_seconds=write_wait_seconds,
                  smoke_steps=args.smoke_steps, requested_windows=len(indices),
                  full_cache=args.smoke_steps is None and args.limit_windows is None)
    (output / f"build_rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
