"""Bounded CUDA/batch64 parity probe before allocating a full image cache.

Uses the real builder's per-timestamp preprocessing and batch size. This does
not certify full-cache coverage and never writes production validation.json.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.image_frame_cache import window_frame_indices
from openwam.model.video_backbone.encoder.registry import build_video_encoder
from scripts.precompute_flux2_features import CurrentStereoFrames, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E01", "E02"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise ValueError("Positive offline batch size and CUDA are required")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg, _, _ = resolve_experiment(args.experiment + "-Cache")
    cfg.dataloader.frame_feature_cache_dir = None
    reader = build_dataset(cfg.dataloader, split="train")
    frames = CurrentStereoFrames(reader)
    encoder = build_video_encoder(cfg.model.video_backbone.encoder).to("cuda", dtype=torch.bfloat16)
    encoder.requires_grad_(False).eval()
    checks = []
    started = time.monotonic()
    with torch.no_grad():
        for index in (0, 64, len(reader) - 1):
            sample = reader._getitem_impl(index)
            episode = int(np.searchsorted(reader._cum_n_starts, index, side="right") - 1)
            offset = index - int(reader._cum_n_starts[episode])
            keys = window_frame_indices(reader, episode, offset)
            clip = [sample["clean_condition_images"][0], *sample["video"]]
            unique = list(dict.fromkeys(keys))
            features = {}
            for start in range(0, len(unique), args.batch_size):
                chunk = unique[start:start + args.batch_size]
                images = [Image.fromarray(frames[key]["pixels"].permute(1, 2, 0).numpy()) for key in chunk]
                for key, image in zip(chunk, images):
                    assert np.array_equal(np.asarray(image), np.asarray(clip[keys.index(key)]))
                pixels = torch.cat([encoder.preprocess_video([image]) for image in images])
                raw = encoder._encode_frames(pixels).squeeze(2).cpu()
                features.update({key: value.clone() for key, value in zip(chunk, raw)})
                del pixels, raw
            frozen = torch.stack([features[key] for key in keys], dim=1)
            pixels = encoder.preprocess_video(clip).expand(64, -1, -1, -1, -1).contiguous()
            torch.cuda.synchronize()
            stamp = time.monotonic()
            online = encoder._encode_frames(pixels)[0].cpu()
            seconds = time.monotonic() - stamp
            difference = (online.float() - frozen.float()).abs()
            normalize = encoder._apply_feature_norm if args.experiment == "E01" else encoder._normalize
            online_norm = normalize(online.unsqueeze(0).cuda()).cpu()
            frozen_norm = normalize(frozen.unsqueeze(0).cuda()).cpu()
            row = dict(index=index, pixel_equal=True, bitwise_equal=torch.equal(online, frozen),
                       normalized_bitwise_equal=torch.equal(online_norm, frozen_norm),
                       max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                       mean_cosine=float(torch.nn.functional.cosine_similarity(
                           online.float(), frozen.float(), dim=0).mean()),
                       normalized_max_abs=float((online_norm.float() - frozen_norm.float()).abs().max()),
                       online_encoder_seconds=seconds)
            checks.append(row)
            print(json.dumps(row), flush=True)
            del pixels, online, frozen, online_norm, frozen_norm
    report = dict(passed=all(c["bitwise_equal"] and c["normalized_bitwise_equal"] for c in checks),
                  experiment=args.experiment, online_batch=64, offline_batch=args.batch_size,
                  checks=checks, seconds=time.monotonic() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  torch_version=str(torch.__version__), formal_cache_acceptance=False)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Offline batching differs from online BF16 encoding; inspect probe before full generation")


if __name__ == "__main__":
    main()
