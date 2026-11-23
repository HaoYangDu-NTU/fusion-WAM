"""Bounded real-weight stereo cache probe; never builds the full dataset cache."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.model.video_backbone.encoder.vggt_omega import VGGTOmegaSpatialEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "reports/E05_cache_probe"))
    parser.add_argument("--sample-index", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg, _, _ = resolve_experiment("E05")
    dataset = build_dataset(cfg.dataloader, split="train")
    config = dict(cfg.model.video_backbone.encoder)
    config.pop("name")
    model_path = config.pop("model_path")
    encoder = VGGTOmegaSpatialEncoder.from_pretrained(model_path, **config)
    rows = [dataset._getitem_impl(args.sample_index + offset) for offset in (0, 4)]
    started = time.monotonic()
    pixels = [encoder.preprocess_video([row["clean_condition_images"][0], *row["video"]]) for row in rows]
    preprocess = time.monotonic() - started
    started = time.monotonic()
    online = [encoder.batch_encode(value) for value in pixels]
    online_time = time.monotonic() - started
    encoder.configure_feature_cache(args.cache_dir, Path(model_path) / config["checkpoint"])
    started = time.monotonic()
    cached = [encoder.batch_encode(value) for value in pixels]
    cold = time.monotonic() - started
    cold_hits, cold_misses = encoder.feature_cache.hits, encoder.feature_cache.misses
    started = time.monotonic()
    warm = [encoder.batch_encode(value) for value in pixels]
    warm_time = time.monotonic() - started
    for expected, actual, reread in zip(online, cached, warm):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(expected, reread, rtol=0, atol=0)
    # Check actual formal-width spatial projection, not only the tiny smoke one.
    projection = encoder.build_dit_input_proj(1536).to(torch.bfloat16)
    started = time.monotonic()
    with torch.no_grad():
        projected = projection(online[0], num_clean_condition_latents=2)
    projection_seconds = time.monotonic() - started
    assert projected.shape == (1, 1536, 10, 7, 14)
    cache_files = list(encoder.feature_cache.directory.rglob("*.safetensors"))
    stored = sum(path.stat().st_size for path in cache_files)
    per_timestamp = online[0][:, :, :1].numel() * online[0].element_size()
    episode_frames = int(dataset._eps_df["length"].sum())
    report = dict(passed=True, device="cpu", threads=4, real_weights=True,
        windows=2, timestamp_requests=20, unique_cached_timestamps=len(cache_files),
        cold_hits=cold_hits, cold_misses=cold_misses, max_abs_difference=0.0,
        rgb_preprocess_seconds=preprocess, online_seconds=online_time,
        cache_fill_seconds=cold, warm_read_seconds=warm_time,
        warm_read_effective_mib_per_second=(20*per_timestamp/2**20)/warm_time,
        cache_bytes=stored, bytes_per_stereo_timestamp=per_timestamp,
        dataset_windows=len(dataset), dataset_episode_frames=episode_frames,
        full_cache_dedup_frame_upper_gib=episode_frames*per_timestamp/2**30,
        naive_per_window_cache_gib=len(dataset)*10*per_timestamp/2**30,
        full_projection_shape=list(projected.shape), tokens=980,
        full_width_projection_seconds=projection_seconds,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        limitations=["CPU timing; full GPU M29 not measured", "Content keys preserve exact input semantics",
            "This cache path still decodes RGB to establish content identity",
            "Capacity estimate assumes deterministic preprocessing; augmentations create extra keys",
            "Official-resolution end-to-end data/eval pipeline not validated"])
    (ROOT / "reports/E05_cache_profile_cpu.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
