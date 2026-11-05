"""Isolated A/B VAE microbenchmark. Analysis only; never changes experiment view layout."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.model.video_backbone.encoder.wan_vae import WanVAEFeatureEncoder
from openwam.train.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="reports/E03_speed_optimization/views.json")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    seed_everything(42)
    cfg, _, _ = resolve_experiment("E03")
    ds = build_dataset(cfg.dataloader)
    sample = ds._getitem_impl(0)
    enc = WanVAEFeatureEncoder.from_pretrained(cfg.model.video_backbone.encoder.model_path, vae_version="2.1").cuda()
    video = enc.preprocess_video(sample["video"])
    methods = {"A_independent": lambda: enc.batch_encode(video),
               "B_rgb_stitch": lambda: enc._m.batch_encode(video, device=video.device)}
    outputs, result = {}, {}
    with torch.no_grad():
        for name, encode in methods.items():
            for _ in range(2):
                encode()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(5):
                start = time.perf_counter()
                z = encode()
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
            outputs[name] = z.cpu()
            result[name] = dict(input_shape=list(video.shape), latent_shape=list(z.shape),
                                tokens=z.shape[2]*(z.shape[3]//2)*(z.shape[4]//2),
                                seconds=sum(durations)/len(durations),
                                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    result["max_abs_representation_difference"] = float((outputs["A_independent"].float()-outputs["B_rgb_stitch"].float()).abs().max())
    result["changes_training_config"] = False
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
