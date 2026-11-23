"""Certify E01/E02 cache only after exact CUDA BF16 production-batch comparison.

Run after full cache generation on idle GPUs. A failed check records diagnostics
and prevents formal cached training; online experiments remain available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.image_frame_cache import CachedImageDataset, ImageFrameCache, ImageFrameCacheCollator
from openwam.dataloader.spatial_frame_cache_specs import frame_cache_identity
from openwam.model.video_backbone.encoder.registry import build_video_encoder
from scripts.feature_cache.frames import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=("E01","E02","E07","E08"),required=True)
    parser.add_argument("--cache-dir")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Production cache acceptance requires CUDA")
    torch.set_num_threads(1)
    cfg,_,_=resolve_experiment(args.experiment+"-Cache")
    directory=Path(args.cache_dir or cfg.dataloader.frame_feature_cache_dir)
    cfg.dataloader.frame_feature_cache_dir=None
    reader=build_dataset(cfg.dataloader,split="train")
    identity=frame_cache_identity(cfg.dataloader,cfg.model.video_backbone.encoder,reader)
    cache=ImageFrameCache(directory,identity)
    cache.require_complete()
    metadata=CachedImageDataset(reader,cache)
    collate=ImageFrameCacheCollator(directory)
    encoder=build_video_encoder(cfg.model.video_backbone.encoder).to(device="cuda",dtype=torch.bfloat16)
    encoder.requires_grad_(False).eval()
    checks=[]
    with torch.no_grad():
        for index in (0,64,len(reader)-1):
            sample=reader._getitem_impl(index)
            clip=[sample["clean_condition_images"][0],*sample["video"]]
            pixels=encoder.preprocess_video(clip).expand(64,-1,-1,-1,-1).contiguous()
            online=encoder._encode_frames(pixels)[0].cpu()
            frozen=collate([metadata[index]])[0]["image_frame_features"]
            diff=(online.float()-frozen.float()).abs()
            checks.append(dict(index=index,bitwise_equal=torch.equal(online,frozen),max_abs=float(diff.max()),
                               mean_abs=float(diff.mean()),
                               mean_cosine=float(torch.nn.functional.cosine_similarity(online.float(),frozen.float(),dim=0).mean())))
            del pixels,online,frozen
    report=dict(passed=all(row["bitwise_equal"] for row in checks),device="cuda",online_batch_size=64,
                identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),checks=checks)
    atomic_json(directory/"validation.json",report)
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise RuntimeError("Cache changes BF16 numerical values at production batch size; keep online training")


if __name__=="__main__":
    main()
