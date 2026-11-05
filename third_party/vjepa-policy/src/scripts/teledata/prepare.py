"""Prepare quantile normalization and untruncated T5 caches for TeleData v3."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from transformers import AutoTokenizer, T5EncoderModel

from vjepa_policy.datasets.lerobot_video import DEFAULT_PROMPT
from vjepa_policy.datasets.lerobot_video_v3 import read_v3_tasks
from vjepa_policy.datasets.normalization import ActionStateNormalizer


def compute_quantile_stats(dataset_dir: Path, output_path: Path) -> None:
    with (dataset_dir / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    with (dataset_dir / "meta" / "stats.json").open(encoding="utf-8") as handle:
        source_stats = json.load(handle)
    data_paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No frame parquet files under {dataset_dir / 'data'}")

    stats = {}
    for key in ("action", "observation.state"):
        width = int(info["features"][key]["shape"][0])
        arrays = []
        for path in data_paths:
            values = pq.read_table(path, columns=[key])[key].combine_chunks()
            flat = pc.list_flatten(values).to_numpy(zero_copy_only=False)
            arrays.append(flat.reshape(-1, width))
        values = np.concatenate(arrays, axis=0)
        if values.shape != (int(info["total_frames"]), width):
            raise ValueError(f"Unexpected {key} values shape {values.shape}")
        quantiles = np.quantile(values, (0.01, 0.99), axis=0)
        stats[key] = {
            name: np.asarray(value)
            for name, value in source_stats[key].items()
        }
        stats[key].setdefault("q01", stats[key]["min"])
        stats[key].setdefault("q99", stats[key]["max"])
        stats[key]["q01"] = quantiles[0]
        stats[key]["q99"] = quantiles[1]

    normalizer = ActionStateNormalizer(
        features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(int(info["features"]["action"]["shape"][0]),),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(int(info["features"]["observation.state"]["shape"][0]),),
            ),
        },
        stats=stats,
        action_mode="QUANTILE",
        state_mode="QUANTILE",
        clip_values=False,
        metadata={
            "action_codec": {
                "action_indices": list(range(int(info["features"]["action"]["shape"][0]))),
                "state_indices": list(range(int(info["features"]["observation.state"]["shape"][0]))),
                "relative_action_indices": [],
                "action_chunk_size": 32,
            },
            "proprio_encoding": {
                "type": "identity",
                "raw_dim": int(info["features"]["observation.state"]["shape"][0]),
                "encoded_dim": int(info["features"]["observation.state"]["shape"][0]),
            },
            "source_codebase_version": info["codebase_version"],
            "source_total_frames": int(info["total_frames"]),
        },
    )
    normalizer.save(output_path)


def cache_t5(
    dataset_dir: Path,
    cache_dir: Path,
    model_name: str,
    context_length: int,
    device: str,
) -> None:
    tasks = read_v3_tasks(dataset_dir)
    prompts = sorted({DEFAULT_PROMPT.format(task=task) for task in tasks.values()})
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    pending = []
    for prompt in prompts:
        tokens = tokenizer(prompt, padding=False, truncation=False, return_tensors="pt")
        prompt_length = int(tokens.input_ids.shape[1])
        if prompt_length > context_length:
            raise ValueError(
                f"Prompt requires {prompt_length} tokens, context_length={context_length}: "
                f"{prompt!r}"
            )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output_path = cache_dir / f"{digest}.t5_len{context_length}.pt"
        if not output_path.is_file():
            pending.append((prompt, output_path))
    if not pending:
        return

    encoder = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    for prompt, output_path in pending:
        tokens = tokenizer(
            prompt,
            max_length=context_length,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            context = encoder(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            ).last_hidden_state[0].float().cpu()
        torch.save(
            {"context": context, "mask": tokens.attention_mask[0].bool()},
            output_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--stats-output", required=True, type=Path)
    parser.add_argument("--text-cache-dir", required=True, type=Path)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.stats_output.is_file():
        compute_quantile_stats(args.dataset_dir, args.stats_output)
    cache_t5(
        args.dataset_dir,
        args.text_cache_dir,
        args.text_model,
        args.context_length,
        args.device,
    )


if __name__ == "__main__":
    main()
