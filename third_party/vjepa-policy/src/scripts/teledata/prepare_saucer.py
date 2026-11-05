"""Prepare stats and untruncated T5 caches for the Tianji saucer dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from transformers import AutoTokenizer, T5EncoderModel

from vjepa_policy.datasets.lerobot_video import DEFAULT_PROMPT
from vjepa_policy.datasets.lerobot_video_teledata import read_teledata_tasks
from vjepa_policy.datasets.normalization import ActionStateNormalizer


def write_stats(dataset_dir: Path, output_path: Path) -> None:
    with (dataset_dir / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    with (dataset_dir / "meta" / "norm_stats.json").open(encoding="utf-8") as handle:
        source = json.load(handle)
    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(16,)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
    }
    stats = {
        key: {name: np.asarray(value) for name, value in values.items()}
        for key, values in source.items()
    }
    normalizer = ActionStateNormalizer(
        features=features,
        stats=stats,
        action_mode="QUANTILE",
        state_mode="QUANTILE",
        clip_values=False,
        metadata={
            "action_codec": {
                "action_indices": list(range(16)),
                "state_indices": list(range(16)),
                "relative_action_indices": [],
                "action_chunk_size": 32,
            },
            "proprio_encoding": {
                "type": "identity",
                "raw_dim": 16,
                "encoded_dim": 16,
            },
            "source_codebase_version": info.get("codebase_version"),
            "source_total_frames": int(info["total_frames"]),
            "source_total_episodes": int(info["total_episodes"]),
        },
    )
    normalizer.save(output_path)


def write_t5_cache(
    dataset_dir: Path,
    cache_dir: Path,
    model_name: str,
    context_length: int,
    device: str,
) -> None:
    tasks = read_teledata_tasks(dataset_dir)
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.stats_output.is_file():
        write_stats(args.dataset_dir, args.stats_output)
    write_t5_cache(
        args.dataset_dir,
        args.text_cache_dir,
        args.text_model,
        args.context_length,
        args.device,
    )


if __name__ == "__main__":
    main()
