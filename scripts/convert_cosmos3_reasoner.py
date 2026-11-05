"""Extract a strict HF Qwen3-VL reasoner from Cosmos3-Nano diffusers safetensors.

Mapping follows NVIDIA/cosmos-framework's convert_model_to_vlm_safetensors.py
and inference/model.py. No Cosmos source checkout, generator model, or base
Qwen weights are needed. The base Nano checkpoint supplies the full vision tower.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil


def reasoner_key(key, shard):
    if shard.startswith("vision_encoder/"):
        return "model.visual." + key
    if not shard.startswith("transformer/") or "_moe_gen" in key or "k_norm_und_for_gen" in key:
        return None
    if any(part in key for part in (".add_q_proj.", ".add_k_proj.", ".add_v_proj.", ".to_add_out.", ".norm_added_")):
        return None
    if key == "lm_head.weight":
        return key
    if not re.match(r"^(embed_tokens\.|layers\.|norm\.)", key):
        return None
    for old, new in (("to_q", "q_proj"), ("to_k", "k_proj"), ("to_v", "v_proj"),
                     ("to_out", "o_proj"), ("norm_q", "q_norm"), ("norm_k", "k_norm")):
        key = key.replace(f".self_attn.{old}.", f".self_attn.{new}.")
    return "model.language_model." + key


def convert(source, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing conversion: {output}")
    raw = json.loads((source / "config.json").read_text())
    config = Qwen3VLConfig(**{k: raw[k] for k in ("text_config", "vision_config", "image_token_id", "video_token_id",
                                                "vision_start_token_id", "vision_end_token_id", "tie_word_embeddings")})
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    del model
    index_path = source / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    groups, mapped = defaultdict(list), {}
    for key, shard in weight_map.items():
        target = reasoner_key(key, shard)
        if target is None:
            continue
        if target not in expected or target in mapped:
            raise ValueError(f"Unexpected/duplicate reasoner tensor: {key} -> {target}")
        mapped[target] = key
        groups[shard].append((key, target))
    missing = set(expected) - set(mapped)
    if missing:
        raise ValueError(f"Incomplete reasoner checkpoint: {sorted(missing)[:20]}")
    temporary = output.with_name(output.name + ".converting")
    temporary.mkdir(parents=True, exist_ok=False)
    output_index, total_size = {}, 0
    for i, (shard, entries) in enumerate(sorted(groups.items()), 1):
        tensors = {}
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            for key, target in entries:
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != expected[target]:
                    raise ValueError(f"Shape mismatch for {target}")
                tensors[target] = tensor.to(torch.bfloat16).contiguous()
                total_size += tensors[target].numel() * tensors[target].element_size()
        filename = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        save_file(tensors, str(temporary / filename), metadata={"format": "pt"})
        output_index.update({key: filename for key in tensors})
        print(f"Saved {filename}: {len(tensors)} tensors", flush=True)
        del tensors
    config.save_pretrained(temporary)
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "chat_template.json",
                 "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, temporary / name)
    (temporary / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_size}, "weight_map": output_index}, indent=2))
    (temporary / "conversion.json").write_text(json.dumps({"source": str(source), "tensor_count": len(expected),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "missing_keys": [], "unexpected_keys": [], "generator_excluded": True}, indent=2))
    temporary.rename(output)
    print(f"Converted reasoner: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/team_data/fcy/model/cosmos3-nano")
    parser.add_argument("--output", default="/team_data/fcy/model/cosmos3-nano-reasoner")
    args = parser.parse_args()
    convert(args.source, args.output)
