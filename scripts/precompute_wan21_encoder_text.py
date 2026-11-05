"""Generate native UMT5 empty-text conditioning once for the frozen Wan encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from openwam.model.video_backbone.encoder.wan21_features import load_weights
from openwam.model.video_backbone.wan.models.text_encoder import WanTextEncoder, HuggingfaceTokenizer
from openwam.model.video_backbone.wan.encode import encode_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(__file__).resolve().parents[1] / "encoder/wan21-i2v-14b-480p")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.model_path / "empty_context.safetensors"
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing encoder conditioning: {output}")
    with init_empty_weights():
        model = WanTextEncoder()
    load_weights(model, [args.model_path / "models_t5_umt5-xxl-enc-bf16.pth"])
    model.eval().requires_grad_(False).to(args.device)
    tokenizer = HuggingfaceTokenizer(str(args.model_path / "google/umt5-xxl"), seq_len=512, clean="whitespace")
    with torch.no_grad():
        context, lengths = encode_text([""], tokenizer=tokenizer, text_encoder=model, device=args.device)
    if not torch.isfinite(context).all():
        raise ValueError("Non-finite native empty-text context")
    source = args.model_path / "models_t5_umt5-xxl-enc-bf16.pth"
    identity = {"prompt": "", "encoder": "native_wan21_umt5", "padding": "zero_after_valid_tokens",
                "source_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns}
    temporary = output.with_suffix(".tmp")
    save_file({"context": context.cpu().contiguous(), "seq_lens": lengths.cpu()}, str(temporary), metadata={"identity": json.dumps(identity, sort_keys=True)})
    temporary.replace(output)
    print(json.dumps({"path": str(output), "shape": list(context.shape), "seq_lens": lengths.tolist(),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
