"""CPU-resident frozen text features, indexed by exact prompt text/hash."""
from collections import OrderedDict
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


class TextEmbeddingCache:
    def __init__(self, directory, *, encoder_type="umt5-xxl", method="avg_all_layers", max_tokens=None, fingerprint=None):
        from openwam.text_condition.config import text_spec

        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        expected = text_spec(encoder_type, method)
        self.encoder_type, self.method = encoder_type, method
        self.fingerprint = manifest.get("model_fingerprint")
        if fingerprint is not None and self.fingerprint != fingerprint:
            raise ValueError("Text cache model fingerprint differs from training checkpoint")
        self.version = manifest.get("version")
        self.shape = tuple(manifest.get("shape", ()))
        if self.version == 1:
            if encoder_type != "umt5-xxl" or self.shape != (512, 4096):
                raise ValueError(f"Unsupported legacy text embedding manifest: {self.directory}")
        elif self.version == 2:
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError(f"Text cache {key} mismatch: {manifest.get(key)!r} != {value!r}")
            if manifest.get("padding_side") != "right":
                raise ValueError("Text cache requires right padding")
        else:
            raise ValueError(f"Unsupported text embedding manifest: {self.directory}")
        self.max_tokens = max_tokens or self.shape[0]
        self.files = manifest["prompts"]
        self._loaded = OrderedDict()
        # Concat caches can be tens of MB per prompt. Bound per-rank CPU memory.
        self._capacity = 64 if encoder_type != "umt5-xxl" else None

    def encode(self, prompts, *, device):
        contexts, lengths = [], []
        for prompt in prompts:
            if prompt not in self.files:
                raise KeyError(f"Text embedding cache has no prompt {prompt!r}; regenerate {self.directory}")
            if prompt not in self._loaded:
                tensors = load_file(str(self.directory / self.files[prompt]))
                context, length = tensors["context"], tensors["seq_len"]
                if tuple(context.shape) != self.shape or context.dtype != torch.bfloat16:
                    raise ValueError(f"Invalid cached context for {prompt!r}")
                if length.numel() != 1 or not 0 < length.item() <= self.shape[0]:
                    raise ValueError(f"Invalid cached sequence length for {prompt!r}")
                if self.version == 2:
                    mask = tensors["attention_mask"]
                    expected_mask = torch.arange(self.shape[0]) < length.item()
                    if mask.shape != expected_mask.shape or not torch.equal(mask.bool(), expected_mask):
                        raise ValueError(f"Cached attention mask is not a right-padded prefix: {prompt!r}")
                context = context[:self.max_tokens].contiguous()
                length = length.clamp(max=self.max_tokens)
                self._loaded[prompt] = (context, length)
                if self._capacity and len(self._loaded) > self._capacity:
                    self._loaded.popitem(last=False)
            self._loaded.move_to_end(prompt)
            context, length = self._loaded[prompt]
            contexts.append(context)
            lengths.append(length)
        # Existing attention builds the exact prefix mask from these lengths.
        return torch.stack(contexts).to(device), torch.stack(lengths).to(device=device, dtype=torch.long)
