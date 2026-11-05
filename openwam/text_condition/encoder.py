"""HF VLMs used only by offline cache generation, with text-only inputs."""
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import EMBEDDING_METHODS, GEOMETRY, MAX_TOKENS, MODEL_PATHS, text_spec

SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."


def resolve_hf_path(path):
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    ref = path / "refs/main"
    if ref.is_file():
        snapshot = path / "snapshots" / ref.read_text().strip()
        if (snapshot / "config.json").is_file():
            return snapshot
    raise FileNotFoundError(f"No Hugging Face config.json or refs/main snapshot in {path}")


def select_hidden_states(hidden_states, method):
    """Exclude input embeddings (entry 0); normalize each decoder layer independently."""
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"Unknown VLM embedding method: {method}")
    if method == "last_layer":
        return hidden_states[-1]
    layers = hidden_states[1:]
    if not layers:
        raise ValueError("VLM did not return decoder hidden states")
    normalized = (F.layer_norm(h.float(), (h.shape[-1],), eps=1e-5) for h in layers)
    if method == "concat_all_layers":
        return torch.cat([h.to(hidden_states[-1].dtype) for h in normalized], dim=-1)
    total = next(normalized)
    for state in normalized:
        total = total + state
    return (total / len(layers)).to(hidden_states[-1].dtype)


class VLMTextEncoder:
    """One frozen interface for Cosmos3 and Qwen-family VLMs, including RynnBrain."""
    def __init__(self, encoder_type, *, model_path=None, device="cuda"):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration, Cosmos3EdgeForConditionalGeneration

        if encoder_type not in MODEL_PATHS:
            raise ValueError(f"Not a supported VLM text encoder: {encoder_type}")
        self.encoder_type = encoder_type
        self.model_path = resolve_hf_path(model_path or MODEL_PATHS[encoder_type])
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        cls = {"cosmos3-nano": Qwen3VLForConditionalGeneration,
               "qwen3p5-4b": Qwen3_5ForConditionalGeneration,
               "rynnbrain1p1-2b": Qwen3_5ForConditionalGeneration,
               "rynnbrain1p1-9b": Qwen3_5ForConditionalGeneration,
               "cosmos3-edge": Cosmos3EdgeForConditionalGeneration}[encoder_type]
        model, info = cls.from_pretrained(self.model_path, dtype=torch.bfloat16,
                                          attn_implementation="sdpa", local_files_only=True,
                                          output_loading_info=True)
        if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise ValueError(f"Incomplete/incompatible HF text encoder checkpoint: {info}")
        config = model.config.text_config
        if (config.hidden_size, config.num_hidden_layers) != GEOMETRY[encoder_type]:
            raise ValueError(f"Unexpected {encoder_type} text geometry")
        self.num_layers = config.num_hidden_layers
        # HF's base multimodal model returns hidden states without allocating LM logits.
        # Vision is never called: only input_ids and attention_mask are supplied.
        self.model = model.model.eval().requires_grad_(False).to(self.device)

    def tokenize(self, prompts):
        conversations = [[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}] for prompt in prompts]
        rendered = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        return self.tokenizer(rendered, add_special_tokens=False, padding="max_length",
                              truncation=True, max_length=MAX_TOKENS, return_tensors="pt")

    @torch.inference_mode()
    def encode(self, prompts, methods=("avg_all_layers",)):
        inputs = self.tokenize(prompts)
        mask = inputs["attention_mask"].to(self.device)
        output = self.model(input_ids=inputs["input_ids"].to(self.device), attention_mask=mask,
                            output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = output.hidden_states
        if hidden is None or len(hidden) != self.num_layers + 1:
            raise ValueError("HF must return the input embedding plus every decoder layer")
        result = {}
        for method in methods:
            context = select_hidden_states(hidden, method).to(torch.bfloat16)
            context = context.masked_fill(~mask.bool().unsqueeze(-1), 0)
            expected = (len(prompts), *text_spec(self.encoder_type, method)["shape"])
            if tuple(context.shape) != expected or not torch.isfinite(context).all():
                raise ValueError(f"Invalid {method} text embedding; expected {expected}")
            result[method] = context.cpu()
        return result, mask.bool().cpu()
