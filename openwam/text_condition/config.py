"""Shared training, precomputation and evaluation text-cache contract."""
from pathlib import Path

TEXT_ENCODERS = ("umt5-xxl", "cosmos3-nano", "qwen3p5-4b", "cosmos3-edge",
                 "rynnbrain1p1-2b", "rynnbrain1p1-9b")
EMBEDDING_METHODS = ("concat_all_layers", "last_layer", "avg_all_layers")
MAX_TOKENS = 128
# Architecture metadata, checked against the actual HF config during encoding.
GEOMETRY = {
    "cosmos3-edge": (2048, 28),
    "umt5-xxl": (4096, 1),
    "cosmos3-nano": (4096, 36),
    "qwen3p5-4b": (2560, 32),
    "rynnbrain1p1-2b": (2048, 24),
    "rynnbrain1p1-9b": (4096, 32),
}
CACHE_NAMES = {
    "cosmos3-edge": "cosmos3-edge-embedding",
    "umt5-xxl": "umt5-embedding",
    "cosmos3-nano": "cosmos3-embedding",
    "qwen3p5-4b": "qwen3p5-embedding",
    "rynnbrain1p1-2b": "rynnbrain1p1-2b-embedding",
    "rynnbrain1p1-9b": "rynnbrain1p1-9b-embedding",
}
MODEL_PATHS = {
    "cosmos3-edge": "/team_data/fcy/model/cosmos3-edge",
    "cosmos3-nano": "/team_data/fcy/model/cosmos3-nano-reasoner",
    "qwen3p5-4b": "/team_data/rhodes_team/huggingface_cache/hub/models--Qwen--Qwen3.5-4B",
    "rynnbrain1p1-2b": "/team_data/fcy/model/RynnBrain1_1-2B",
    "rynnbrain1p1-9b": "/team_data/fcy/model/RynnBrain1_1-9B",
}


def text_spec(encoder_type="umt5-xxl", method="avg_all_layers"):
    if encoder_type not in TEXT_ENCODERS:
        raise ValueError(f"Unknown text_encoder_type: {encoder_type!r}; choose {TEXT_ENCODERS}")
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"Unknown vlm_text_embedding_method: {method!r}; choose {EMBEDDING_METHODS}")
    width, layers = GEOMETRY[encoder_type]
    return {
        "text_encoder_type": encoder_type,
        "vlm_text_embedding_method": method if encoder_type != "umt5-xxl" else None,
        "shape": [MAX_TOKENS, width * layers if encoder_type != "umt5-xxl" and method == "concat_all_layers" else width],
    }


def cache_directory(encoder_type="umt5-xxl", method="avg_all_layers", benchmark="libero", *, data_root="data", corrected=False):
    text_spec(encoder_type, method)
    if benchmark not in ("libero", "libero-plus"):
        raise ValueError(f"Unknown benchmark: {benchmark}")
    path = Path(data_root) / CACHE_NAMES[encoder_type] / benchmark
    if encoder_type != "umt5-xxl":
        path /= method
    if benchmark == "libero-plus":
        path /= "corrected_text" if corrected else "original_text"
    return path


def cache_from_config(cfg):
    """Missing fields identify legacy UMT5 configs; explicit dirs remain supported."""
    from openwam.model.video_backbone.wan.text_embedding_cache import TextEmbeddingCache

    kind = cfg.get("text_encoder_type", "umt5-xxl")
    method = cfg.get("vlm_text_embedding_method", "avg_all_layers")
    text_spec(kind, method)
    directory = cfg.get("text_embedding_dir")
    if not directory and "text_encoder_type" not in cfg:
        return None  # historical online T5 configuration
    directory = directory or cache_directory(kind, method, data_root=Path(__file__).resolve().parents[2] / "data")
    return TextEmbeddingCache(directory, encoder_type=kind, method=method,
                              fingerprint=cfg.get("text_embedding_fingerprint"),
                              max_tokens=MAX_TOKENS if "text_encoder_type" in cfg else None)


def adapt_text_projection(holder, cfg):
    """Only the first raw-text Linear changes width; all DiT blocks stay intact."""
    import torch.nn as nn

    cache = holder.text_embedding_cache
    if cache is None:
        return
    width = cache.shape[1]
    for name in ("dit", "dit2"):
        dit = getattr(holder, name, None)
        if dit is None:
            continue
        projection = dit.text_embedding[0]
        if projection.in_features == width:
            continue
        if cfg.get("components"):
            raise ValueError("Saved DiT text input width disagrees with selected text cache")
        if not cfg.get("from_scratch", False):
            raise ValueError("Changing raw text width requires from_scratch=true or a matching trained checkpoint")
        dit.text_embedding[0] = nn.Linear(width, projection.out_features,
                                         bias=projection.bias is not None,
                                         device=projection.weight.device, dtype=projection.weight.dtype)


def save_text_cache_config(cfg, cache):
    """Pin encoding identity without saving/loading the encoder in a policy checkpoint."""
    if cache is None:
        return
    from omegaconf import open_dict

    with open_dict(cfg):
        vb = cfg.model.video_backbone
        vb.text_encoder_type = cache.encoder_type
        vb.vlm_text_embedding_method = cache.method
        vb.text_embedding_fingerprint = cache.fingerprint
