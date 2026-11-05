"""Atomic, resumable offline cache writer shared by LIBERO and LIBERO-Plus."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from .config import EMBEDDING_METHODS, MAX_TOKENS, MODEL_PATHS, cache_directory, text_spec


def model_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json", "chat_template.jinja",
                 "model.safetensors.index.json", "conversion.json"):
        file = path / name
        if file.is_file():
            digest.update(name.encode())
            digest.update(file.read_bytes())
    shards = set(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards.update(path / name for name in json.loads(index.read_text())["weight_map"].values())
    for file in sorted(shards):
        stat = file.stat()
        digest.update(f"{file.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def add_encoder_arguments(parser):
    from .config import TEXT_ENCODERS

    parser.add_argument("--text-encoder-type", choices=TEXT_ENCODERS, default="umt5-xxl")
    parser.add_argument("--vlm-text-embedding-method", choices=(*EMBEDDING_METHODS, "all"), default="avg_all_layers")
    parser.add_argument("--model-path", type=Path, help="HF VLM directory (default: encoder-specific shared weights)")
    parser.add_argument("--text-encoder", help="Native UMT5 safetensors file")
    parser.add_argument("--tokenizer", help="Native UMT5 tokenizer directory")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output-dir", type=Path, help="Explicit cache directory; with all methods, their parent")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)


def write_caches(catalogs, args, benchmark):
    """catalogs maps mode ('' / original_text / corrected_text) to prompt lists."""
    import fcntl
    from .encoder import SYSTEM_PROMPT, VLMTextEncoder, resolve_hf_path

    if args.batch_size < 1 or args.sleep_seconds < 0:
        raise ValueError("batch-size must be positive and sleep-seconds nonnegative")
    kind = args.text_encoder_type
    methods = list(EMBEDDING_METHODS) if args.vlm_text_embedding_method == "all" else [args.vlm_text_embedding_method]
    if kind == "umt5-xxl":
        methods = ["avg_all_layers"]  # VLM aggregation is deliberately ignored for UMT5.
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = args.device
    if device == "cuda":
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    if world > 1:
        torch.distributed.init_process_group("gloo")
    torch.set_num_threads(2)
    if kind == "umt5-xxl":
        if not args.text_encoder or not args.tokenizer:
            raise ValueError("UMT5 requires --text-encoder and --tokenizer")
        stat = Path(args.text_encoder).stat()
        fingerprint = hashlib.sha256(f"{Path(args.text_encoder).resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        source = str(Path(args.text_encoder).resolve())
    else:
        source = str(resolve_hf_path(args.model_path or MODEL_PATHS[kind]))
        fingerprint = model_fingerprint(source)
    jobs, locks = [], []
    for method in methods:
        for mode, prompts in catalogs.items():
            prompts = sorted(set(prompts))
            if not prompts or any(not prompt.strip() for prompt in prompts):
                raise ValueError("Empty text catalog/prompt")
            if args.output_dir:
                directory = args.output_dir
                if args.vlm_text_embedding_method == "all" and kind != "umt5-xxl":
                    directory /= method
                if mode:
                    directory /= mode
            else:
                directory = cache_directory(kind, method, benchmark, data_root=args.data_root,
                                            corrected=mode == "corrected_text")
            directory.mkdir(parents=True, exist_ok=True)
            lock = (directory / f".encode-rank{rank}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
            identity = {"version": 2, **text_spec(kind, method), "padding_side": "right", "dtype": "bfloat16",
                        "model_fingerprint": fingerprint, "model_path": source,
                        "prompt_format": "cosmos-predict25-chat-v1" if kind != "umt5-xxl" else "wan-whitespace-v1",
                        "system_prompt": SYSTEM_PROMPT if kind != "umt5-xxl" else None,
                        "layer_norm_eps": 1e-5 if kind != "umt5-xxl" else None,
                        "correct_instruction_suffix": mode == "corrected_text" if mode else None}
            identity_path = directory / "encoding.json"
            if identity_path.is_file():
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError(f"Refusing to mix encoders/revisions in {directory}")
            elif (directory / "manifest.json").exists() or any(directory.glob("*.safetensors")):
                raise ValueError(f"Refusing to overwrite an unversioned/legacy cache: {directory}")
            elif rank == 0:
                identity_path.write_text(json.dumps(identity, indent=2))
            files = {text: hashlib.sha256(text.encode()).hexdigest() + ".safetensors" for text in prompts}
            jobs.append((method, directory, identity, files))
    if world > 1:
        torch.distributed.barrier()
    all_prompts = sorted(set().union(*(set(job[3]) for job in jobs)))
    pending = []
    for prompt in all_prompts[rank::world]:
        needed = []
        for method, directory, identity, files in jobs:
            if prompt not in files:
                continue
            file = directory / files[prompt]
            if file.exists():
                tensors = load_file(str(file))
                context, mask, length = tensors["context"], tensors["attention_mask"], tensors["seq_len"]
                if (list(context.shape) != identity["shape"] or context.dtype != torch.bfloat16
                        or mask.shape != (MAX_TOKENS,) or length.numel() != 1
                        or not 0 < length.item() <= MAX_TOKENS
                        or not torch.equal(mask.bool(), torch.arange(MAX_TOKENS) < length.item())
                        or not torch.isfinite(context).all()):
                    raise ValueError(f"Invalid existing cache tensor: {file}")
            else:
                needed.append((method, file))
        if needed:
            pending.append((prompt, needed))
    print(json.dumps({"rank": rank, "encoder": kind, "pending": len(pending), "catalog_prompts": len(all_prompts)}), flush=True)
    encoder = tokenizer = None
    if pending:
        if kind == "umt5-xxl":
            from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer, WanTextEncoder
            with torch.device("meta"):
                encoder = WanTextEncoder()
            encoder.load_state_dict(load_file(args.text_encoder), strict=True, assign=True)
            encoder = encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            tokenizer = HuggingfaceTokenizer(name=args.tokenizer, seq_len=MAX_TOKENS, clean="whitespace")
        else:
            encoder = VLMTextEncoder(kind, model_path=source, device=device)
    start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            prompts = [item[0] for item in batch]
            if kind == "umt5-xxl":
                ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True,
                                      max_length=MAX_TOKENS, padding="max_length", truncation=True)
                context = encoder(ids.to(device), mask.to(device))
                context = context.masked_fill(~mask.to(device).bool().unsqueeze(-1), 0)
                if not torch.isfinite(context).all():
                    raise ValueError("Nonfinite UMT5 embedding")
                results, mask = {methods[0]: context.cpu()}, mask.bool().cpu()
            else:
                results, mask = encoder.encode(prompts, methods)
            for i, (_, destinations) in enumerate(batch):
                for method, file in destinations:
                    save_file({"context": results[method][i].contiguous(), "attention_mask": mask[i].contiguous(),
                               "seq_len": mask[i].sum().long()}, str(file) + f".tmp.{rank}")
                    os.replace(str(file) + f".tmp.{rank}", file)
            if offset % (args.batch_size * 10) == 0 or offset + len(batch) == len(pending):
                print(json.dumps({"rank": rank, "encoded": offset + len(batch), "total": len(pending),
                                  "elapsed_s": time.monotonic() - start}), flush=True)
            time.sleep(args.sleep_seconds)
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        for _, directory, identity, files in jobs:
            if not all((directory / filename).is_file() for filename in files.values()):
                raise ValueError(f"Incomplete cache: {directory}")
            temporary = directory / "manifest.json.tmp"
            temporary.write_text(json.dumps({**identity, "prompts": files}, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(directory / "manifest.json")
        print(json.dumps({"complete": True, "directories": [str(job[1]) for job in jobs]}), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()
