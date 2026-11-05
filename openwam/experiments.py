"""Experiment orchestration. Each encoder retains its own config and model API."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
LIBERO_CATALOG = Path("/data/fcy/code/OpenWAM_v2/data/umt5-embedding/libero/manifest.json")
PLUS_CATALOG = Path("/data/fcy/code/OpenWAM_v2/data/cosmos3-embedding/libero-plus/avg_all_layers")


def load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def source_fingerprint():
    """Content hash of local code/config; weight files are inventoried separately."""
    digest = hashlib.sha256()
    for directory in ("openwam", "scripts", "configs", "third_party", "benchmarks", "assets"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_symlink() and path.is_file() and path.suffix in (".py", ".yaml", ".sh", ".json"):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_experiment(experiment, seed=42, batch_size=None, smoke_steps=None, resume=None):
    if batch_size is not None and batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if smoke_steps is not None and smoke_steps <= 0:
        raise ValueError("--smoke-steps must be positive")
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    matrix = load_yaml(ROOT / "configs/experiments.yaml")
    if experiment not in matrix["experiments"]:
        raise ValueError(f"Unknown experiment {experiment}; register a profile in configs/experiments.yaml")
    profile = load_yaml(ROOT / matrix["experiments"][experiment])
    if profile.get("base_experiment"):
        profile = {**load_yaml(ROOT / matrix["experiments"][profile["base_experiment"]]), **profile}
    registry = json.loads((ROOT / "assets/registry.json").read_text())
    asset = registry["pretrained"][profile["asset"]]
    t = protocol["training"]
    data = protocol["data"]
    text = protocol["text"]
    policy = protocol["policy"]
    if t["global_batch_size"] != t["world_size"] * t["batch_size_per_gpu"] * t["gradient_accumulation_steps"]:
        raise ValueError("Protocol global_batch_size disagrees with world size, batch and accumulation")
    if not smoke_steps and batch_size is not None and batch_size != t["batch_size_per_gpu"]:
        raise ValueError("Formal comparisons require the protocol batch size; use --smoke-steps for probes")
    freeze = [
        "video_backbone.text_encoder", "video_backbone.reason1",
        "video_backbone.vae", "video_backbone.image_encoder", profile["freeze_visual"],
    ]
    overrides = [
        "dataloader=libero", "model/video_backbone=wan22_style_1_3b",
        *profile["overrides"],
        f"model.video_backbone.encoder.model_path={asset['directory']}",
        "model.video_backbone.model_path=null", "model.video_backbone.from_scratch=true",
        f"model.video_backbone.text_encoder_type={text['encoder_type']}",
        f"model.video_backbone.vlm_text_embedding_method={text['aggregation']}",
        f"model.video_backbone.text_embedding_dir={ROOT / text['cache_train']}",
        "model.freeze=[" + ",".join(freeze) + "]",
        "model.architecture.video_prediction_type=x", "model.architecture.action_prediction_type=x",
        "model.architecture.variant=joint_self_attn",
        "model.architecture.attention_mask_mode=mutual",
        "model.architecture.video_attention_mask_mode=clean_condition_causal",
        "model.architecture.use_proprioception=true", "model.architecture.repa_enabled=false",
        f"model.video_backbone.shift_video={t['shift_video']}",
        f"model.action_backbone.shift_action={t['shift_action']}",
        f"dataloader.dataset_dir={data['dataset_dir']}",
        f"dataloader.num_frames={data['num_frames']}",
        f"dataloader.video_stride={data['video_stride']}",
        f"dataloader.window_stride={data['window_stride']}",
        f"dataloader.history_num_tubelets={data['history_num_tubelets']}",
        f"dataloader.seperate_encode_view={str(data['encode_views_independently']).lower()}",
        f"dataloader.color_jitter.enabled={str(data['color_jitter']).lower()}",
        f"training.batch_size={batch_size or t['batch_size_per_gpu']}",
        f"training.num_epochs={t['num_epochs']}",
        f"training.max_steps={smoke_steps or t['max_steps'] or 'null'}",
        f"training.resume_ckpt_path={str(Path(resume).resolve()) if resume else 'null'}",
        f"training.output_path={ROOT / 'outputs' / profile['name'] / ('smoke' if smoke_steps else 'train') / ('s'+str(seed))}",
        "training.save_full_states_for_resume=true",
        f"training.save_steps={t['save_steps']}", f"training.keep_last_k_ckpts={t['keep_last_k_ckpts']}",
        "+training.epoch_seeded_sampler=true",
        f"project.seed={seed}", f"project.wandb.run_name={profile['name']}_s{seed}",
    ]
    # Explicit protocol fields must reach Hydra instead of silently using inherited defaults.
    for key in ("dim", "ffn_dim", "num_layers", "num_heads"):
        overrides.append(f"model.video_backbone.dit_config.{key}={policy['video_dit'][key]}")
    for key in ("dim", "ffn_dim"):
        overrides.append(f"model.action_backbone.{key}={policy['action_dit'][key]}")
    for key in ("variant", "video_prediction_type", "action_prediction_type",
                "attention_mask_mode", "video_attention_mask_mode",
                "use_proprioception", "repa_enabled", "action_scale"):
        overrides.append(f"model.architecture.{key}=" + json.dumps(policy[key]))
    overrides.extend([
        f"model.architecture.bridge_interval={policy['action_dit']['bridge_interval']}",
        f"model.architecture.action_dim={data['action_dim']}",
        f"model.architecture.state_dim={data['state_dim']}",
        f"model.video_backbone.dit_config.text_dim={policy['text_context_dim']}",
    ])
    for key in (
        "learning_rate", "adam_betas", "weight_decay", "max_grad_norm",
        "gradient_accumulation_steps", "lr_scheduler", "warmup_ratio", "lr_min_ratio",
        "mixed_precision", "zero_stage", "use_gradient_checkpointing",
        "offload_optimizer_device", "use_gradient_checkpointing_offload",
        "noise_schedule_distribution", "lambda_video", "lambda_action",
    ):
        overrides.append("training." + key + "=" + json.dumps(t[key], separators=(",", ":")))
    overrides.append(f"training.dataset_num_workers={t['dataset_num_workers_per_rank']}")
    # Native video encoders may require dense frames and their own history;
    # these are explicitly recorded in the profile and resolved run identity.
    for key, value in profile.get("data_overrides", {}).items():
        if key not in ("type", "native_view_size", "video_stride", "history_num_tubelets", "clean_condition_frame_offsets", "latent_cache_dir", "frame_feature_cache_dir"):
            raise ValueError(f"Unsupported encoder data override: {key}")
        prefix = "+" if key in ("frame_feature_cache_dir", "native_view_size") else ""
        overrides.append(prefix + "dataloader." + key + "=" + json.dumps(value, separators=(",", ":")))
    allowed_ablation = {"model.architecture.video_prediction_type", "model.architecture.action_prediction_type",
                        "model.video_backbone.shift_video", "training.lambda_video"}
    for key, value in profile.get("policy_overrides", {}).items():
        if key not in allowed_ablation:
            raise ValueError(f"Unsupported controlled ablation: {key}")
        overrides.append(key + "=" + json.dumps(value))
    if profile.get("report_future_quality"):
        overrides.append("+training.report_future_quality=true")
    for key, value in profile.get("action_router", {}).items():
        overrides.append("model.architecture.action_router." + key + "=" + json.dumps(value))
    if "worker_threads" in profile:
        worker_threads = int(profile["worker_threads"])
        if worker_threads < 1:
            raise ValueError("worker_threads must be positive")
        overrides.append(f"+training.dataloader_worker_threads={worker_threads}")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg, profile, overrides


def validate_cache(directory, catalog, fingerprint=None):
    """Read every tensor through the production cache validator, on CPU."""
    from openwam.model.video_backbone.wan.text_embedding_cache import TextEmbeddingCache
    from openwam.text_condition.encoder import SYSTEM_PROMPT

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = json.loads((directory / "encoding.json").read_text())
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"Cache manifest/encoding mismatch: {key}")
    for key, value in {
        "padding_side": "right", "dtype": "bfloat16", "layer_norm_eps": 1e-5,
        "prompt_format": "cosmos-predict25-chat-v1", "system_prompt": SYSTEM_PROMPT,
    }.items():
        if manifest.get(key) != value:
            raise ValueError(f"Cache protocol mismatch: {key}")
    expected_prompts = set(json.loads(Path(catalog).read_text())["prompts"])
    if set(manifest["prompts"]) != expected_prompts:
        raise ValueError(f"Cache prompt catalog differs: {directory}")
    if not manifest.get("model_fingerprint"):
        raise ValueError("Cache has no model fingerprint")
    cache = TextEmbeddingCache(directory, encoder_type="rynnbrain1p1-9b",
                               method="avg_all_layers", max_tokens=128, fingerprint=fingerprint)
    if cache.shape != (128, 4096):
        raise ValueError(f"Unexpected Rynn cache shape {cache.shape}")
    import torch
    for prompt in sorted(expected_prompts):
        context, _ = cache.encode([prompt], device="cpu")
        if not torch.isfinite(context).all():
            raise ValueError(f"Nonfinite cache tensor: {prompt}")
    return {"directory": str(directory), "prompts": len(expected_prompts),
            "model_fingerprint": cache.fingerprint,
            "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()}


def preflight(cfg, profile):
    from openwam.text_condition.precompute import model_fingerprint

    registry = json.loads((ROOT / "assets/registry.json").read_text())
    for name in ("rynnbrain_9b", profile["asset"]):
        asset = registry["pretrained"][name]
        for filename, size in asset["required_files"].items():
            file = Path(asset["directory"]) / filename
            if not file.is_file() or file.stat().st_size != size:
                raise ValueError(f"Missing or unexpected pretrained file: {file}")
    dataset = Path(cfg.dataloader.dataset_dir)
    if cfg.dataloader.get("frame_feature_cache_dir"):
        from openwam.dataloader.registry import build_dataset
        from openwam.dataloader.image_frame_cache import ImageFrameCache
        from openwam.dataloader.spatial_frame_cache_specs import frame_cache_identity
        raw = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
        raw.frame_feature_cache_dir = None
        ds = build_dataset(raw, split="train")
        cache = ImageFrameCache(cfg.dataloader.frame_feature_cache_dir,
                                frame_cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, ds))
        if cfg.training.max_steps is None:
            cache.require_complete()
            validation = Path(cfg.dataloader.frame_feature_cache_dir) / "validation.json"
            expected_hash = hashlib.sha256(json.dumps(cache.manifest, sort_keys=True).encode()).hexdigest()
            if (not validation.exists() or not json.loads(validation.read_text()).get("passed")
                    or json.loads(validation.read_text()).get("identity_sha256") != expected_hash):
                raise ValueError("Frame cache requires GPU/batch64 equivalence acceptance: scripts/validate_image_features.py")
    if cfg.dataloader.get("latent_cache_dir"):
        from openwam.dataloader.registry import build_dataset
        from openwam.dataloader.wan_latent_cache import WanLatentCache, cache_identity
        raw = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
        raw.latent_cache_dir = None
        ds = build_dataset(raw, split="train")
        cache = WanLatentCache(cfg.dataloader.latent_cache_dir,
                              cache_identity(cfg.dataloader, cfg.model.video_backbone.encoder, len(ds)))
        if cfg.training.max_steps is None:
            cache.require_complete()
    for name in ("conversion.json", "info.json"):
        if not (dataset / "meta" / name).is_file():
            raise FileNotFoundError(dataset / "meta" / name)
    if profile["asset"] == "wan21_i2v_14b":
        from openwam.model.video_backbone.encoder.wan21_features import load_empty_context
        path = Path(registry["pretrained"][profile["asset"]]["directory"])
        if Path(cfg.model.video_backbone.encoder.empty_context_path).resolve() != (path / "empty_context.safetensors").resolve():
            raise ValueError("E03 requires the registered native Wan empty-text conditioning asset")
        load_empty_context(cfg.model.video_backbone.encoder.empty_context_path, path)
    cache = validate_cache(
        cfg.model.video_backbone.text_embedding_dir, LIBERO_CATALOG,
        model_fingerprint(registry["pretrained"]["rynnbrain_9b"]["directory"]),
    )
    # Pin the exact cache identity in saved deployment config.
    cfg.model.video_backbone.text_embedding_fingerprint = cache["model_fingerprint"]
    return cache


def run_identity(cfg, profile, world_size, cache):
    config = OmegaConf.to_container(cfg, resolve=True)
    for key in ("output_path", "resume_ckpt_path"):
        config["training"].pop(key)
    # Hydra's ${now:...} output location changes on every process invocation.
    # It does not affect model/data/optimizer state and must not invalidate resume.
    config["project"].pop("output_dir", None)
    dataset = Path(cfg.dataloader.dataset_dir)
    metadata = {}
    for path in sorted((dataset / "meta").glob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".npy"):
            metadata[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assets = json.loads((ROOT / "assets/registry.json").read_text())["pretrained"]
    weights = {}
    for name in ("rynnbrain_9b", profile["asset"]):
        entry = assets[name]
        weights[name] = {
            file: {"size": (Path(entry["directory"]) / file).stat().st_size,
                   "mtime_ns": (Path(entry["directory"]) / file).stat().st_mtime_ns}
            for file in entry["required_files"]
        }
    return {"version": 1, "experiment": profile["id"], "geometry": profile["geometry"],
            "world_size": world_size, "config": config, "cache": cache,
            "dataset_metadata_sha256": metadata, "pretrained_size_mtime": weights,
            "source_sha256": source_fingerprint()}


def validate_resume(directory, identity):
    previous = json.loads((Path(directory) / "experiment_identity.json").read_text())
    if previous != identity:
        different = [key for key in identity if previous.get(key) != identity[key]]
        raise ValueError("Full-state resume identity changed: " + ", ".join(different))
    if (Path(directory) / "STOP_TRAINING").exists():
        raise ValueError("Remove STOP_TRAINING from the run directory before resuming")


def runtime_env(gpus):
    # WebSocket evaluation servers are local even when the login shell has a proxy.
    no_proxy = ",".join(filter(None, [os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")),
                                    "127.0.0.1", "localhost"]))
    return {
        "CUDA_VISIBLE_DEVICES": gpus, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1", "OPENWAM_DISABLE_FUSED_ROPE": "1",
        "OPENWAM_DISABLE_FUSED_ATTENTION": "1", "OPENWAM_DISABLE_COMPILED_RMS_NORM": "1",
        "WAM_ATTENTION_IMPL": "sdpa",
        "NO_PROXY": no_proxy, "no_proxy": no_proxy,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "third_party/vla-evaluation-harness/src")]),
    }


def train(args):
    cfg, profile, overrides = resolve_experiment(
        args.experiment, args.seed, args.batch_size, args.smoke_steps, args.resume)
    gpus = args.gpus.split(",")
    if not all(g.strip().isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus requires unique numeric GPU IDs")
    world_size = len(gpus)
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    if world_size != protocol["training"]["world_size"] and not args.smoke_steps:
        raise ValueError("The comparison protocol requires four GPUs; use --smoke-steps for smoke runs")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={world_size}", str(ROOT / "scripts/train.py"), *overrides,
    ]
    env = runtime_env(args.gpus)
    env.update(profile.get("environment", {}))
    plan = {"experiment": profile, "world_size": world_size, "environment": env,
            "command": command, "cache_validation": "not_run"}
    if args.dry_run:
        destination = ROOT / "reports/resolved" / f"{args.experiment}_s{args.seed}"
        destination.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, destination / "config.yaml", resolve=True)
        (destination / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Resolved {profile['name']}: {profile['geometry']}")
        print(f"Config: {destination / 'config.yaml'}")
        print(shlex.join(command))
        return 0
    cache = preflight(cfg, profile)
    identity = run_identity(cfg, profile, world_size, cache)
    if args.resume:
        validate_resume(args.resume, identity)
    launch_dir = ROOT / "reports/launches" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.experiment)
    launch_dir.mkdir(parents=True)
    if args.smoke_steps:
        env["OPENWAM_TIMING_DIR"] = str(launch_dir / "timing")
    lock_path = launch_dir / "experiment_identity.json"
    lock_path.write_text(json.dumps(identity, indent=2) + "\n")
    plan["cache_validation"] = cache
    command += [
        f"+experiment.identity_file={lock_path}",
        f"model.video_backbone.text_embedding_fingerprint={cache['model_fingerprint']}",
    ]
    (launch_dir / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Launch report: {launch_dir}", flush=True)
    print(shlex.join(command), flush=True)
    os.execvpe(sys.executable, command, {**os.environ, **env})


def cache_command(args):
    common = ["--text-encoder-type", "rynnbrain1p1-9b",
              "--model-path", str(ROOT / "assets/pretrained/rynnbrain-9b"),
              "--vlm-text-embedding-method", "avg_all_layers"]
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    if args.benchmark == "libero":
        command = [sys.executable, str(ROOT / "scripts/precompute_libero_text.py"),
                   *common, "--catalog-manifest", str(LIBERO_CATALOG),
                   "--output-dir", str(ROOT / protocol["text"]["cache_train"])]
    else:
        command = [sys.executable, str(ROOT / "scripts/precompute_libero_plus_text.py"),
                   *common, "--catalog-dir", str(PLUS_CATALOG), "--text-mode", "corrected_text",
                   "--output-dir", str((ROOT / protocol["text"]["cache_plus"]).parent)]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        os.execvpe(sys.executable, command, {**os.environ, **runtime_env(args.gpus)})
    return 0


def evaluate(args):
    protocol = load_yaml(ROOT / "configs/protocol.yaml")
    cache = ROOT / protocol["text"]["cache_plus" if args.benchmark == "libero-plus" else "cache_train"]
    command = ["bash", str(ROOT / "scripts/eval_libero_vjepa.sh"),
               str(Path(args.run_dir).resolve()), args.checkpoint,
               "--benchmark", args.benchmark, "--gpus", args.gpus,
               "--render-gpus", args.gpus.split(",")[-1],
               "--output-dir", str(Path(args.output).resolve())]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        catalog = PLUS_CATALOG / "corrected_text/manifest.json" if args.benchmark == "libero-plus" else LIBERO_CATALOG
        cfg = OmegaConf.load(Path(args.run_dir) / "config.yaml")
        validate_cache(cache, catalog, cfg.model.video_backbone.text_embedding_fingerprint)
        os.execvpe("bash", command, {**os.environ, **runtime_env(args.gpus),
                                    "OPENWAM_PYTHON": sys.executable, "TEXT_EMBEDDING_DIR": str(cache)})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--smoke-steps", type=int)
    p.add_argument("--resume")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=train)
    p = sub.add_parser("cache")
    p.add_argument("benchmark", choices=("libero", "libero-plus"))
    p.add_argument("--gpus", default="0")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cache_command)
    p = sub.add_parser("eval")
    p.add_argument("run_dir")
    p.add_argument("checkpoint")
    p.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")
