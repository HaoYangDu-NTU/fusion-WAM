"""Read the official Plus protocol/catalog without importing either simulator."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

OFFICIAL = Path(__file__).resolve().parents[1] / "libero-plus"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
scheduler = importlib.import_module("benchmarks.libero-plus.scheduler")


def load_config():
    scheduler._validate_protocol(
        SimpleNamespace(
            policy_config=OFFICIAL / "policy_config.yml", seed=None, smoke=False, num_trials=1, trial_start=0
        )
    )
    cfg = yaml.safe_load((OFFICIAL / "policy_config.yml").read_text())
    if cfg["action_mode"] != "eef":
        raise ValueError("LIBERO-plus requires the official EEF action contract")
    return cfg


def task_catalog(root: Path, suites=SUITES, task_ids=None, categories=None):
    """Use official 1-based metadata IDs, checking against the actual task map."""
    source = root / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(source.read_text())
    mapping = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets)
    )
    counts = {suite: len(mapping[suite]) for suite in suites}
    if any(len(set(mapping[suite])) != counts[suite] for suite in suites):
        raise ValueError("LIBERO-plus task map contains duplicate names")
    if task_ids is not None and (not task_ids or any(i < 0 for i in task_ids) or len(set(task_ids)) != len(task_ids)):
        raise ValueError("task_ids must be distinct non-negative indices")
    jobs = scheduler._build_jobs(suites, counts, task_ids)
    metadata = scheduler._load_task_metadata(root, jobs)
    tasks = []
    for job in jobs:
        item = metadata[job]
        if mapping[job.suite][job.task_id] != item.name:
            raise ValueError("LIBERO-plus metadata and task-map order disagree")
        tasks.append(
            {
                "name": f"{job.suite}/{job.task_id:04d}/{item.name}",
                "suite": job.suite,
                "task_id": job.task_id,
                "task_name": item.name,
                "category": item.category,
                "difficulty_level": item.difficulty_level,
            }
        )
    if categories is not None:
        if (not categories or len(set(categories)) != len(categories)
                or any(c not in scheduler.PLUS_CATEGORY_ORDER for c in categories)):
            raise ValueError(f"categories must be distinct entries from {scheduler.PLUS_CATEGORY_ORDER}")
        tasks = [task for task in tasks if task["category"] in categories]
        if not tasks:
            raise ValueError("No tasks match the selected perturbation categories")
    order = {category: i for i, category in enumerate(scheduler.PLUS_CATEGORY_ORDER)}
    return sorted(tasks, key=lambda task: (order[task["category"]], task["suite"], task["task_id"]))


def check_text_cache(checkpoint_dir, tasks, languages):
    """Fail before loading GPUs if an offline LIBERO cache lacks Plus prompts."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    cache_dir = OmegaConf.select(cfg, "model.video_backbone.text_embedding_dir", default=None)
    if not cache_dir:
        return {"mode": "native_text_encoder"}
    directory = Path(cache_dir)
    # Relocated legacy deployments may carry their cache beside the checkpoint.
    # Encoder/method/fingerprint and exact prompt coverage are still checked below.
    local_cache = checkpoint_dir / "text_embedding"
    if not (directory / "manifest.json").is_file() and (local_cache / "manifest.json").is_file():
        directory = local_cache
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(
            f"LIBERO-plus text cache is not ready: {directory / 'manifest.json'}. "
            "Prepare exact Plus text embeddings as documented in "
            "benchmarks/vla_harness/LIBERO_PLUS.md; no model has been loaded."
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    from openwam.text_condition.config import text_spec

    kind = OmegaConf.select(cfg, "model.video_backbone.text_encoder_type", default="umt5-xxl")
    method = OmegaConf.select(cfg, "model.video_backbone.vlm_text_embedding_method", default="avg_all_layers")
    fingerprint = OmegaConf.select(cfg, "model.video_backbone.text_embedding_fingerprint", default=None)
    if fingerprint is not None and manifest.get("model_fingerprint") != fingerprint:
        raise ValueError("Text cache model fingerprint differs from training checkpoint")
    if manifest.get("version") == 2:
        for key, value in text_spec(kind, method).items():
            if manifest.get(key) != value:
                raise ValueError(f"Text cache {key} disagrees with trained checkpoint")
    elif kind != "umt5-xxl":
        raise ValueError("VLM evaluation requires its matching version-2 text cache")
    selected = {(task["suite"], task["task_id"]) for task in tasks}
    described = {(task["suite"], task["task_id"]) for task in languages}
    if not selected <= described:
        raise ValueError("Official Plus language catalog is missing selected tasks")
    prompts = {task["language"] for task in languages if (task["suite"], task["task_id"]) in selected}
    files = manifest["prompts"]
    missing = sorted(prompt for prompt in prompts if prompt not in files or not (directory / files[prompt]).is_file())
    if missing:
        raise ValueError(
            f"Offline text cache {directory} lacks {len(missing)}/{len(prompts)} exact LIBERO-plus prompts. "
            f"Example: {missing[0]!r}. Prepare Plus text embeddings as documented in "
            "benchmarks/vla_harness/README.md before evaluation; no model has been loaded."
        )
    return {"mode": "offline", "directory": str(directory), "unique_prompts": len(prompts), "missing": 0}


def text_cache_preflight(args, tasks):
    """Use the official client interpreter to resolve language perturbations on CPU."""
    repo = OFFICIAL.parents[1]
    osmesa = repo / "third_party/osmesa/usr/lib/x86_64-linux-gnu"
    with tempfile.TemporaryDirectory(prefix="openwam-plus-prompts-") as directory:
        root = Path(directory)
        output = root / "tasks.json"
        env = {
            **os.environ,
            "PYTHONPATH": f"{repo}:{args.libero_path}",
            "LIBERO_PLUS_PATH": str(args.libero_path),
            "LIBERO_PLUS_CONFIG_ROOT": str(root / "config"),
            "LIBERO_CONFIG_PATH": str(root / "config"),
            "CUDA_VISIBLE_DEVICES": "",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LD_LIBRARY_PATH": str(osmesa) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        result = subprocess.run(
            [str(args.libero_plus_python), "-m", "benchmarks.vla_harness.libero_plus_env", "--describe", str(output),
             "--correct-instruction-suffix", str(getattr(args, "correct_instruction_suffix", True))],
            env=env,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError("Official Plus task/language preflight failed:\n" + result.stdout[-6000:])
        languages = json.loads(output.read_text())
        if getattr(args, "policy_backend", "openwam") == "vpp2":
            from benchmarks.vla_harness.vpp2_policy import validate_assets
            validate_assets(args.ckpt_dir)
            selected = {(t["suite"], t["task_id"]) for t in tasks}
            described = {(t["suite"], t["task_id"]) for t in languages}
            if not selected <= described:
                raise ValueError("VPP2 language catalog is missing selected tasks")
            return {"mode": "native-vpp2-online-umt5-128",
                    "unique_prompts": len({t["language"] for t in languages
                                           if (t["suite"], t["task_id"]) in selected})}
        if getattr(args, "policy_backend", "openwam") == "fm-vjepa":
            from benchmarks.vla_harness.fm_policy import check_text_cache as check_fm_cache
            return check_fm_cache(args.fm_text_cache, tasks, languages)
        return check_text_cache(args.ckpt_dir, tasks, languages)
