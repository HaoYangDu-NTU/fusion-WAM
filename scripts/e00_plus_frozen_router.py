"""Matched E00 decoder-only vs FP32 dense-router fine-tunes and 600-trial pilot.

Preparation and launch are explicit stages. All checkpoints/reports are written
under outputs/e00_plus_frozen_router and reports/E00_plus/frozen_router_*.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf, open_dict
from openwam.experiments import preflight, resolve_experiment, runtime_env

SOURCE_DIR = Path("/data/fcy/code/OpenWAM_v2/outputs/libero_vjepa_1_3b_dev_vx_ax_s42.0/2026-10-26_20-36-02")
SOURCE_CKPT = "checkpoint_step_10690.safetensors"
ROUTER_INIT = ROOT / "outputs/e00_plus_finetune/initialization/B1"
OUT = ROOT / "outputs/e00_plus_frozen_router"
REPORT = ROOT / "reports/E00_plus/frozen_router_20261111"
ARMS = {
    "decoder_only": ("E00-FrozenDecoder", ("action_backbone.action_decoder.",), False),
    "fp32_dense": ("E00+-FrozenDense", ("action_backbone.action_decoder.", "action_router."), True),
}


def init_dir(arm):
    # This revision combines the original E00 model config with B1's
    # seed-matched initial Router tensors. Keep the earlier experimental
    # initialization directory intact for auditability.
    name = f"{arm}_e00_v2"
    return OUT / "initialization" / name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def prepare():
    source_weights = SOURCE_DIR / SOURCE_CKPT
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    if not (ROUTER_INIT / "checkpoint_step_0.safetensors").is_file():
        raise FileNotFoundError(ROUTER_INIT)
    source_cfg = OmegaConf.load(SOURCE_DIR / "config.yaml")
    rows = {}
    for arm, (experiment, prefixes, router) in ARMS.items():
        profile_cfg, profile, _ = resolve_experiment(experiment)
        cache = preflight(profile_cfg, profile)
        directory = init_dir(arm)
        if directory.exists():
            manifest = directory / "PREPARED.json"
            if not manifest.exists():
                raise RuntimeError(f"Incomplete initialization exists: {directory}")
            existing = json.loads(manifest.read_text())
            if existing["source_sha256"] != sha(source_weights) or existing["router"] != router:
                raise RuntimeError(f"Initialization identity mismatch: {directory}")
            rows[arm] = existing
            continue
        directory.mkdir(parents=True)
        if router:
            # Rebuild the Router on top of the exact original E00 config.
            # The profile's current defaults can drift from the checkpoint's
            # frozen representation path, so only copy the Router schema from
            # the matched B1 initialization.
            cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=False))
            router_cfg = OmegaConf.load(ROUTER_INIT / "config.yaml").model.architecture.action_router
            weights = ROUTER_INIT / "checkpoint_step_0.safetensors"
            with open_dict(cfg.model.architecture):
                cfg.model.architecture.action_router = OmegaConf.create(
                    OmegaConf.to_container(router_cfg, resolve=False)
                )
            with open_dict(cfg.model.architecture.action_router):
                cfg.model.architecture.action_router.fp32_gate = True
                cfg.model.architecture.action_router.fp32_residual = True
                cfg.model.architecture.action_router.sparse_weight = 0.0
        else:
            cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=False))
            weights = source_weights
        # These are host-local asset locations only. Keep model behavior and
        # all architectural/fusion settings from the source E00 checkpoint.
        cfg.model.video_backbone.text_embedding_dir = profile_cfg.model.video_backbone.text_embedding_dir
        cfg.model.video_backbone.encoder.model_path = profile_cfg.model.video_backbone.encoder.model_path
        saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        saved.training.finetune_ckpt_path = None
        saved.training.resume_ckpt_path = None
        saved.training.trainable_parameter_prefixes = list(prefixes)
        saved.training.output_path = str(OUT / "train" / arm)
        saved.training.max_steps = 1000
        saved.training.batch_size = 64
        saved.training.learning_rate = 1.0e-5
        saved.training.mixed_precision = "bf16"
        saved.training.zero_stage = 1
        saved.training.save_steps = 500
        saved.training.save_full_states_for_resume = True
        saved.training.keep_last_k_ckpts = 2
        saved.project.seed = 42
        saved.project.wandb.run_name = f"e00_frozen_{arm}_s42"
        saved.model.video_backbone.text_embedding_fingerprint = cache["model_fingerprint"]
        OmegaConf.save(saved, directory / "config.yaml", resolve=True)
        os.link(weights, directory / "checkpoint_step_0.safetensors")
        metadata_source = ROUTER_INIT if router else SOURCE_DIR
        for name in ("manifest.json", "normalization_stats.npy"):
            if (metadata_source / name).exists():
                shutil.copyfile(metadata_source / name, directory / name)
        row = dict(arm=arm, experiment=experiment, source_checkpoint=str(source_weights),
                   source_sha256=sha(source_weights), initialization_checkpoint=str(weights),
                   initialization_sha256=sha(weights), router=router,
                   trainable_prefixes=list(prefixes), text_cache=cache,
                   output=str(OUT / "train" / arm), steps=1000, learning_rate=1e-5,
                   batch_per_gpu=64, global_batch=256, seed=42,
                   gate_dtype="float32" if router else None,
                   residual_compute="FP32 multiply and residual add" if router else None,
                   model_config_source="original E00 checkpoint; host-local V-JEPA/text-cache paths resolved")
        atomic_json(directory / "PREPARED.json", row)
        rows[arm] = row
    atomic_json(REPORT / "plan.json", dict(created_at=datetime.now().isoformat(), arms=rows,
                shared_source_checkpoint=True, router_init_seed=42,
                evaluation_manifest=str(REPORT.parent / "diagnosis_20261111/plus_selection.json")))


def command(arm, stage, steps):
    experiment, prefixes, router = ARMS[arm]
    cfg, _, overrides = resolve_experiment(experiment)
    init = init_dir(arm)
    is_smoke = stage.startswith("smoke")
    target = OUT / stage / arm if is_smoke else OUT / "train" / arm
    prefix_json = json.dumps(list(prefixes), separators=(",", ":"))
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            str(ROOT / "scripts/train.py"), *overrides,
            f"training.finetune_ckpt_path={init}", "training.resume_ckpt_path=null",
            "+training.model_config_from_finetune_checkpoint=true",
            f"+training.trainable_parameter_prefixes={prefix_json}",
            f"training.learning_rate={1e-5}", f"training.max_steps={steps}",
            "training.batch_size=64", "training.gradient_accumulation_steps=1",
            "training.mixed_precision=bf16", "training.zero_stage=1",
            "training.save_full_states_for_resume=false" if is_smoke else "training.save_full_states_for_resume=true",
            "training.save_steps=0" if is_smoke else "training.save_steps=500",
            "training.keep_last_k_ckpts=1" if is_smoke else "training.keep_last_k_ckpts=2",
            "training.dataset_num_workers=4", f"training.output_path={target}",
            f"model.video_backbone.text_embedding_fingerprint={json.load((OUT / 'initialization' / arm / 'PREPARED.json').open())['text_cache']['model_fingerprint']}",
            "project.seed=42", f"project.wandb.run_name=e00_frozen_{arm}_{stage}"]


def train_stage(arm, stage, steps):
    directory = REPORT / stage / arm
    if (directory / "DONE.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    cmd = command(arm, stage, steps)
    atomic_json(directory / "command.json", cmd)
    env = {**os.environ, **runtime_env("0,1,2,3"), "OPENBLAS_NUM_THREADS": "1"}
    with (directory / "console.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        atomic_json(directory / "status.json", dict(state="running", pid=proc.pid, arm=arm,
                    stage=stage, steps=steps, started=datetime.now().isoformat()))
        code = proc.wait()
    if code:
        atomic_json(directory / "status.json", dict(state="failed", returncode=code))
        raise RuntimeError(f"Training failed: {arm}/{stage}; inspect {directory / 'console.log'}")
    atomic_json(directory / "status.json", dict(state="completed", returncode=0,
                arm=arm, stage=stage, steps=steps, finished=datetime.now().isoformat()))
    atomic_json(directory / "DONE.json", dict(finished=datetime.now().isoformat(),
                arm=arm, stage=stage, steps=steps, output=str(OUT / stage / arm)))


def evaluate(arm):
    directory = REPORT / "evaluation" / arm
    if (directory / "DONE.json").exists():
        return
    selection = REPORT.parent / "diagnosis_20261111/plus_excluded_tasks.json"
    model_dir = OUT / "train" / arm
    checkpoints = sorted(model_dir.glob("*/checkpoint_step_1000.safetensors"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected exactly one final checkpoint_step_1000.safetensors under {model_dir}; "
            f"found {len(checkpoints)}. Refusing to launch an evaluation with an ambiguous path."
        )
    checkpoint_dir = checkpoints[0].parent
    command_args = [sys.executable, "-m", "benchmarks.vla_harness.run_eval",
        "--benchmark", "libero-plus", "--ckpt-dir", str(checkpoint_dir),
        "--ckpt-name", "checkpoint_step_1000.safetensors",
        "--libero-path", "/data/fcy/code/benchmarks/LIBERO-plus",
        "--libero-plus-python", "/data/fcy/basics/miniconda3/envs/libero-plus/bin/python",
        "--correct-instruction-suffix", "true",
        "--categories", "Camera Viewpoints,Light Conditions,Background Textures",
        "--exclude-tasks-file", str(selection), "--gpus", "0,1,2,3", "--workers", "32",
        "--max-batch-size", "8", "--render", "gpu", "--render-gpus", "3",
        "--image-format", "raw", "--base-port", "10300" if arm == "decoder_only" else "10304",
        "--output-dir", str(directory / "evaluation")]
    atomic_json(directory / "command.json", command_args)
    env = {**os.environ, **runtime_env("0,1,2,3"), "OPENBLAS_NUM_THREADS": "1"}
    with (directory / "console.log").open("w") as log:
        proc = subprocess.Popen(command_args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        atomic_json(directory / "status.json", dict(state="running", pid=proc.pid, arm=arm,
                    trials=600, started=datetime.now().isoformat()))
        code = proc.wait()
    if code:
        atomic_json(directory / "status.json", dict(state="failed", returncode=code))
        raise RuntimeError(f"Evaluation failed: {arm}; inspect {directory / 'console.log'}")
    atomic_json(directory / "status.json", dict(state="completed", returncode=0,
                arm=arm, trials=600, finished=datetime.now().isoformat()))
    atomic_json(directory / "DONE.json", dict(finished=datetime.now().isoformat(), arm=arm, trials=600))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "smoke", "smoke_e00", "train", "evaluate", "all"))
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("smoke", "all"):
        for arm in ARMS:
            train_stage(arm, "smoke", 4)
    if args.stage == "smoke_e00":
        for arm in ARMS:
            train_stage(arm, "smoke_e00", 4)
    if args.stage in ("train", "all"):
        for arm in ARMS:
            train_stage(arm, "train", 1000)
    if args.stage in ("evaluate", "all"):
        for arm in ARMS:
            evaluate(arm)


if __name__ == "__main__":
    main()
