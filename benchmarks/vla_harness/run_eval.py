"""Launch harness workers against one batched OpenWAM model server per GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from benchmarks.libero_plus_instructions import parse_bool
from benchmarks.vla_harness.checkpoint_contract import validate_feature_fusion_checkpoint
from benchmarks.vla_harness.libero import SUITES, OpenWAMLIBEROBenchmark
from benchmarks.vla_harness.recording import persist_recording_db, prepare_recording_db
from benchmarks.vla_harness.runtime import RuntimeMonitor
from benchmarks.vla_harness.summarize import write_summary
from benchmarks.vla_harness.fm_policy import add_policy_arguments, fm_arguments

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "outputs/libero/vjepa21_cached_text_step10690_20260918_125937/checkpoint_for_eval"
OSMESA_LIB = ROOT / "third_party/osmesa/usr/lib/x86_64-linux-gnu"


def make_plan(args):
    benchmark = getattr(args, "benchmark", "libero")
    if getattr(args, "dynamic_tasks", False) and benchmark != "libero-plus":
        raise ValueError("Dynamic tasks are restricted to independent single-trial Plus tasks")
    backend = getattr(args, "policy_backend", "openwam")
    if backend == "fm-vjepa":
        fm_arguments(args)
    cfg_path = args.ckpt_dir / "config.yaml"
    checkpoint_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
    fusion = validate_feature_fusion_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg) if backend == "openwam" else None
    from benchmarks.vla_harness.checkpoint_contract import validate_dinov3_checkpoint, validate_siglip_checkpoint, validate_ijepa_checkpoint, validate_wan21_features_checkpoint, validate_wan_vae_checkpoint, validate_flux2_framewise_checkpoint, validate_vggt_omega_checkpoint

    visual = ((validate_wan_vae_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg)
               or validate_wan21_features_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg)
               or validate_dinov3_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg)
               or validate_ijepa_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg)
               or validate_flux2_framewise_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg)
               or validate_vggt_omega_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg)
               or validate_siglip_checkpoint(args.ckpt_dir, args.ckpt_name, checkpoint_cfg))
              if backend == "openwam" else None)
    encoder_cfg = checkpoint_cfg.get("model", {}).get("video_backbone", {}).get("encoder") or {}
    seperate_encode_view = bool(encoder_cfg.get("seperate_encode_view", False))
    gpus = args.gpus.split(",")
    suites = args.suites.split(",")
    categories_arg = getattr(args, "categories", None)
    categories = None if categories_arg is None else [c.strip() for c in categories_arg.split(",")]
    if categories is not None and benchmark != "libero-plus":
        raise ValueError("--categories applies only to libero-plus")
    if benchmark == "libero-plus" and suites != list(SUITES):
        raise ValueError("LIBERO-Plus is selected by --categories; --suites is for standard LIBERO")
    task_ids = None if args.task_ids is None else sorted(int(i) for i in args.task_ids.split(","))
    if benchmark == "libero" and task_ids is None:
        task_ids = list(range(10))
    if args.smoke and args.task_ids is None and categories is None:
        task_ids = [0, 1]
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError("--gpus must contain distinct numeric GPU IDs")
    if not suites or len(set(suites)) != len(suites) or any(s not in SUITES for s in suites):
        raise ValueError(f"--suites must select from {SUITES}")
    if benchmark == "libero" and (
        not task_ids or len(set(task_ids)) != len(task_ids) or any(i < 0 or i > 9 for i in task_ids)
    ):
        raise ValueError("--task-ids must contain distinct indices in 0..9")
    if getattr(args, "smoke_max_steps", 20) < 1:
        raise ValueError("smoke-max-steps must be positive")
    if not args.smoke and getattr(args, "smoke_max_steps", 20) != 20:
        raise ValueError("smoke-max-steps requires --smoke")
    if (args.workers is not None and args.workers < 1) or args.max_batch_size < 1 or args.max_wait_time < 0:
        raise ValueError("workers/batch must be positive; wait time must be non-negative")
    if benchmark == "libero-plus":
        from benchmarks.vla_harness.libero_plus_protocol import load_config, task_catalog

        load_config()
        tasks = task_catalog(args.libero_path, suites, task_ids, categories)
        if args.smoke and args.task_ids is None and categories is not None:
            # Choose smoke IDs from the selected category, not unrelated IDs 0/1.
            smoke_ids = {t["task_id"] for t in tasks[:2]}
            tasks = [t for t in tasks if t["task_id"] in smoke_ids]
    else:
        tasks = [{"suite": s, "task_id": t} for s in suites for t in task_ids]
    exclude_file = getattr(args, "exclude_tasks_file", None)
    if exclude_file is not None:
        excluded = {tuple(pair) for pair in json.loads(exclude_file.read_text())}
        available = {(t["suite"], t["task_id"]) for t in tasks}
        if not excluded <= available:
            raise ValueError("Excluded tasks must belong to the selected catalog")
        tasks = [t for t in tasks if (t["suite"], t["task_id"]) not in excluded]
        if not tasks:
            raise ValueError("No tasks remain after exclusions")
    # More environments keep inference batches ready while other environments
    # simulate/reset. This scales with the requested GPUs, not checkpoint type.
    workers = min(args.workers if args.workers is not None else (16 * len(gpus) if benchmark == "libero-plus" else 32), len(tasks))
    gpus = gpus[:workers]
    # Explicit render devices share EGL work; keep the default on one device.
    renders = args.render_gpus.split(",") if args.render_gpus else ([gpus[-1]] if benchmark == "libero-plus" else ["3"])
    if args.render == "cpu" and args.render_gpus:
        raise ValueError("--render-gpus applies only to --render gpu")
    if any(not gpu.isdigit() for gpu in renders):
        raise ValueError("--render-gpus must contain numeric GPU IDs")
    trials = 1 if args.smoke or benchmark == "libero-plus" else 50
    # All suites share one work list, sharded by whole task.
    items = [(i, task, ep) for i, task in enumerate(tasks) for ep in range(trials)]
    servers = []
    for index, gpu in enumerate(gpus):
        port = args.base_port + index
        if not 1 <= port <= 65535:
            raise ValueError("Invalid server port")
        servers.append(
            {
                "gpu": gpu,
                "port": port,
                "command": [
                    sys.executable,
                    "-m",
                    "benchmarks.vla_harness.model_server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--args.checkpoint_dir",
                    str(args.ckpt_dir),
                    "--args.checkpoint_name",
                    args.ckpt_name,
                    "--args.max_batch_size",
                    str(args.max_batch_size),
                    "--args.max_wait_time",
                    str(args.max_wait_time),
                ],
            }
        )
    if backend == "fm-vjepa":
        for server in servers:
            server["command"] = [str(args.fm_python), "-m", "benchmarks.vla_harness.fm_model_server",
                "--host", "127.0.0.1", "--port", str(server["port"]),
                "--args.checkpoint_dir", str(args.ckpt_dir), "--args.checkpoint_name", args.ckpt_name,
                "--args.encoder", str(args.fm_encoder), "--args.text_cache", str(args.fm_text_cache),
                "--args.max_batch_size", str(args.max_batch_size), "--args.max_wait_time", str(args.max_wait_time)]
    if backend == "vpp2":
        for server in servers:
            server["command"] = [sys.executable, "-m", "benchmarks.vla_harness.vpp2_model_server",
                "--host", "127.0.0.1", "--port", str(server["port"]),
                "--args.checkpoint_dir", str(args.ckpt_dir),
                "--args.max_batch_size", str(args.max_batch_size), "--args.max_wait_time", str(args.max_wait_time)]
    client_action_queue = bool(getattr(args, "client_action_queue", False))
    if client_action_queue:
        for server in servers:
            server["command"].extend(["--args.client_action_queue", "true"])
    clients = []
    for shard in range(workers):
        slot = shard % len(servers)
        assigned = OpenWAMLIBEROBenchmark.shard_work_items(items, workers, shard)
        clients.append(
            {
                "shard": shard,
                "server": slot,
                "render_gpu": renders[shard % len(renders)],
                "tasks": [t for _, t, ep in assigned if ep == 0],
                "command": [
                    sys.executable,
                    "-m",
                    "benchmarks.vla_harness.worker",
                    "run",
                    "--no-docker",
                    "--config",
                    str(args.output_dir / "config.yaml"),
                    "--server-url",
                    f"ws://127.0.0.1:{servers[slot]['port']}",
                    "--shard-id",
                    str(shard),
                    "--num-shards",
                    str(workers),
                    "--eval-id",
                    args.eval_id,
                ],
            }
        )
    return {
        **({"feature_fusion": fusion} if fusion else {}),
        **({"visual_encoder": visual} if visual else {}),
        "benchmark": benchmark,
        "policy_backend": backend,
        "client_action_queue": client_action_queue,
        "categories": categories,
        "seperate_encode_view": seperate_encode_view,
        "correct_instruction_suffix": getattr(args, "correct_instruction_suffix", benchmark == "libero-plus"),
        "servers": servers,
        "workers": clients,
        "tasks": tasks,
        "num_trials": trials,
        "smoke": args.smoke,
        "dynamic_tasks": bool(getattr(args, "dynamic_tasks", False)),
    }


def launch(args, plan):
    plus = plan.get("benchmark") == "libero-plus"
    if importlib.metadata.version("mujoco") != "3.3.2":
        raise RuntimeError("OpenWAM LIBERO evaluation requires mujoco==3.3.2")
    if plus:
        version = subprocess.check_output(
            [
                str(args.libero_plus_python),
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('mujoco'))",
            ],
            text=True,
        ).strip()
        if version != "3.3.2":
            raise RuntimeError("The official LIBERO-plus client requires mujoco==3.3.2")
    if not (args.libero_path / "libero/libero/assets").is_dir():
        raise FileNotFoundError(args.libero_path / "libero/libero/assets")
    fm = plan.get("policy_backend", "openwam") == "fm-vjepa"
    vpp2 = plan.get("policy_backend") == "vpp2"
    required = (("config.json", "libero/action_step030000.pt", "libero/video_step010000.pt", "libero/dataset_stats.json")
                if vpp2 else ("resolved_args.json", "dataset_stats.json", args.ckpt_name) if fm
                else ("config.yaml", "normalization_stats.npy", args.ckpt_name))
    for name in required:
        if not (args.ckpt_dir / name).is_file():
            raise FileNotFoundError(args.ckpt_dir / name)
    if plus:
        from benchmarks.vla_harness.libero_plus_protocol import text_cache_preflight

        text_preflight = text_cache_preflight(args, plan["tasks"])
    if fm and not plus:
        # Use the same benchmark task languages as rollout, without loading a policy.
        from benchmarks.vla_harness.fm_policy import check_text_cache
        # Isolate LIBERO's configuration writes from other running evaluations.
        with tempfile.TemporaryDirectory(prefix="fm-libero-prompts-") as directory:
            env_keys = ("LIBERO_PATH", "LIBERO_CONFIG_ROOT", "LIBERO_CONFIG_PATH")
            previous = {key: os.environ.get(key) for key in env_keys}
            sys.path.insert(0, str(args.libero_path))
            try:
                os.environ.update(LIBERO_PATH=str(args.libero_path), LIBERO_CONFIG_ROOT=directory,
                                  LIBERO_CONFIG_PATH=directory)
                benchmark = OpenWAMLIBEROBenchmark(suites=args.suites.split(","), policy_backend="fm-vjepa")
                text_preflight = check_text_cache(args.fm_text_cache, plan["tasks"], benchmark.get_tasks())
            finally:
                sys.path.pop(0)
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    if args.render == "cpu" and not (OSMESA_LIB / "libOSMesa.so.8").is_file():
        raise FileNotFoundError("CPU rendering requires the isolated OSMesa runtime")
    import inspect

    from vla_eval.orchestrator import Orchestrator

    if 'getattr(benchmark, "shard_work_items"' not in inspect.getsource(Orchestrator._run_benchmark_inner):
        raise RuntimeError("Use the bundled harness with the benchmark-specific sharding hook")
    for server in plan["servers"]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", server["port"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_root = args.output_dir / "libero_config"
    config_root.mkdir()
    config_name = "libero_plus.yaml" if plus else "libero.yaml"
    cfg = yaml.safe_load((ROOT / "benchmarks/vla_harness/configs" / config_name).read_text())
    cfg.update(render=args.render, output_dir=str(args.output_dir))
    entry = cfg["benchmarks"][0]
    entry["episodes_per_task"] = plan["num_trials"]
    entry["params"] = {"suites": args.suites.split(","), "task_ids": sorted({t["task_id"] for t in plan["tasks"]})}
    if fm or vpp2:
        entry["params"]["policy_backend"] = plan["policy_backend"]
    entry["params"]["seperate_encode_view"] = plan.get("seperate_encode_view", False)
    if getattr(args, "exclude_tasks_file", None) is not None:
        entry["params"]["excluded_tasks"] = json.loads(args.exclude_tasks_file.read_text())
    if plus:
        entry["params"]["correct_instruction_suffix"] = args.correct_instruction_suffix
        entry["params"]["categories"] = plan.get("categories")
    entry["recording"] = {"record_step": bool(getattr(args, "export_step_jsonl", False)), "record_video": False}
    if plus and args.task_ids is None and not args.smoke:
        entry["params"]["task_ids"] = None  # Plus suites have different task counts.
    if args.smoke:
        entry["max_steps"] = getattr(args, "smoke_max_steps", 20)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    local_db, output_db = prepare_recording_db(args.output_dir, args.eval_id)
    manifest = {
        "eval_id": args.eval_id,
        "python": sys.executable,
        "checkpoint_dir": str(args.ckpt_dir),
        "checkpoint_name": args.ckpt_name,
        "render": args.render,
        "transport": {"image_format": args.image_format, "msgpack": "python"},
        "execution": {
            "client_action_queue": plan.get("client_action_queue", False), "capability_protocol_version": 1,
            "source_sha256": {name: hashlib.sha256((ROOT / "benchmarks/vla_harness" / name).read_bytes()).hexdigest()
                              for name in ("execution.py", "client.py", "chunk_server.py", "worker.py")},
        },
        "performance_dir": str(args.output_dir / "performance"),
        "export_step_jsonl": bool(getattr(args, "export_step_jsonl", False)),
        "recording_db": {"local_path": str(local_db), "output_path": str(output_db)},
        "plan": plan,
        "libero_path": str(args.libero_path),
        "official_policy_sha256": hashlib.sha256(
            (ROOT / "benchmarks" / ("libero-plus" if plus else "libero") / "policy_config.yml").read_bytes()
        ).hexdigest(),
        "upstream": json.loads((ROOT / "benchmarks/vla_harness/UPSTREAM.json").read_text()),
    }
    if fm:
        manifest["fm_assets"] = {"encoder": str(args.fm_encoder), "text_cache": str(args.fm_text_cache), "python": str(args.fm_python), "replan_steps": 16}
        manifest["text_cache_preflight"] = text_preflight
    if vpp2:
        from benchmarks.vla_harness.vpp2_policy import SOURCE
        manifest["vpp2"] = {
            "source_revision": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
            "video_checkpoint": "libero/video_step010000.pt", "action_checkpoint": "libero/action_step030000.pt",
            "dataset_stats_sha256": hashlib.sha256((args.ckpt_dir / "libero/dataset_stats.json").read_bytes()).hexdigest(),
            "policy_action_seed": 42, "policy_video_seed": 43,
            "simulator_seed": yaml.safe_load((ROOT / "benchmarks" / ("libero-plus" if plus else "libero") / "policy_config.yml").read_text())["seed"],
            "replan_steps": 10, "action_horizon": 32, "denoise_steps": 10, "action_shift": 5,
            "tokenizer_max_len": 128, "camera_preprocess_mode": "training_exact",
            "image_size": [224, 448], "normalization": "min/max", "binarize_gripper": True,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "numpy", "mujoco")},
            "adapter_sha256": {name: hashlib.sha256((ROOT / "benchmarks/vla_harness" / name).read_bytes()).hexdigest()
                               for name in ("vpp2_policy.py", "vpp2_model_server.py", "vpp2_batch.py")},
            "inference_backend": "batched-euler", "max_batch_size": args.max_batch_size,
        }
    if plus:
        from benchmarks.vla_harness.libero_plus_protocol import task_catalog

        manifest["libero_plus_python"] = str(args.libero_plus_python)
        manifest["text_cache_preflight"] = text_preflight
        catalog = task_catalog(args.libero_path)
        (args.output_dir / "task_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        manifest["task_catalog_sha256"] = hashlib.sha256(
            (args.output_dir / "task_catalog.json").read_bytes()
        ).hexdigest()
    (args.output_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    if plan.get("dynamic_tasks"):
        from benchmarks.vla_harness.task_queue import initialize_queue
        queue_path = Path(local_db).parent / "task_queue.sqlite"
        initialize_queue(queue_path, plan["tasks"])
        (args.output_dir / "task_queue.sqlite").symlink_to(queue_path)
    handles, owned, servers, clients = [], [], [], []
    monitor = RuntimeMonitor(args.output_dir, recording_db=local_db)

    def start(command, env, filename):
        handle = (args.output_dir / filename).open("w")
        handles.append(handle)
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        owned.append(proc)
        return proc

    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        **({"OPENWAM_DYNAMIC_TASK_QUEUE": str(queue_path.resolve())} if plan.get("dynamic_tasks") else {}),
        # Scope the bundled msgpack fallback to evaluation subprocesses. The
        # native decoder raised KeyError('shape') during the first long run.
        "MSGPACK_PUREPYTHON": "1",
        "OPENWAM_PERFORMANCE_DIR": str(args.output_dir / "performance"),
        "PYTHONFAULTHANDLER": "1",
        "MALLOC_ARENA_MAX": "2",
        "VLA_EVAL_WATCHDOG_TIMEOUT_S": "2400",
    }
    # Workers connect only to local model servers, even when tmux inherits a proxy.
    for key in ("NO_PROXY", "no_proxy"):
        base_env[key] = ",".join(filter(None, [base_env.get(key, ""), "127.0.0.1", "localhost", "::1"]))
    base_env["PYTHONPATH"] = f"{ROOT}:{args.libero_path}:" + base_env.get("PYTHONPATH", "")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        loading_started = time.monotonic()
        pending = {}
        for slot, item in enumerate(plan["servers"]):
            env = {**base_env, "CUDA_VISIBLE_DEVICES": item["gpu"]}
            servers.append(start(item["command"], env, f"model_{slot}.log"))
            pending[slot] = time.monotonic() + 2400
        # Independent GPUs load concurrently; workers start only when every
        # replica is healthy. Keep checking all children while any is loading.
        while pending:
            monitor.sample(owned)
            monitor.check(servers, clients)
            for slot, deadline in list(pending.items()):
                port = plan["servers"][slot]["port"]
                try:
                    with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            del pending[slot]
                            continue
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Model {slot} initialization exceeded 2400 seconds")
            if pending:
                time.sleep(1)
        print(f"Model servers ready in {time.monotonic() - loading_started:.1f}s", flush=True)
        for item in plan["workers"]:
            shard = item["shard"]
            worker_config = config_root / str(shard)
            env = {
                **base_env,
                "OPENWAM_SHARD_ID": str(shard),
                "OPENWAM_NUM_WORKERS": str(len(plan["workers"])),
                "LIBERO_PATH": str(args.libero_path),
                "LIBERO_CONFIG_ROOT": str(worker_config),
                "LIBERO_CONFIG_PATH": str(worker_config),
                "LP_NUM_THREADS": "4",
                "OPENWAM_HARNESS_IMAGE_FORMAT": args.image_format,
            }
            if plus:
                env.update(
                    LIBERO_PLUS_PATH=str(args.libero_path),
                    LIBERO_PLUS_PYTHON=str(args.libero_plus_python),
                    LIBERO_PLUS_CONFIG_ROOT=str(worker_config),
                    NUMEXPR_NUM_THREADS="1",
                )
            if args.render == "cpu":
                env.update(
                    CUDA_VISIBLE_DEVICES="",
                    MUJOCO_GL="osmesa",
                    PYOPENGL_PLATFORM="osmesa",
                    LD_LIBRARY_PATH=str(OSMESA_LIB) + ":" + env.get("LD_LIBRARY_PATH", ""),
                )
            else:
                env.update(
                    CUDA_VISIBLE_DEVICES=item["render_gpu"],
                    MUJOCO_EGL_DEVICE_ID=item["render_gpu"],
                    MUJOCO_GL="egl",
                    PYOPENGL_PLATFORM="egl",
                    EGL_PLATFORM="device",
                )
            clients.append(start(item["command"], env, f"worker_{shard}.log"))
        print(f"Started {len(servers)} batched servers / {len(clients)} harness workers: {args.output_dir}", flush=True)
        while any(p.poll() is None for p in clients):
            monitor.sample(owned)
            monitor.check(servers, clients)
            time.sleep(1)
        monitor.check(servers, clients, force=True)
    finally:
        monitor.sample(owned, force=True)
        # Signal only process groups created by this launcher, never training.
        for proc in owned:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc in owned:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for handle in handles:
            handle.close()
        persist_recording_db(local_db, output_db)
        from vla_eval.results.merge import merge_db

        try:
            # The installed harness merge_db API accepts only the database
            # path and output directory. Step export is handled by the
            # recorder configuration and is not a merge_db argument.
            merge_db(local_db, args.output_dir)
        except FileNotFoundError:
            pass
        result = write_summary(args.output_dir)
    return 0 if result["selected_run_complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_arguments(parser)
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero")
    parser.add_argument("--correct-instruction-suffix", "--correct_instruction_suffix", type=parse_bool, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ckpt-name", default="checkpoint_step_10690.safetensors")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dynamic-tasks", action="store_true", help="Plus only: workers atomically claim whole tasks from a shared queue")
    parser.add_argument("--workers", type=int, help="Environment concurrency (default: Plus 16 per selected GPU; LIBERO 32 total)")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-wait-time", type=float, default=0.01)
    parser.add_argument("--client-action-queue", action="store_true",
                        help="Consume action chunks and maintain per-control-step history in each worker")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--categories", help="LIBERO-Plus only: comma-separated perturbation categories: Objects Layout, Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise (default: all seven)")
    parser.add_argument("--task-ids")
    parser.add_argument("--exclude-tasks-file", type=Path, help="JSON suite/task pairs already completed")
    parser.add_argument("--render", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--render-gpus", help="Physical EGL devices (default: Plus last selected GPU; LIBERO 3)")
    parser.add_argument("--image-format", choices=("raw", "png"), default="raw", help="Lossless image transport")
    parser.add_argument("--libero-path", type=Path, help="Benchmark checkout (default follows --benchmark)")
    parser.add_argument(
        "--libero-plus-python",
        type=Path,
        default=Path(sys.executable).parents[2] / "libero-plus/bin/python",
        help="Official Plus simulator interpreter; harness/model services keep the launcher's Python",
    )
    parser.add_argument("--base-port", type=int, default=8940)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/vla_harness" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument("--export-step-jsonl", action="store_true", help="Record and export per-step traces (default: episode outcomes only for LIBERO and Plus)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-max-steps", type=int, default=20, help="Isolated smoke/profile step limit; never changes formal evaluation")
    parser.add_argument(
        "--smoke", action="store_true", help="1 trial/task, first two task IDs, max 20 policy steps; no score"
    )
    args = parser.parse_args()
    if args.correct_instruction_suffix is None:
        args.correct_instruction_suffix = args.benchmark == "libero-plus"
    if args.correct_instruction_suffix and args.benchmark != "libero-plus":
        parser.error("--correct-instruction-suffix applies only to libero-plus")
    if args.libero_path is None:
        args.libero_path = ROOT.parent / "benchmarks" / (
            "LIBERO-plus" if args.benchmark == "libero-plus" else "LIBERO"
        )
    args.libero_plus_python = args.libero_plus_python.resolve()
    args.eval_id = str(uuid.uuid4())
    args.ckpt_dir, args.output_dir, args.libero_path = (
        args.ckpt_dir.resolve(),
        args.output_dir.resolve(),
        args.libero_path.resolve(),
    )
    plan = make_plan(args)
    if args.dry_run:
        print(
            json.dumps(
                {"python": sys.executable, "render": args.render, "output_dir": str(args.output_dir), "plan": plan},
                indent=2,
            )
        )
        return 0

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    return launch(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
