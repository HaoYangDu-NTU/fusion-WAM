"""Train the matched seed-42 E00 FP32 Router with a penalty on attention mass outside its top 32."""
import argparse
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys

if "--preflight" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from safetensors import safe_open

from openwam.model.architectures.utils.action_router import ActionVisualRouter
from scripts import e00_sequence_train as seq

SEED = 42
SCHEDULE = dict(sparse_weight=0.01, warmup_steps=1000, ramp_steps=4000)
DENSE = dict(sparse_weight=0.0, warmup_steps=0, ramp_steps=0)


def configure(output):
    cfg, profile = seq.configuration("router", SEED, output)
    cfg.project.wandb.run_name = "e00_scratch_sparse_s42"
    cfg.model.architecture.action_router.update(SCHEDULE)
    profile = {**profile, "id": "E00-FP32-Sparse", "action_router": dict(cfg.model.architecture.action_router)}
    return cfg, profile


def expected_weight(completed_updates):
    return 0.01 * max(0.0, min(1.0, (completed_updates - 1000) / 4000))


def compare_reference_protocol(args, identity, required):
    # The identity's source hash covers the native trainer used by both arms.
    dense_path = args.matched_root / "router/identity.json"
    if not required and not dense_path.is_file():
        return "pending Dense training"
    dense = seq.read(dense_path)
    if seq.matched_protocol(identity) != seq.matched_protocol(dense):
        raise ValueError(f"Training code, data, seed or optimization differ from {dense_path}")
    sparse_router = identity["config"]["model"]["architecture"]["action_router"]
    dense_router = dense["config"]["model"]["architecture"]["action_router"]
    if ({**dense_router, **SCHEDULE} != sparse_router or
            any(dense_router[key] != value for key, value in DENSE.items())):
        raise ValueError("The paired router configurations differ outside the sparsity schedule")
    if required:
        baseline = seq.read(args.matched_root / "baseline/identity.json")
        if seq.matched_protocol(baseline) != seq.matched_protocol(dense):
            raise ValueError("The matched Dense run and its baseline have different provenance")
        for name in ("sampling.json", "shared_initialization.json"):
            if seq.read(args.matched_root / "baseline" / name) != seq.read(args.matched_root / "router" / name):
                raise ValueError(f"The reference arms differ in {name}")
    return "matched Dense identity"


def cpu_router_check(cfg):
    options = dict(cfg.model.architecture.action_router)
    del options["enabled"]
    digest = []
    with torch.random.fork_rng(devices=[]):
        for schedule in (DENSE, SCHEDULE):
            torch.manual_seed(SEED)
            router = ActionVisualRouter(cfg.model.action_backbone.dim, cfg.model.video_backbone.dit_config.dim,
                                        **{**options, **schedule})
            digest.append(seq.shared_parameters(router))
        for n in (0, 999, 1000, 1001, 3000, 4999, 5000, seq.FINAL - 1):
            if not math.isclose(router.coefficient(n), expected_weight(n), abs_tol=1e-12):
                raise ValueError(f"Sparse schedule mismatch after {n} updates")
    if digest[0] != digest[1]:
        raise ValueError("Router initialization depends on the sparsity schedule")
    return digest[0]


class SparseTrainer(seq.SequenceTrainer):
    def __init__(self, cfg, accelerator, dataset, args):
        # Construct the exact Dense seed-42 model first. Only scalar schedule
        # attributes change after construction. The saved config describes Sparse.
        init_cfg = copy.deepcopy(cfg)
        if not cfg.training.resume_ckpt_path:
            init_cfg.model.architecture.action_router.update(DENSE)
        super().__init__(init_cfg, accelerator, dataset, args)
        self.cfg = cfg
        self.architecture.cfg["action_router"].update(SCHEDULE)
        router = self.architecture.action_router
        if not cfg.training.resume_ckpt_path and accelerator.is_main_process:
            self.dense_replay = seq.shared_parameters(router)
        for name, value in SCHEDULE.items():
            setattr(router, name, value)

    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        optimizer, loader, scheduler = seq.OpenWAMTrainer.prepare_accelerate(self, optimizer, dataloader, scheduler)
        args = self.args
        model = self.accelerator.unwrap_model(self.architecture)
        router = model.action_router
        if (router.top_k != 32 or router.sparse_weight != 0.01 or
                (router.warmup_steps, router.ramp_steps) != (1000, 4000)):
            raise RuntimeError("Sparse router did not retain its configured top-k loss")
        rank_order = seq.sampler_order(loader, self.dataset, SEED)
        order = [None] * seq.WORLD
        torch.distributed.all_gather_object(order, rank_order)
        if self.accelerator.is_main_process:
            if order != seq.read(args.matched_root / "router/sampling.json"):
                raise ValueError("Sparse and Dense samplers or LIBERO window maps differ")
            seq.matched(args.output / "sampling.json", order)
            if not self.cfg.training.resume_ckpt_path:
                shared = {"seed": SEED, **seq.shared_parameters(model)}
                if shared != seq.read(args.matched_root / "router/shared_initialization.json"):
                    raise ValueError("Sparse shared weights differ from the recorded Dense initialization")
                actual = seq.shared_parameters(router)
                if self.dense_replay != actual:
                    raise ValueError("Sparse router tensors changed from their Dense initialization")
                seq.matched(args.output / "initialization.json", {
                    "shared": shared, "router_after_zero": actual, "dense_constructor_replay": self.dense_replay,
                    "router_reference": "Same seed and Dense constructor replay. Historical Dense router bytes were not recorded."})

        def first_forward(module, inputs, output):
            aux = output[1]
            flat = self.architecture.optimizer.bit16_groups_flat[-1]
            if (not module.fp32_gate or not module.fp32_residual or module.gate_logit.dtype != torch.float32
                    or module.gate_logit.data_ptr() != flat.data_ptr() or aux["router_gate"].dtype != torch.float32
                    or aux["router_residual_ratio"].dtype != torch.float32):
                raise RuntimeError("Actual sparse forward failed the FP32 gate/ZeRO storage check")
            if self.accelerator.is_main_process:
                seq.atomic_json(args.output / f"first_forward_{self.resume_step}.json", {
                    "starting_step": self.resume_step, "gate_storage": str(module.gate_logit.dtype),
                    "gate_is_zero_optimizer_view": True, "fp32_residual": True,
                    "gate": aux["router_gate"].item(), "top_k": module.top_k, **SCHEDULE})
            handle.remove()

        handle = router.register_forward_hook(first_forward)
        return optimizer, loader, scheduler

    def log_step(self, **kwargs):
        if self.accelerator.is_main_process:
            validate_row(dict(step=kwargs["global_step"], opt_step=kwargs["opt_step"], **kwargs["metrics"]),
                         kwargs["global_step"])
        super().log_step(**kwargs)


def validate_row(row, step):
    weight = expected_weight(step - 1)  # the native trainer supplies completed optimizer steps before this update
    raw, mass = row["router_sparse_raw"], row["router_topk_mass"]
    if (row["step"] != step or row["opt_step"] != step or not 0 < row["router_gate"] < 1
            or not math.isfinite(row["router_residual_ratio"]) or not -1e-5 <= raw <= 1.00001
            or not -1e-5 <= mass <= 1.00001 or not math.isfinite(row["loss_total"])
            or not math.isclose(row["router_sparse_weight"], weight, rel_tol=0.004, abs_tol=1e-10)
            or not math.isclose(row["loss_router_sparse"], weight * raw, rel_tol=1e-5, abs_tol=1e-9)):
        raise ValueError(f"Invalid sparse metrics or schedule at update {step}")


def audit_metrics(run, step, trim=False):
    path = run / "router_metrics.jsonl"
    with path.open("r+b" if trim else "rb") as stream:
        for n in range(1, step + 1):
            validate_row(json.loads(stream.readline()), n)
        end = stream.tell()
        if stream.read(1):
            if not trim:
                raise ValueError("Unexpected metrics beyond the final update")
            stream.truncate(end)


def publish(args, run):
    result = seq.read(run / "training_result.json")
    if result["status"] != "completed" or result["opt_step"] != seq.FINAL or result["global_step"] != seq.FINAL:
        raise ValueError(f"Training incomplete: {result}")
    audit_metrics(run, seq.FINAL)
    cfg = OmegaConf.load(run / "config.yaml")
    router = cfg.model.architecture.action_router
    if (not router.fp32_gate or not router.fp32_residual or router.top_k != 32 or
            any(router[k] != v for k, v in SCHEDULE.items())):
        raise ValueError("Deploy config does not describe the sparse arm")
    init = seq.read(args.output / "initialization.json")
    if (init["router_after_zero"] != init["dense_constructor_replay"] or
            init["shared"] != seq.read(args.matched_root / "router/shared_initialization.json") or
            seq.read(args.output / "sampling.json") != seq.read(args.matched_root / "router/sampling.json") or
            not seq.read(args.output / "first_forward_0.json")["gate_is_zero_optimizer_view"]):
        raise ValueError("Missing matched initialization, sampler, or FP32 forward evidence")
    for name in ("manifest.json", "normalization_stats.npy"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    path = run / f"checkpoint_step_{seq.FINAL}.safetensors"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    with safe_open(path, framework="pt", device="cpu") as f:
        if "action_backbone.action_decoder.weight" not in f.keys() or f.get_tensor("action_router.gate_logit").dtype != torch.float32:
            raise ValueError("Unexpected final checkpoint schema or gate precision")
        for name in f.keys():
            tensor = f.get_tensor(name)
            if tensor.is_floating_point():
                for chunk in tensor.reshape(-1).split(1 << 20):
                    if not torch.isfinite(chunk).all().item():
                        raise ValueError(f"Nonfinite final weight: {name}")
    seq.atomic_json(args.output / "index.json", {str(seq.FINAL): str(path)})


def worker(args):
    from openwam.dataloader.registry import build_dataset
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = OmegaConf.load(args.output / "worker_config.yaml")
    seq.seed_everything(SEED, rank=int(os.environ["RANK"]))
    accelerator = seq._build_accelerator(cfg)
    if accelerator.num_processes != seq.WORLD:
        raise RuntimeError("E00 training requires four ranks")
    SparseTrainer(cfg, accelerator, build_dataset(cfg.dataloader, split="train"), args).train()
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


def launch(args):
    cfg, profile = configure(args.output)
    cache = seq.preflight(cfg, profile)
    identity = seq.run_identity(cfg, profile, seq.WORLD, cache)
    identity["source_sha256"] = seq.training_source_digest()
    reference_status = compare_reference_protocol(args, identity, required=not args.preflight)
    identity["sparse_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity["matched_root"] = str(args.matched_root)
    if args.preflight:
        from openwam.dataloader.registry import build_dataset
        ds = build_dataset(cfg.dataloader, split="train")
        if math.ceil(len(ds) / (seq.WORLD * cfg.training.batch_size)) != seq.STEPS_PER_EPOCH:
            raise ValueError("Dataset no longer yields the matched E00 budget")
        print(json.dumps({"seed": SEED, "updates": seq.FINAL, "windows": len(ds), "cache": cache,
                          "matched_reference": reference_status, "top_k": 32, **SCHEDULE,
                          "equal_router_cpu_seed_fixture": cpu_router_check(cfg)}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = args.output / "identity.json"
        seq.matched(identity_path, identity)
        step, run = seq.latest_run(args.output)
        if run:
            seq.validate_resume(run, identity)
            if step == seq.FINAL and (run / "training_result.json").is_file() and seq.read(run / "training_result.json")["status"] == "completed":
                publish(args, run)
                return
            audit_metrics(run, step, trim=True)
            cfg.training.resume_ckpt_path = str(run)
        OmegaConf.update(cfg, "experiment.identity_file", str(identity_path), force_add=True)
        OmegaConf.update(cfg, "experiment.resume_step", step, force_add=True)
        OmegaConf.save(cfg, args.output / "worker_config.yaml", resolve=True)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(Path(__file__).resolve()), "--worker", "--output", str(args.output),
               "--matched-root", str(args.matched_root)]
        env = {**os.environ, **seq.runtime_env("0,1,2,3"), "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        step, run = seq.latest_run(args.output)
        if step != seq.FINAL:
            raise RuntimeError(f"Last complete resumable update: {step}/{seq.FINAL}")
        publish(args, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True, help="Seed-42 parent of baseline/ and router/")
    parser.add_argument("--preflight", action="store_true", help="Validate CPU configuration, data and initialization")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.matched_root = args.matched_root.resolve()
    if args.output == args.matched_root or any(args.output.is_relative_to(args.matched_root / arm) for arm in ("router", "baseline")):
        raise ValueError("Sparse output must have its own directory")
    args.arm = "sparse"
    args.seed = SEED
    if args.worker:
        worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
