"""Matched E00 scratch training on GPUs 0–3. Run baseline, then router, in sibling directories."""
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
import zipfile

if "--preflight" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openwam.experiments import preflight, resolve_experiment, run_identity, runtime_env, validate_resume
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.epoch_sampler import EpochRandomSampler
from openwam.train.utils.seeding import _candidate_samplers, seed_everything
from scripts.train import _build_accelerator, _inject_project_seed

WORLD = 4
EPOCHS = 10
STEPS_PER_EPOCH = 1069
FINAL = EPOCHS * STEPS_PER_EPOCH
MILESTONES = (1000, 5500)
PROFILES = {"baseline": "E00", "router": "E00+-FrozenDense"}


def read(path):
    return json.loads(path.read_text())


def atomic_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def configuration(arm, seed, output):
    cfg, profile, _ = resolve_experiment(PROFILES[arm], seed=seed)
    cfg.training.max_steps = FINAL
    cfg.training.output_path = str(output / "runs")
    cfg.project.wandb.run_name = f"e00_scratch_{arm}_s{seed}"
    _inject_project_seed(cfg)
    if (cfg.training.num_epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps,
        cfg.training.epoch_seeded_sampler, cfg.training.finetune_ckpt_path,
        cfg.training.save_full_states_for_resume) != (EPOCHS, 64, 1, True, None, True):
        raise ValueError("E00 scratch training protocol changed")
    if cfg.training.get("trainable_parameter_prefixes") or not cfg.model.video_backbone.from_scratch:
        raise ValueError("Expected full E00 policy scratch initialization")
    router = cfg.model.architecture.action_router
    if router.enabled != (arm == "router") or (router.enabled and
            (not router.fp32_residual or not router.fp32_gate or router.sparse_weight != 0)):
        raise ValueError("Unexpected router precision or sparsity configuration")
    return cfg, profile


def training_source_digest():
    # Unrelated evaluation and reporting edits can occur during these multi-day runs.
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), ROOT / "scripts/train.py"]
    paths.extend(sorted((ROOT / "openwam").rglob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def matched_protocol(identity):
    protocol = copy.deepcopy(identity)
    del protocol["experiment"]
    del protocol["config"]["project"]["wandb"]["run_name"]
    del protocol["config"]["model"]["architecture"]["action_router"]
    return protocol


def matched(path, evidence):
    if path.exists() and read(path) != evidence:
        raise ValueError(f"Evidence disagrees with the existing record: {path}")
    atomic_json(path, evidence)


def shared_parameters(model):
    digest = hashlib.sha256()
    count = 0
    for name, p in sorted(model.named_parameters()):
        if name.startswith("action_router."):
            continue
        digest.update(json.dumps([name, list(p.shape), str(p.dtype), p.requires_grad]).encode())
        array = p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(array))
        count += 1
    return {"parameter_tensors": count, "sha256": digest.hexdigest()}


def sampler_order(loader, dataset, seed):
    samplers = [s for s in _candidate_samplers(loader) if isinstance(s, EpochRandomSampler)]
    if len(samplers) != 1 or samplers[0].seed != seed or len(loader) != STEPS_PER_EPOCH:
        raise RuntimeError("Prepared loader does not implement the locked E00 sampler")
    # Hash the actual padded, sharded batch order for all ten epochs without decoding images.
    order = hashlib.sha256()
    for epoch in range(EPOCHS):
        loader.set_epoch(epoch)
        if samplers[0].epoch != epoch:
            raise RuntimeError("Epoch did not reach the prepared sampler")
        for batch in loader.batch_sampler:
            order.update(np.asarray(batch, dtype="<i8").tobytes())
    loader.set_epoch(0)
    windows = hashlib.sha256()
    for v in (dataset._eps_df["episode_index"], dataset._ep_valid_start, dataset._cum_n_starts):
        windows.update(np.asarray(v, dtype="<i8").tobytes())
    return {"seed": seed, "windows": len(dataset), "window_map_sha256": windows.hexdigest(),
            "rank_batch_order_sha256": order.hexdigest(), "steps_per_epoch": len(loader)}


def protect_gate_storage(router):
    # Model-wide dtype transitions surround ZeRO flattening. Preserve the scalar's value
    # on device moves and its view into ZeRO's flat buffer on same-device BF16 casts.
    original = router._apply

    def apply(fn, recurse=True):
        previous = router.gate_logit.detach()
        result = original(fn, recurse=recurse)
        router.gate_logit.data = previous.to(device=router.gate_logit.device)
        return result

    router._apply = apply


def preserve(run, step):
    directory = run / "snapshots" / str(step)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "normalization_stats.npy", "manifest.json", f"checkpoint_step_{step}.safetensors"):
        source = run / name
        kept = directory / name
        if not kept.exists() or not os.path.samefile(source, kept):
            temp = kept.with_suffix(".link.tmp")
            temp.unlink(missing_ok=True)
            os.link(source, temp)
            temp.replace(kept)


class SequenceTrainer(OpenWAMTrainer):
    def __init__(self, cfg, accelerator, dataset, args):
        super().__init__(cfg, accelerator=accelerator, dataset=dataset)
        self.args = args
        self.resume_step = 0
        router = self.architecture.action_router
        if router is not None:
            protect_gate_storage(router)
            if not cfg.training.resume_ckpt_path:
                init = float(cfg.model.architecture.action_router.gate_init)
                with torch.no_grad():
                    router.gate_logit.fill_(math.log(init / (1 - init)))

    def build_optimizer(self):
        optimizer = super().build_optimizer()
        router = self.architecture.action_router
        if router is not None:
            # Mixed-dtype torch.flatten would promote all policy weights to FP32.
            gate = router.gate_logit
            group = next(g for g in optimizer.param_groups if any(p is gate for p in g["params"]))
            group["params"] = [p for p in group["params"] if p is not gate]
            optimizer.add_param_group({**group, "params": [gate]})
        return optimizer

    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        optimizer, loader, scheduler = super().prepare_accelerate(optimizer, dataloader, scheduler)
        args = self.args
        model = self.accelerator.unwrap_model(self.architecture)
        sample = sampler_order(loader, self.dataset, args.seed)
        evidence = [None] * WORLD
        torch.distributed.all_gather_object(evidence, sample)
        if self.accelerator.is_main_process:
            matched(args.output / "sampling.json", evidence)
            if args.arm == "router" and read(args.output.parent / "baseline/sampling.json") != evidence:
                raise ValueError("Router and baseline sample orders differ")
            if not self.cfg.training.resume_ckpt_path:
                shared = {"seed": args.seed, **shared_parameters(model)}
                matched(args.output / "shared_initialization.json", shared)
                if args.arm == "router" and read(args.output.parent / "baseline/shared_initialization.json") != shared:
                    raise ValueError("Router and baseline shared parameters differ at step zero")
        router = model.action_router
        if router is not None:
            flat = self.architecture.optimizer.bit16_groups_flat[-1]

            def first_forward(module, inputs, output):
                gate = module.gate_logit
                aux = output[1]
                if (gate.dtype != torch.float32 or gate.data_ptr() != flat.data_ptr()
                        or not module.fp32_residual or module.sparse_weight != 0
                        or aux["router_gate"].dtype != torch.float32 or aux["router_residual_ratio"].dtype != torch.float32):
                    raise RuntimeError("First router forward failed the FP32/ZeRO storage check")
                if self.accelerator.is_main_process:
                    atomic_json(args.output / f"first_forward_{self.resume_step}.json", {
                        "starting_step": self.resume_step, "gate_storage": str(gate.dtype),
                        "gate": gate.detach().sigmoid().item(), "gate_is_zero_optimizer_view": True,
                        "fp32_residual": module.fp32_residual, "sparse_weight": module.sparse_weight,
                        "actions_dtype": str(inputs[0].dtype), "returned_actions_dtype": str(output[0].dtype)})
                hook.remove()

            hook = router.register_forward_hook(first_forward)
        return optimizer, loader, scheduler

    def setup_output_dir(self, debug, resume_path):
        directory, state = super().setup_output_dir(debug, resume_path)
        if state is not None:
            state = str(Path(directory) / f"accel_state_step_{self.cfg.experiment.resume_step}")
        return directory, state

    def resume_if_configured(self, *args):
        result = super().resume_if_configured(*args)
        self.resume_step = result[0]
        return result

    def log_step(self, **kwargs):
        super().log_step(**kwargs)
        previous = kwargs["global_step"] - 1
        if self.args.arm == "router" and self.accelerator.is_main_process and previous in MILESTONES:
            preserve(Path(kwargs["output_path"]), previous)


def latest_run(output):
    candidates = []
    for run in (output / "runs").glob("*/"):
        states = [p for p in run.glob("accel_state_step_*") if p.name.split("_")[-1].isdigit()]
        for state in sorted(states, key=lambda p: int(p.name.split("_")[-1]), reverse=True):
            step = int(state.name.split("_")[-1])
            marker = state / "trainer_state.json"
            if not marker.is_file():
                continue
            try:
                meta = read(marker)
                files = [state / "scheduler.bin", state / "pytorch_model/mp_rank_00_model_states.pt"]
                for rank in range(WORLD):
                    files.extend([state / f"random_states_{rank}.pkl",
                                  state / f"pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"])
                valid = (meta["opt_step"] == meta["global_step"] == step and 0 < step <= FINAL
                         and (state / "latest").read_text().strip() == "pytorch_model"
                         and all(file.is_file() and zipfile.is_zipfile(file) for file in files))
            except (OSError, KeyError, ValueError):
                valid = False
            if valid:
                candidates.append((step, run))
                break
            print(f"Incomplete or unreadable resume state: {state}", file=sys.stderr)
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple resumable runs under {output}")
    return candidates[0] if candidates else (0, None)


def audit_metrics(run, limit, trim=False):
    path = run / "router_metrics.jsonl"
    with path.open() as stream:
        rows = []
        for step in range(1, limit + 1):
            row = stream.readline()
            value = json.loads(row)
            if (value["step"] != step or value["opt_step"] != step or value["router_sparse_weight"] != 0
                    or value["loss_router_sparse"] != 0 or not math.isfinite(value["router_residual_ratio"])
                    or not 0 < value["router_gate"] < 1):
                raise ValueError(f"Unexpected router metrics at step {step}")
            if trim:
                rows.append(row)
        extra = stream.read(1)
    if extra:
        if not trim:
            raise ValueError("Unexpected router metrics beyond the final update")
        temp = path.with_suffix(".jsonl.tmp")
        temp.write_text("".join(rows))
        temp.replace(path)


def publish_index(args, run):
    from safetensors import safe_open
    result = read(run / "training_result.json")
    if result["status"] != "completed" or result["global_step"] != FINAL or result["opt_step"] != FINAL:
        raise RuntimeError(f"Training has not completed {FINAL} updates: {result}")
    for name in ("config.yaml", "manifest.json", "normalization_stats.npy"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    router = args.arm == "router"
    if router:
        audit_metrics(run, FINAL)
        for name in ("shared_initialization.json", "sampling.json"):
            if read(args.output / name) != read(args.output.parent / "baseline" / name):
                raise ValueError(f"Baseline evidence changed: {name}")
        if read(args.output / "first_forward_0.json")["gate_storage"] != "torch.float32":
            raise ValueError("Missing FP32 gate verification")
    paths = {s: run / "snapshots" / str(s) / f"checkpoint_step_{s}.safetensors" for s in MILESTONES if router}
    paths[FINAL] = run / f"checkpoint_step_{FINAL}.safetensors"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    for path in paths.values():
        with safe_open(path, framework="pt", device="cpu") as f:
            if "action_backbone.action_decoder.weight" not in f.keys() or ("action_router.gate_logit" in f.keys()) != router:
                raise ValueError(f"Unexpected checkpoint schema: {path}")
            if router and f.get_tensor("action_router.gate_logit").dtype != torch.float32:
                raise ValueError(f"Checkpoint lost FP32 gate: {path}")
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.is_floating_point():
                    for chunk in tensor.reshape(-1).split(1 << 20):
                        if not torch.isfinite(chunk).all().item():
                            raise ValueError(f"Nonfinite deployment weight: {path}/{name}")
    atomic_json(args.output / "index.json", {str(s): str(path) for s, path in paths.items()})


def worker(args):
    from openwam.dataloader.registry import build_dataset
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = OmegaConf.load(args.output / "worker_config.yaml")
    seed_everything(args.seed, rank=int(os.environ["RANK"]))
    accelerator = _build_accelerator(cfg)
    if accelerator.num_processes != WORLD:
        raise RuntimeError("E00 training requires four ranks")
    dataset = build_dataset(cfg.dataloader, split="train")
    trainer = SequenceTrainer(cfg, accelerator, dataset, args)
    trainer.train()
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


def launch(args):
    cfg, profile = configuration(args.arm, args.seed, args.output)
    cache = preflight(cfg, profile)
    if args.preflight:
        from openwam.dataloader.registry import build_dataset
        ds = build_dataset(cfg.dataloader, split="train")
        assert math.ceil(len(ds) / (WORLD * cfg.training.batch_size)) == STEPS_PER_EPOCH
        print(json.dumps({"arm": args.arm, "seed": args.seed, "windows": len(ds), "updates": FINAL,
                          "router": OmegaConf.to_container(cfg.model.architecture.action_router), "cache": cache}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = run_identity(cfg, profile, WORLD, cache)
        identity["source_sha256"] = training_source_digest()
        identity_path = args.output / "identity.json"
        matched(identity_path, identity)
        if args.arm == "router":
            baseline_identity = read(args.output.parent / "baseline/identity.json")
            if matched_protocol(identity) != matched_protocol(baseline_identity):
                raise ValueError("E00 arm configurations, data, pretrained assets or training code differ")
            for name in ("shared_initialization.json", "sampling.json"):
                if not (args.output.parent / "baseline" / name).is_file():
                    raise ValueError("Run the sibling baseline first for the matched initialization and sampling audit")
        step, run = latest_run(args.output)
        if run:
            validate_resume(run, identity)
            if step == FINAL and (run / "training_result.json").exists() and read(run / "training_result.json")["status"] == "completed":
                publish_index(args, run)
                return
            if args.arm == "router":
                audit_metrics(run, step, trim=True)
            cfg.training.resume_ckpt_path = str(run)
        OmegaConf.update(cfg, "experiment.identity_file", str(identity_path), force_add=True)
        OmegaConf.update(cfg, "experiment.resume_step", step, force_add=True)
        OmegaConf.save(cfg, args.output / "worker_config.yaml", resolve=True)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(Path(__file__).resolve()), "--worker", "--arm", args.arm, "--seed", str(args.seed),
               "--output", str(args.output)]
        env = {**os.environ, **runtime_env("0,1,2,3"), "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        step, run = latest_run(args.output)
        if step != FINAL:
            raise RuntimeError(f"Last complete resumable update is {step}/{FINAL}")
        publish_index(args, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true", help="Check configuration, text cache and dataset on CPU only")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
