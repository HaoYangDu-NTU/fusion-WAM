"""Standalone S-VAE trainer (data-parallel via Accelerate + torchrun).

Reads the feature shards + per-channel stats written by
``scripts/svae_train/collect_svae_features.py``, fits the S-VAE reconstruction + KL
objective, and writes ``svae.pt`` for the encoder's ``svae_path`` to load:

    {"format_version": 2,
     "model_config":  {input_dim, latent_dim, num_heads, num_layers, intermediate_size, dropout},
     "state_dict":    <SVAE weights incl. input_mean/std buffers>,
     "metrics":       [...]}

The feature cache is read through a memory-mapped, lazily-indexed dataset
(``_ShardedFeatureDataset``) so a full-dataset collection never has to fit in
host RAM — only the touched pages of each shard are paged in on demand.

The S-VAE is small (a few M params) and fully replicated, so plain DDP is the
right tool — no DeepSpeed/ZeRO. We use HuggingFace Accelerate exactly like
``scripts/train.py``: launched under ``torchrun`` it scales from 1 GPU to many
nodes with no code change (and runs single-process when invoked as plain
``python``). Launch via ``scripts/svae_train/train_svae.sh``:

    cd /path/to/workspace/openwam/openwam-feat-encoder-svae
    bash scripts/svae_train/train_svae.sh train.features_dir=/path/to/<encoder>_<ts>
"""

from __future__ import annotations

import bisect
import datetime
import glob
import logging
import math
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


class _EarlyStopping:
    """Compare consecutive full, post-KL-warmup epochs by reconstruction loss."""

    def __init__(self, patience: int = 3, min_delta: float = 0.01) -> None:
        if patience < 1 or not 0 <= min_delta < 1:
            raise ValueError("early stopping requires patience >= 1 and 0 <= min_delta < 1")
        self.patience = patience
        self.min_delta = min_delta
        self.previous: float | None = None
        self.bad_epochs = 0

    def update(self, reconstruction: float) -> tuple[bool, float | None]:
        if not math.isfinite(reconstruction):
            raise ValueError("Non-finite epoch reconstruction loss")
        improvement = None
        if self.previous is not None:
            improvement = (self.previous - reconstruction) / max(abs(self.previous), 1e-12)
            # Equality counts as a sufficient (>= 1%) improvement.
            sufficient = reconstruction <= self.previous - self.min_delta * max(abs(self.previous), 1e-12)
            self.bad_epochs = 0 if sufficient else self.bad_epochs + 1
        self.previous = reconstruction
        return self.bad_epochs >= self.patience, improvement


class _ShardedFeatureDataset(Dataset):
    """Index shard metadata, then map/copy/unmap one requested clip at a time.

    The returned fp32 sample owns its storage. Clean feature pages are advised
    away after each read, so global shuffling does not retain a dataset-sized
    file cache. The sample values and Accelerate sampling order are unchanged.
    """

    def __init__(self, features_dir: Path) -> None:
        shards = sorted(glob.glob(str(features_dir / "features_rank*_part*.pt")))
        if not shards:
            raise FileNotFoundError(
                f"No features_rank*_part*.pt shards under {features_dir}. Run scripts/svae_train/collect_svae_features.py first."
            )
        self._parts: list[str] = []
        self._cum: list[int] = []  # cumulative clip counts (exclusive-end per part)
        self.raw_dim: int | None = None
        self.use_avg_t_pooling: bool | None = None
        total = 0
        for s in shards:
            d = torch.load(s, map_location="cpu", mmap=True)
            shard_dim = int(d["raw_dim"])
            shard_pooling = d.get("use_avg_t_pooling", None)
            if shard_pooling is not None:
                shard_pooling = bool(shard_pooling)
                if self.use_avg_t_pooling is None:
                    self.use_avg_t_pooling = shard_pooling
                elif shard_pooling != self.use_avg_t_pooling:
                    raise ValueError(
                        f"use_avg_t_pooling mismatch across feature shards: {s} has "
                        f"{shard_pooling}, expected {self.use_avg_t_pooling}."
                    )
            # All shards must agree on raw_dim — they come from one collection
            # run. Fail fast (naming the offending shard) instead of letting a
            # later stack raise an opaque channel-dim error.
            if self.raw_dim is None:
                self.raw_dim = shard_dim
            elif shard_dim != self.raw_dim:
                raise ValueError(
                    f"raw_dim mismatch across feature shards: {s} has raw_dim={shard_dim}, "
                    f"expected {self.raw_dim}. All shards must come from a single collection run."
                )
            feats = d["features"]
            n = int(feats.shape[0])
            if n == 0:
                continue
            self._parts.append(s)
            total += n
            self._cum.append(total)
        if not self._parts:
            raise RuntimeError(f"All feature shards under {features_dir} are empty.")
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        pi = bisect.bisect_right(self._cum, idx)
        base = self._cum[pi - 1] if pi > 0 else 0
        path = self._parts[pi]
        # Map only this sample's shard. The returned tensor MUST own its memory,
        # including for fp32 caches, before the mmap is released.
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        sample = payload["features"][idx - base].to(dtype=torch.float32, copy=True)
        del payload
        # Global random sampling has almost no short-term shard reuse. Retaining
        # all touched file pages made the 1.7-TiB cache thrash the 800-GiB cgroup.
        # Advise only this feature file; never drop unrelated host caches.
        if hasattr(os, "posix_fadvise"):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        return sample

    def close(self) -> None:
        """Release shard metadata after all DataLoader workers exit."""
        self._parts.clear()


def _init_wandb(cfg: DictConfig, accelerator):
    """Init a wandb run on the main process only; mirrors ``openwam_trainer``.

    Reads ``cfg.train.wandb`` (project / run_name / entity). Disabled — returns
    ``None`` — when the block or ``project`` is absent, so ``project: null`` on
    the CLI turns logging off. Graceful no-op (warn) if wandb is not installed.
    The fully-resolved training config is attached for run provenance.
    """
    if not accelerator.is_main_process:
        return None
    wandb_cfg = cfg.train.get("wandb", None)
    if wandb_cfg is None:
        return None
    project = wandb_cfg.get("project", None)
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None
    run = wandb.init(
        project=str(project),
        name=wandb_cfg.get("run_name", None),
        entity=wandb_cfg.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info("wandb initialized: %s/%s", project, run.name)
    return run


def _latent_diagnostics(out: dict, *, active_kl_threshold: float = 1e-2) -> dict:
    """Cheap posterior-health metrics from one forward's ``mu`` / ``logvar``.

    Computed on the LOCAL rank-0 batch (a monitor, not reduced across ranks —
    these are means over B*T*H*W tokens, already statistically stable). The key
    signal is ``active_units``: the count of latent dims whose per-dim KL clears
    ``active_kl_threshold`` (the standard posterior-collapse probe). For a
    healthy 48-d S-VAE you want it near 48, not drifting toward 0. ``mu_abs_mean``
    / ``logvar_mean`` track posterior spread (logvar -> 0 means q collapses to
    the N(0,1) prior).
    """
    mu, logvar = out["mu"], out["logvar"]
    c_lat = mu.shape[1]
    mu_f = mu.detach().permute(0, 2, 3, 4, 1).reshape(-1, c_lat).float()
    lv_f = logvar.detach().permute(0, 2, 3, 4, 1).reshape(-1, c_lat).float()
    kl_per_dim = (-0.5 * (1.0 + lv_f - mu_f.pow(2) - lv_f.exp())).mean(dim=0)
    return {
        "active_units": int((kl_per_dim > active_kl_threshold).sum().item()),
        "mu_abs_mean": float(mu_f.abs().mean().item()),
        "logvar_mean": float(lv_f.mean().item()),
    }


@hydra.main(
    version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="model/video_backbone/encoder/svae/train"
)
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from openwam.model.video_backbone.encoder.svae import _CHECKPOINT_FORMAT_VERSION, SVAE, svae_loss

    tr = cfg.train
    # Accelerate reads RANK/WORLD_SIZE/LOCAL_RANK from torchrun (DDP) and is a
    # no-op single process otherwise. mixed_precision drives the autocast dtype.
    accelerator = Accelerator(mixed_precision=str(tr.get("mixed_precision", "no")))
    set_seed(int(tr.seed))

    features_dir = Path(str(tr.features_dir))
    dataset = _ShardedFeatureDataset(features_dir)  # mmap'd, not materialised in RAM
    stats = torch.load(str(features_dir / "stats.pt"), map_location="cpu")
    mean, std = stats["mean"], stats["std"]
    raw_dim = int(stats.get("raw_dim", dataset.raw_dim))
    feature_pooling = stats.get("use_avg_t_pooling", dataset.use_avg_t_pooling)
    requested_pooling = tr.get("use_avg_t_pooling", None)
    if isinstance(requested_pooling, str):
        normalized = requested_pooling.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            requested_pooling = True
        elif normalized in {"false", "0", "no", "off"}:
            requested_pooling = False
        else:
            raise ValueError("train.use_avg_t_pooling must be true or false when specified")
    if requested_pooling is not None:
        requested_pooling = bool(requested_pooling)
    if feature_pooling is not None:
        feature_pooling = bool(feature_pooling)
    if requested_pooling is not None and feature_pooling is not None and requested_pooling != feature_pooling:
        raise ValueError(
            "S-VAE feature cache pooling mode does not match train.use_avg_t_pooling: "
            f"cache={feature_pooling}, requested={requested_pooling}. "
            "Collect features and train S-VAE with the same flag."
        )
    use_avg_t_pooling = requested_pooling if requested_pooling is not None else feature_pooling
    # Old feature caches predate this metadata and used the legacy pooled path.
    svae_pooling = True if use_avg_t_pooling is None else bool(use_avg_t_pooling)

    if accelerator.is_main_process:
        print("=" * 60)
        print("OpenWAM S-VAE training")
        print(
            f"num_clips={len(dataset)} raw_dim={raw_dim} "
            f"world_size={accelerator.num_processes} mixed_precision={accelerator.mixed_precision}"
        )
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)

    wandb_run = _init_wandb(cfg, accelerator)  # main-process only; None when disabled/uninstalled

    model_config = dict(
        input_dim=raw_dim,
        latent_dim=int(cfg.svae.latent_dim),
        num_heads=int(cfg.svae.num_heads),
        num_layers=int(cfg.svae.num_layers),
        intermediate_size=int(cfg.svae.intermediate_size),
        dropout=float(cfg.svae.dropout),
        use_avg_t_pooling=svae_pooling,
    )
    svae = SVAE(**model_config)
    # Install standardisation stats BEFORE prepare(): DDP broadcasts buffers from
    # rank 0 at wrap time, and every rank loaded the same global stats.pt anyway.
    svae.set_input_stats(mean, std)

    loader = DataLoader(
        dataset,
        batch_size=int(tr.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(tr.get("num_workers", 4)),
        pin_memory=True,
    )
    opt = torch.optim.AdamW(
        svae.parameters(), lr=float(tr.lr), betas=tuple(tr.betas), weight_decay=float(tr.weight_decay)
    )

    svae, opt, loader = accelerator.prepare(svae, opt, loader)

    steps_per_epoch = max(1, len(loader))  # per-rank batches (Accelerate pads to equal length)
    total_steps = steps_per_epoch * int(tr.num_epochs)
    if tr.get("max_steps") is not None:
        total_steps = min(total_steps, int(tr.max_steps))
    kl_warmup_steps = int(float(tr.kl_warmup_ratio) * total_steps)

    # LR schedule: LinearLR warmup over warmup_epochs, then cosine decay to
    # lr * min_lr_ratio. Stepped once per optimizer step on every rank (NOT
    # prepared through Accelerate, so it advances exactly once/step); ranks stay
    # in lock-step because Accelerate pads each shard to equal length. Skip the
    # tick when an fp16 GradScaler skipped the optimizer step.
    lr_warmup_steps = max(1, steps_per_epoch * int(tr.get("warmup_epochs", 1)))
    scheduler = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=lr_warmup_steps),
            CosineAnnealingLR(
                opt,
                T_max=max(total_steps - lr_warmup_steps, 1),
                eta_min=float(tr.lr) * float(tr.get("min_lr_ratio", 0.9)),
            ),
        ],
        milestones=[lr_warmup_steps],
    )

    metrics: list[dict] = []
    epoch_metrics: list[dict] = []
    early_cfg = tr.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_stop = _EarlyStopping(
        patience=int(early_cfg.get("patience", 3)), min_delta=float(early_cfg.get("min_delta", 0.01))
    )
    save_every = int(tr.get("save_every_epochs", 10))
    if save_every < 1:
        raise ValueError("save_every_epochs must be >= 1")
    out_dir = None
    if accelerator.is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(str(tr.output_dir)) / f"svae_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    step = 0
    t0 = time.time()
    done = False
    stop_reason = "completed"

    def save_checkpoint(filename: str, epoch: int, reason: str) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(svae)
            payload = {
                "format_version": _CHECKPOINT_FORMAT_VERSION,
                "model_config": unwrapped.config_dict(),
                "state_dict": unwrapped.state_dict(),
                "metrics": metrics,
                "epoch_metrics": epoch_metrics,
                "raw_dim": raw_dim,
                "use_avg_t_pooling": use_avg_t_pooling,
                "num_clips": len(dataset),
                "world_size": accelerator.num_processes,
                "wallclock_seconds": time.time() - t0,
                "training_state": {
                    "step": step, "epoch": epoch, "total_steps": total_steps,
                    "kl_warmup_steps": kl_warmup_steps, "stop_reason": reason,
                    "early_stopping": {"enabled": early_enabled, **vars(early_stop)},
                    "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                },
            }
            out_path = out_dir / filename
            temporary = out_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(out_path)
            logger.info("S-VAE checkpoint written to: %s", out_path)
        accelerator.wait_for_everyone()

    svae.train()
    for _epoch in range(int(tr.num_epochs)):
        if done:
            break
        epoch_start_step = step
        # Device-side accumulation; one collective at epoch end, no per-step sync.
        epoch_sums = torch.zeros(5, device=accelerator.device, dtype=torch.float64)
        for batch in loader:  # prepared loader already moves the batch to device
            beta_t = float(tr.beta) * (min(1.0, step / kl_warmup_steps) if kl_warmup_steps > 0 else 1.0)
            with accelerator.autocast():
                # Features arrive fp32 from the dataset; the explicit .float() is
                # a no-op guard so the matmul input matches the fp32 module
                # weights when mixed_precision=no (autocast is a no-op then).
                out = svae(batch.float())
                loss, stat = svae_loss(
                    out,
                    beta=beta_t,
                    free_bits_per_dim=float(tr.free_bits_per_dim),
                    cos_weight=float(tr.cos_weight),
                )
            epoch_sums[:4] += torch.stack([stat[k].detach() for k in ("loss", "mse", "cos", "kl")]) * batch.shape[0]
            epoch_sums[4] += batch.shape[0]
            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            # Keep the pre-clip total norm to log (gradients are already
            # all-reduced by DDP, so this is the global value; .item() is
            # deferred to the log step to avoid a per-step device sync).
            grad_norm = accelerator.clip_grad_norm_(svae.parameters(), float(tr.grad_clip))
            opt.step()
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
            step += 1
            if step % int(tr.log_every) == 0 or step == 1:
                # reduce(mean) across ranks for an accurate GLOBAL metric. This is
                # a collective, so it must run on every rank — the step counter is
                # synchronised because Accelerate pads each rank's shard to equal
                # length, so all ranks hit this branch together.
                red = {
                    k: accelerator.reduce(stat[k].detach().to(accelerator.device), reduction="mean")
                    for k in ("loss", "mse", "cos", "kl")
                }
                if accelerator.is_main_process:
                    elapsed = time.time() - t0
                    diag = _latent_diagnostics(out)  # local rank-0 posterior health
                    gn = float(grad_norm) if grad_norm is not None else float("nan")
                    rec = {
                        "step": step,
                        "beta": beta_t,
                        "lr": opt.param_groups[0]["lr"],
                        "grad_norm": gn,
                        "cos_sim": 1.0 - float(red["cos"]),  # reconstruction cosine, ->1 is better
                        **{k: float(red[k]) for k in red},
                        **diag,
                    }
                    metrics.append(rec)
                    logger.info(
                        "step %d/%d | loss %.4f | mse %.4f | cos %.4f | kl %.4f | "
                        "au %d | gnorm %.3f | beta %.2e | lr %.2e | %.1fs",
                        step,
                        total_steps,
                        rec["loss"],
                        rec["mse"],
                        rec["cos"],
                        rec["kl"],
                        int(diag["active_units"]),
                        gn,
                        beta_t,
                        rec["lr"],
                        elapsed,
                    )
                    if wandb_run is not None:
                        clips_per_s = (step * int(tr.batch_size) * accelerator.num_processes) / max(elapsed, 1e-6)
                        wandb_run.log(
                            {
                                "train/loss": rec["loss"],
                                "train/mse": rec["mse"],
                                "recon/cos_loss": rec["cos"],
                                "recon/cos_sim": rec["cos_sim"],
                                "kl/value": rec["kl"],
                                "kl/active_units": diag["active_units"],
                                "latent/mu_abs_mean": diag["mu_abs_mean"],
                                "latent/logvar_mean": diag["logvar_mean"],
                                "opt/lr": rec["lr"],
                                "opt/beta": beta_t,
                                "opt/grad_norm": gn,
                                "perf/clips_per_s": clips_per_s,
                            },
                            step=step,
                        )
            if tr.get("max_steps") is not None and step >= int(tr.max_steps):
                done = True
                stop_reason = "max_steps"
                break

        sums = accelerator.reduce(epoch_sums, reduction="sum").cpu().tolist()
        epoch_rec = {k: sums[i] / sums[4] for i, k in enumerate(("loss", "mse", "cos", "kl"))}
        if not all(math.isfinite(v) for v in epoch_rec.values()):
            raise ValueError("Non-finite epoch training metrics")
        reconstruction = epoch_rec["mse"] + float(tr.cos_weight) * epoch_rec["cos"]
        full_epoch = step - epoch_start_step == steps_per_epoch
        monitored = early_enabled and full_epoch and epoch_start_step >= kl_warmup_steps
        should_stop, improvement = early_stop.update(reconstruction) if monitored else (False, None)
        epoch_rec.update(epoch=_epoch + 1, step=step, reconstruction=reconstruction,
                         monitored=monitored, full_epoch=full_epoch,
                         relative_improvement=improvement, bad_epochs=early_stop.bad_epochs)
        epoch_metrics.append(epoch_rec)
        if accelerator.is_main_process:
            logger.info("epoch %d | reconstruction %.6f | monitored %s | relative_improvement %s | bad_epochs %d/%d",
                        _epoch + 1, reconstruction, monitored, improvement, early_stop.bad_epochs, early_stop.patience)
            if wandb_run is not None:
                wandb_run.log({f"epoch/{k}": v for k, v in epoch_rec.items() if v is not None}, step=step)
        if should_stop:
            done = True
            stop_reason = "early_stopping"
            if accelerator.is_main_process:
                logger.info("Early stopping at epoch %d: %d consecutive improvements below %.2f%%",
                            _epoch + 1, early_stop.patience, early_stop.min_delta * 100)
        if full_epoch and (_epoch + 1) % save_every == 0:
            save_checkpoint(f"svae_epoch_{_epoch + 1:04d}.pt", _epoch + 1, "periodic")

    save_checkpoint("svae.pt", epoch_metrics[-1]["epoch"], stop_reason)
    # Every rank must unmap before any node evicts shared feature file pages.
    # The non-persistent DataLoader workers have exited with their iterators.
    dataset.close()
    accelerator.wait_for_everyone()
    from scripts.svae_train.feature_cache import release_feature_cache

    cache_errors = 0
    if accelerator.is_local_main_process:
        cache_errors = len(release_feature_cache(features_dir)["errors"])
    cache_errors = accelerator.reduce(
        torch.tensor(cache_errors, device=accelerator.device), reduction="sum"
    ).item()
    if cache_errors:
        raise RuntimeError("S-VAE checkpoint saved, but feature cache release failed; see [svae-cache] logs")
    if accelerator.is_main_process:
        out_path = out_dir / "svae.pt"
        print()
        print(f"S-VAE checkpoint written to: {out_path}")
        if metrics:
            first, last = metrics[0], metrics[-1]
            print(
                f"loss {first['loss']:.4f} -> {last['loss']:.4f} | "
                f"mse {first['mse']:.4f} -> {last['mse']:.4f} | "
                f"kl {first['kl']:.4f} -> {last['kl']:.4f} over {step} steps"
            )
        if wandb_run is not None:
            wandb_run.summary["svae_path"] = str(out_path)
            wandb_run.finish()


if __name__ == "__main__":
    main()
