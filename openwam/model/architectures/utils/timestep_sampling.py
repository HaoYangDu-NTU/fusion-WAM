"""Training-only timestep distributions on the existing pre-shift index grid."""

import math

import torch

NOISE_SCHEDULE_DISTRIBUTIONS = ("uniform", "logit_normal", "logit_logistic", "mode")


def validate_noise_schedule_distribution(distribution: str) -> str:
    if distribution not in NOISE_SCHEDULE_DISTRIBUTIONS:
        raise ValueError(
            f"noise_schedule_distribution must be one of {NOISE_SCHEDULE_DISTRIBUTIONS}, got {distribution!r}"
        )
    return distribution


def sample_timestep_ids(
    num_timesteps: int,
    batch_size: int,
    *,
    distribution: str = "uniform",
    low: int = 0,
    high: int | None = None,
) -> torch.Tensor:
    """Sample IDs, leaving shift, sigma lookup and loss weighting to the scheduler.

    New distributions sample a float32 base noise level b, then quantize onto
    the descending pre-shift grid. A restricted [low, high) index window scales
    the distribution to that window; it does not condition/truncate the full
    distribution. Uniform keeps the legacy randint call and RNG consumption.

    Presets: logit-normal(0, 1), logit-logistic(scale=0.75), SD3 mode(scale=1.29).
    All streams draw independently. This helper is never used for inference.
    """
    validate_noise_schedule_distribution(distribution)
    high = num_timesteps if high is None else high
    if not 0 <= low < high <= num_timesteps:
        raise ValueError(f"Invalid timestep ID window [{low}, {high}) for {num_timesteps} timesteps")
    if distribution == "uniform":
        return torch.randint(low, high, (batch_size,))
    if distribution == "logit_normal":
        base = torch.randn((batch_size,), dtype=torch.float32, device="cpu").sigmoid()
    else:
        u = torch.rand((batch_size,), dtype=torch.float32, device="cpu")
        if distribution == "logit_logistic":
            u = u.clamp(1e-6, 1 - 1e-6)
            base = (0.75 * (u.log() - torch.log1p(-u))).sigmoid()
        else:
            base = 1 - u - 1.29 * (torch.cos(math.pi * u / 2).square() - 1 + u)
    ids = low + ((1 - base) * (high - low)).floor().long()
    return ids.clamp_(low, high - 1)
