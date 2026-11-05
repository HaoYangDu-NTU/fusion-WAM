"""Output parameterization for linear flow matching (sigma=1 is noise)."""

import torch


def prediction_to_velocity(prediction, noisy_sample, sigma, prediction_type):
    """Return v unchanged, or convert a clean x prediction to v in FP32.

    Sigma must be scalar or broadcastable to the sample (e.g. B,1,1,1,1).
    JiT uses t=1-sigma, so its (x-z)/(1-t) becomes (z-x)/sigma here.
    The 0.05 denominator floor matches JiT's default t_eps. Training must
    apply the same conversion to clean targets. Inference casts the result
    back to the model dtype before the existing Euler update.
    """
    if prediction_type == "v":
        return prediction
    if prediction_type != "x":
        raise ValueError(f"Unknown prediction_type: {prediction_type!r}")
    denominator = torch.as_tensor(sigma, device=noisy_sample.device, dtype=torch.float32).clamp_min(0.05)
    return (noisy_sample.float() - prediction.float()) / denominator
