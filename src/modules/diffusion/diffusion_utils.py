# Modified from OpenAI / DiT diffusion repos (BSD-style).

from __future__ import annotations

import math

import torch


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(
    mean1: torch.Tensor | float,
    logvar1: torch.Tensor | float,
    mean2: torch.Tensor | float,
    logvar2: torch.Tensor | float,
) -> torch.Tensor:
    """KL divergence between two Gaussians (broadcasting)."""
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1_t, logvar2_t = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x, device=tensor.device, dtype=tensor.dtype)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2_t
        - logvar1_t
        + torch.exp(logvar1_t - logvar2_t)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2_t)
    )


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(
    x: torch.Tensor,
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    """Log-likelihood for a Gaussian discretized to 256 bins ([-1, 1] float image)."""
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(cdf_delta.clamp(min=1e-12)),
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs
