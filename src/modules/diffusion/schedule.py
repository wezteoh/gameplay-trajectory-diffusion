# Beta schedule and q_sample adapted from generation-with-vqlatents
# (improved-diffusion / guided-diffusion lineage).

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5,
                linear_end**0.5,
                n_timestep,
                dtype=torch.float64,
            )
            ** 2
        )
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas.numpy(), a_min=0, a_max=0.999)
        return betas
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPMNoiseSchedule(nn.Module):
    """Registers sqrt(alpha_bar) terms and implements q(x_t | x_0)."""

    def __init__(
        self,
        timesteps: int,
        beta_schedule: str = "linear",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        super().__init__()
        betas_np = make_beta_schedule(
            beta_schedule,
            n_timestep=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        betas = torch.tensor(betas_np, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        sqrt_recip_alphas_cumprod = torch.rsqrt(alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        sqrt_alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """x_0 from epsilon prediction (eps parameterization)."""
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """One ancestral DDPM step from x_t toward x_{t-1} given model epsilon."""
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        noise = torch.randn_like(x_t)
        mask = (t != 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
        # return posterior_mean
        return posterior_mean + mask * torch.exp(0.5 * log_var) * noise
