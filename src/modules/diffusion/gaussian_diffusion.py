# Beta schedule and DiT-style Gaussian diffusion (facebookresearch/DiT gaussian_diffusion.py lineage).

from __future__ import annotations

import enum
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from src.modules.diffusion.diffusion_utils import (
    discretized_gaussian_log_likelihood,
    mean_flat,
    normal_kl,
)


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
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas.numpy(), a_min=0, a_max=0.999)
        return betas
    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]
) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def ddim_timestep_sequence(num_timesteps: int, inference_steps: int) -> list[int]:
    """Build a strictly decreasing list of training timestep indices for DDIM.

    ``inference_steps`` is the number of denoising updates (transitions). The
    returned list has length >= 2, spans toward 0, and is suitable for pairing
    consecutive entries ``(t_cur, t_next)`` with ``t_cur > t_next``.
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    if inference_steps < 1:
        raise ValueError(f"inference_steps must be >= 1, got {inference_steps}")
    n = int(num_timesteps)
    s = min(int(inference_steps), n)
    raw = np.linspace(n - 1, 0, s + 1)
    pts = np.unique(np.round(raw).astype(np.int64))
    pts = np.clip(pts, 0, n - 1)
    descending = np.sort(pts)[::-1].tolist()
    if len(descending) < 2:
        descending = [n - 1, 0]
    return descending


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self) -> bool:
        return self == LossType.KL or self == LossType.RESCALED_KL


def _mean_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    if kind == "l2":
        return mean_flat((target - pred) ** 2)
    if kind == "l1":
        return mean_flat((target - pred).abs())
    raise ValueError(f"Unknown mean loss kind: {kind}")


class GaussianDiffusion(nn.Module):
    """Training + sampling utilities; registers same buffer names as former DDPMNoiseSchedule."""

    def __init__(
        self,
        timesteps: int,
        beta_schedule: str = "linear",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        *,
        legacy_posterior_log_variance: bool = False,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
        loss_type: LossType = LossType.MSE,
        clip_denoised: bool = False,
        coord_channels: int = 2,
    ) -> None:
        super().__init__()
        self.legacy_posterior_log_variance = bool(legacy_posterior_log_variance)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.clip_denoised = bool(clip_denoised)
        self.coord_channels = int(coord_channels)

        betas_np = make_beta_schedule(
            beta_schedule,
            n_timestep=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        betas = torch.tensor(betas_np, dtype=torch.float32)
        # Only `betas` is checkpointed; all other terms are recomputed (matches old DDPMNoiseSchedule
        # checkpoints and avoids missing-key errors for buffers added in GaussianDiffusion).
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer(
            "alphas_cumprod", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.empty(len(betas), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "alphas_cumprod_next",
            torch.empty(len(betas), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "posterior_variance", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "posterior_log_variance_clipped", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "posterior_mean_coef1", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "posterior_mean_coef2", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "fixed_large_variance", torch.empty_like(betas), persistent=False
        )
        self.register_buffer(
            "fixed_large_log_variance", torch.empty_like(betas), persistent=False
        )
        self._rebuild_buffers_from_betas()

    def _rebuild_buffers_from_betas(self) -> None:
        """Recompute all derived tensors from ``self.betas`` (checkpoint-safe)."""
        betas = self.betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [
                torch.ones(1, dtype=torch.float32, device=betas.device),
                alphas_cumprod[:-1],
            ]
        )
        alphas_cumprod_next = torch.cat(
            [
                alphas_cumprod[1:],
                torch.zeros(1, dtype=torch.float32, device=betas.device),
            ]
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        if self.legacy_posterior_log_variance:
            posterior_log_variance_clipped = torch.log(
                torch.clamp(posterior_variance, min=1e-20)
            )
        else:
            if posterior_variance.numel() > 1:
                pv = posterior_variance
                stitched = torch.cat([pv[1:2], pv[1:]], dim=0)
                posterior_log_variance_clipped = torch.log(stitched)
            else:
                posterior_log_variance_clipped = torch.log(
                    torch.clamp(posterior_variance, min=1e-20)
                )

        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        sqrt_recip_alphas_cumprod = torch.rsqrt(alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)

        if betas.numel() > 1:
            fixed_large_var = torch.cat([posterior_variance[1:2], betas[1:]], dim=0)
            fixed_large_log_var = torch.log(fixed_large_var)
        else:
            fixed_large_var = posterior_variance
            fixed_large_log_var = torch.log(torch.clamp(posterior_variance, min=1e-20))

        self.alphas_cumprod.copy_(alphas_cumprod)
        self.alphas_cumprod_prev.copy_(alphas_cumprod_prev)
        self.alphas_cumprod_next.copy_(alphas_cumprod_next)
        self.sqrt_alphas_cumprod.copy_(torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod.copy_(torch.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod.copy_(sqrt_recip_alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod.copy_(sqrt_recipm1_alphas_cumprod)
        self.log_one_minus_alphas_cumprod.copy_(log_one_minus_alphas_cumprod)
        self.posterior_variance.copy_(posterior_variance)
        self.posterior_log_variance_clipped.copy_(posterior_log_variance_clipped)
        self.posterior_mean_coef1.copy_(posterior_mean_coef1)
        self.posterior_mean_coef2.copy_(posterior_mean_coef2)
        self.fixed_large_variance.copy_(fixed_large_var)
        self.fixed_large_log_variance.copy_(fixed_large_log_var)

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> Any:
        # Ignore legacy keys under `schedule.*` that we re-derive from `betas`.
        filtered = {k: v for k, v in state_dict.items() if k == "betas"}
        if not filtered:
            raise KeyError(
                "GaussianDiffusion checkpoint must contain key 'betas' "
                f"(got keys: {list(state_dict.keys())})"
            )
        out = super().load_state_dict(filtered, strict=False)
        self._rebuild_buffers_from_betas()
        return out

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def q_mean_variance(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_start_from_noise(x_t, t, eps)

    def _predict_eps_from_xstart(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_xstart: torch.Tensor,
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(
        self,
        model: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if model_kwargs is None:
            model_kwargs = {}
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        model_output = model(x, t, **model_kwargs)
        c = self.coord_channels

        if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
            assert (
                model_output.shape[-1] == c * 2
            ), f"learned variance expects last dim {c*2}, got {model_output.shape[-1]}"
            model_output, model_var_values = torch.split(model_output, c, dim=-1)
            min_log = extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = extract_into_tensor(torch.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                model_variance = extract_into_tensor(
                    self.fixed_large_variance, t, x.shape
                )
                model_log_variance = extract_into_tensor(
                    self.fixed_large_log_variance, t, x.shape
                )
            else:
                model_variance = extract_into_tensor(
                    self.posterior_variance, t, x.shape
                )
                model_log_variance = extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )

        def process_xstart(x0: torch.Tensor) -> torch.Tensor:
            if denoised_fn is not None:
                x0 = denoised_fn(x0)
            if clip_denoised:
                return x0.clamp(-1, 1)
            return x0

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        model: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x.shape) - 1)))
        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample(
        self,
        model: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eta: float = 0.0,
        cond_fn: Callable[..., Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``x_{t-1}`` from the model using DDIM (DiT / OpenAI API).

        Matches ``ddim_sample`` in
        https://github.com/facebookresearch/DiT/blob/main/diffusion/gaussian_diffusion.py
        """
        if cond_fn is not None:
            raise NotImplementedError(
                "cond_fn is not supported; use classifier-free guidance in the model "
                "wrapper."
            )
        if model_kwargs is None:
            model_kwargs = {}
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            float(eta)
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * torch.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_stride(
        self,
        model: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eta: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """DDIM jump from index ``t`` to a lower index ``t_next`` (timestep respacing).

        Uses the same equation as :meth:`ddim_sample` with
        ``alpha_bar = alphas_cumprod[t]`` and ``alpha_bar_prev = alphas_cumprod[t_next]``.
        Noise is masked when ``t_next == 0`` so the final step is deterministic if
        ``eta == 0``.
        """
        if model_kwargs is None:
            model_kwargs = {}
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod, t_next, x.shape)
        sigma = (
            float(eta)
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * torch.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t_next != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def _vb_terms_bpd(
        self,
        model: Callable[..., torch.Tensor],
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        if model_kwargs is None:
            model_kwargs = {}
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        kl = normal_kl(
            true_mean,
            true_log_variance_clipped,
            out["mean"],
            out["log_variance"],
        )
        kl = mean_flat(kl) / np.log(2.0)

        log_scales = 0.5 * out["log_variance"]
        if log_scales.shape != x_start.shape:
            log_scales = log_scales.expand_as(x_start)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            means=out["mean"],
            log_scales=log_scales,
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model: Callable[..., torch.Tensor],
        x_start: torch.Tensor,
        t: torch.Tensor,
        *,
        model_kwargs: dict[str, Any] | None = None,
        noise: torch.Tensor | None = None,
        mean_loss_kind: str = "l2",
        log_diagnostic_vb: bool = True,
    ) -> dict[str, torch.Tensor]:
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        if self.loss_type.is_vb():
            terms: dict[str, torch.Tensor] = {}
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] = terms["loss"] * float(self.num_timesteps)
            terms["vb"] = terms["loss"]
            return terms

        model_output = model(x_t, t, **model_kwargs)
        c = self.coord_channels

        if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
            assert model_output.shape[-1] == c * 2
            model_out_mean, model_var_values = torch.split(model_output, c, dim=-1)
            frozen_out = torch.cat([model_out_mean.detach(), model_var_values], dim=-1)
            vb = self._vb_terms_bpd(
                model=lambda *a, **k: frozen_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs={},
            )["output"]
            if self.loss_type == LossType.RESCALED_MSE:
                vb = vb * float(self.num_timesteps) / 1000.0
            terms = {"vb": vb}
            model_output = model_out_mean
        else:
            terms = {}

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            target, _, _ = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        elif self.model_mean_type == ModelMeanType.START_X:
            target = x_start
        else:
            target = noise

        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = _mean_loss(model_output, target, mean_loss_kind)
        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]
        if "vb" not in terms and log_diagnostic_vb:
            with torch.no_grad():
                terms["vb"] = self._vb_terms_bpd(
                    model=model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )["output"]
        return terms

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM step using fixed posterior variance (epsilon-only models)."""
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        noise = torch.randn_like(x_t)
        mask = (t != 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
        return posterior_mean + mask * torch.exp(0.5 * log_var) * noise


def parse_model_mean_type(s: str) -> ModelMeanType:
    key = str(s).lower().strip()
    if key in ("eps", "epsilon"):
        return ModelMeanType.EPSILON
    if key in ("x0", "start_x", "start"):
        return ModelMeanType.START_X
    if key in ("prev", "previous_x", "previous"):
        return ModelMeanType.PREVIOUS_X
    raise ValueError(f"Unknown model_mean_type: {s!r}")


def parse_model_var_type(s: str) -> ModelVarType:
    key = str(s).lower().strip()
    mapping = {
        "fixed_small": ModelVarType.FIXED_SMALL,
        "fixed_large": ModelVarType.FIXED_LARGE,
        "learned": ModelVarType.LEARNED,
        "learned_range": ModelVarType.LEARNED_RANGE,
    }
    if key not in mapping:
        raise ValueError(f"Unknown model_var_type: {s!r}")
    return mapping[key]


def parse_loss_type(s: str) -> LossType:
    key = str(s).lower().strip()
    mapping = {
        "mse": LossType.MSE,
        "rescaled_mse": LossType.RESCALED_MSE,
        "kl": LossType.KL,
        "rescaled_kl": LossType.RESCALED_KL,
    }
    if key not in mapping:
        raise ValueError(f"Unknown diffusion loss_type: {s!r}")
    return mapping[key]
