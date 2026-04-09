"""DDPM trajectory sampling shared by ``sample_trajectory_ddpm`` and metric eval."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from src.interfaces.trajectory_filling_ddpm import (
    _blend_traj_rollout_from_last_observed,
)
from src.sampling.dpm_solver import sample_trajectory_dpm_solver
from src.utils.trajectory_coords import denormalize_delta


def model_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "model.name", default="trajectory_ddpm"))


def sampling_options_from_cfg(cfg: DictConfig | None) -> tuple[str, dict[str, Any]]:
    """Return ``(sampler, dpm_options)`` from ``cfg.sampling`` (defaults: ancestral, empty)."""
    if cfg is None:
        return "ancestral", {}
    method = str(OmegaConf.select(cfg, "sampling.method", default="ancestral"))
    raw = OmegaConf.select(cfg, "sampling.dpm", default={})
    dpm: dict[str, Any]
    if raw is None or raw == {}:
        dpm = {}
    else:
        container = OmegaConf.to_container(raw, resolve=True)
        dpm = dict(container) if isinstance(container, dict) else {}
    return method, dpm


def _merge_dpm_options(dpm_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge YAML/script DPM hyperparameters.

    ``trace_every`` is not merged: it is only honored via the explicit
    ``trace_every`` argument to :func:`sample_with_trace` (training/validation
    sampling passes ``None`` so denoising checkpoints are not collected).
    """
    defaults: dict[str, Any] = {
        "steps": 20,
        "order": 3,
        "skip_type": "time_uniform",
        "solver_method": "singlestep",
        "algorithm_type": "dpmsolver++",
        "lower_order_final": True,
        "denoise_to_zero": False,
        "solver_type": "dpmsolver",
    }
    if dpm_config:
        for k, v in dpm_config.items():
            if k == "trace_every":
                continue
            if v is not None:
                defaults[k] = v
    return defaults


@torch.no_grad()
def sample_with_trace(
    model: Any,
    *,
    batch_size: int,
    seq_len: int,
    num_agents: int,
    coord_dim: int,
    device: torch.device,
    context: torch.Tensor | None = None,
    obs_mask: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    cfg_null_context: torch.Tensor | None = None,
    cfg_null_mask: torch.Tensor | None = None,
    trace_every: int | None = None,
    sampler: str = "ancestral",
    dpm_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """DDPM reverse sampling with optional denoising snapshots.

    ``sampler`` is ``\"ancestral\"`` (full DDPM) or ``\"dpm\"`` (DPM-Solver). When
    ``sampler==\"dpm\"``, optional ``dpm_config`` overrides defaults (steps, order,
    skip_type, solver_method, algorithm_type, lower_order_final, denoise_to_zero,
    solver_type).

    For ``sampler==\"dpm\"`` and ``trace_every``, trace keys are **solver-step**
    indices (see :func:`sample_trajectory_dpm_solver`), not DDPM timesteps.
    """
    if sampler == "dpm":
        opts = _merge_dpm_options(dpm_config)
        return sample_trajectory_dpm_solver(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            num_agents=num_agents,
            coord_dim=coord_dim,
            device=device,
            context=context,
            obs_mask=obs_mask,
            guidance_scale=guidance_scale,
            dtype=dtype,
            verbose=verbose,
            cfg_null_context=cfg_null_context,
            cfg_null_mask=cfg_null_mask,
            trace_every=trace_every,
            **opts,
        )

    shape = (batch_size, seq_len, num_agents, coord_dim)
    x = torch.randn(shape, device=device, dtype=dtype)
    traces: list[tuple[int, torch.Tensor]] = []

    if trace_every is not None and trace_every > 0:
        traces.append((int(model.schedule.num_timesteps), x.detach().clone()))

    if context is None:
        context = torch.zeros(shape, device=device, dtype=dtype)
    gs = float(guidance_scale)
    use_cfg = gs != 1.0
    null_ctx = (
        torch.zeros_like(context) if cfg_null_context is None else cfg_null_context
    )
    accepts_mask = model._backbone_accepts_mask()
    if use_cfg and accepts_mask:
        if cfg_null_mask is not None:
            null_m = cfg_null_mask
        elif obs_mask is not None:
            null_m = torch.zeros_like(obs_mask, dtype=dtype, device=device)
        else:
            null_m = torch.zeros(
                batch_size, seq_len, num_agents, device=device, dtype=dtype
            )
    else:
        null_m = None

    if verbose:
        print("sampling")
    for step in reversed(range(model.schedule.num_timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        if use_cfg:
            eps_u = model._call_backbone(x, t, null_ctx, null_m)
            eps_c = model._call_backbone(x, t, context, obs_mask)
            noise_pred = eps_u + gs * (eps_c - eps_u)
        else:
            noise_pred = model._call_backbone(x, t, context, obs_mask)
        x = model.schedule.p_sample_step(x, t, noise_pred)
        if trace_every is not None and trace_every > 0:
            if step == 0 or (step % int(trace_every) == 0):
                traces.append((int(step), x.detach().clone()))
        if verbose:
            in_b = (x >= -2) & (x <= 2)
            percentage = in_b.all(dim=-1).float().mean() * 100
            print(f"step {step}: {percentage.item()}% within bounds")
    return x, traces


@torch.no_grad()
def sample_batch_multi_path(
    module: Any,
    cfg: DictConfig,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    num_paths: int,
    *,
    guidance_scale_override: float | None = None,
    verbose: bool = False,
    sampler: str | None = None,
    dpm_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Draw ``num_paths`` samples per row; return normalized court positions ``[B,P,T,A,2]``.

    For ``trajectory_filling_ddpm``, positions are **blended**: observed steps match GT;
    masked steps follow the predicted delta rollout from the last observed coordinate
    (same as validation blend videos).
    """
    mname = model_name(cfg)
    p = int(num_paths)
    if p < 1:
        raise ValueError("num_paths must be >= 1")

    sm, dpm_d = sampling_options_from_cfg(cfg)
    if sampler is not None:
        sm = str(sampler)
    if dpm_config is not None:
        dpm_d = {**dpm_d, **dpm_config}

    if mname == "trajectory_ddpm":
        bsz = next(iter(batch.values())).shape[0]
        x, _ = sample_with_trace(
            module.model,
            batch_size=bsz * p,
            seq_len=int(module.seq_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            verbose=verbose,
            sampler=sm,
            dpm_config=dpm_d,
        )
        return rearrange(x, "(b p) t a c -> b p t a c", b=bsz, p=p)

    bsz = next(iter(batch.values())).shape[0]
    ctx_key = str(module.context_key)

    if mname == "trajectory_completion_ddpm":
        past = batch[ctx_key]
        gs = (
            float(module.guidance_scale)
            if guidance_scale_override is None
            else float(guidance_scale_override)
        )
        past_rep = past.repeat_interleave(p, dim=0)
        dtype = past.dtype
        future_hat, _ = sample_with_trace(
            module.model,
            batch_size=bsz * p,
            seq_len=int(module.future_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            context=past_rep,
            guidance_scale=gs,
            dtype=dtype,
            verbose=verbose,
            sampler=sm,
            dpm_config=dpm_d,
        )
        anchor = past_rep[:, -1]
        future_abs = future_hat + anchor.unsqueeze(1)
        x = torch.cat([past_rep, future_abs], dim=1)
        return rearrange(x, "(b p) t a c -> b p t a c", b=bsz, p=p)

    if mname == "trajectory_filling_ddpm":
        mask_key = str(module.mask_key)
        pos_key = str(module.position_0_key)
        context = batch[ctx_key]
        obs_mask = batch[mask_key].to(dtype=context.dtype)
        pos0 = batch[pos_key]
        dtype = context.dtype
        gs = (
            float(module.guidance_scale)
            if guidance_scale_override is None
            else float(guidance_scale_override)
        )
        context_rep = context.repeat_interleave(p, dim=0)
        obs_rep = obs_mask.repeat_interleave(p, dim=0)
        pos0_rep = pos0.repeat_interleave(p, dim=0)
        fill = module._context_fill_brc.to(device=device, dtype=dtype).expand(
            bsz * p,
            int(module.seq_len),
            int(module.num_agents),
            int(module.context_channels),
        )
        null_mask = torch.zeros(
            bsz * p,
            int(module.seq_len),
            int(module.num_agents),
            device=device,
            dtype=dtype,
        )
        deltas_hat, _ = sample_with_trace(
            module.model,
            batch_size=bsz * p,
            seq_len=int(module.seq_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            context=context_rep,
            obs_mask=obs_rep,
            guidance_scale=gs,
            dtype=dtype,
            verbose=verbose,
            cfg_null_context=fill,
            cfg_null_mask=null_mask,
            sampler=sm,
            dpm_config=dpm_d,
        )
        d_raw = denormalize_delta(
            deltas_hat,
            module._delta_shift,
            module._delta_scale,
        )
        traj_key = str(module.trajectory_key)
        x0 = batch[traj_key]
        x0_rep = x0.repeat_interleave(p, dim=0)
        d_gt_raw = denormalize_delta(
            x0_rep,
            module._delta_shift,
            module._delta_scale,
        )
        traj_gt = pos0_rep.unsqueeze(1) + torch.cumsum(d_gt_raw, dim=1)
        x = _blend_traj_rollout_from_last_observed(
            pos0_rep,
            d_raw,
            traj_gt,
            obs_rep,
        )
        return rearrange(x, "(b p) t a c -> b p t a c", b=bsz, p=p)

    raise ValueError(f"Unsupported model.name for sampling: {mname!r}")
