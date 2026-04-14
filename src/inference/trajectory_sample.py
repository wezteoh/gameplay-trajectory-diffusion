"""DDPM trajectory sampling shared by ``sample_trajectory_ddpm.py`` and metric eval."""

from __future__ import annotations

from typing import Any, Collection

import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from src.interfaces.trajectory_filling_ddpm import (
    _blend_traj_rollout_from_last_observed,
)
from src.modules.diffusion.gaussian_diffusion import ddim_timestep_sequence
from src.sampling.dpm_solver import sample_trajectory_dpm_solver
from src.utils.trajectory_coords import denormalize_delta


def model_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "model.name", default="trajectory_filling_ddpm"))


def minimal_sample_cfg_filling_ddpm(
    *,
    sampling_method: str,
    sampling_dpm: dict[str, Any] | None = None,
    sampling_ddim: dict[str, Any] | None = None,
) -> DictConfig:
    """Minimal config for :func:`sample_batch_multi_path` (filling DDPM only)."""
    dpm = dict(sampling_dpm) if sampling_dpm else {}
    ddim = dict(sampling_ddim) if sampling_ddim else {}
    return OmegaConf.create(
        {
            "model": {"name": "trajectory_filling_ddpm"},
            "sampling": {
                "method": str(sampling_method),
                "dpm": dpm,
                "ddim": ddim,
            },
        }
    )


def ground_truth_normalized_filling(
    module: Any,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Full trajectory in normalized court space ``[B, T, A, C]`` (filling DDPM)."""
    traj_key = str(module.trajectory_key)
    pos0 = batch[str(module.position_0_key)]
    x0 = batch[traj_key]
    d_raw = denormalize_delta(
        x0,
        module._delta_shift,
        module._delta_scale,
    )
    return pos0.unsqueeze(1) + torch.cumsum(d_raw, dim=1)


def sampling_options_from_cfg(
    cfg: DictConfig | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return ``(sampler, dpm_options, ddim_options)`` from ``cfg.sampling``."""
    if cfg is None:
        return "ancestral", {}, {}
    method = str(OmegaConf.select(cfg, "sampling.method", default="ancestral"))
    raw_dpm = OmegaConf.select(cfg, "sampling.dpm", default={})
    if raw_dpm is None or raw_dpm == {}:
        dpm: dict[str, Any] = {}
    else:
        c_dpm = OmegaConf.to_container(raw_dpm, resolve=True)
        dpm = dict(c_dpm) if isinstance(c_dpm, dict) else {}
    raw_ddim = OmegaConf.select(cfg, "sampling.ddim", default={})
    if raw_ddim is None or raw_ddim == {}:
        ddim: dict[str, Any] = {}
    else:
        c_ddim = OmegaConf.to_container(raw_ddim, resolve=True)
        ddim = dict(c_ddim) if isinstance(c_ddim, dict) else {}
    return method, dpm, ddim


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


def _merge_ddim_options(ddim_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge YAML/script DDIM hyperparameters (``trace_every`` is not merged)."""
    defaults: dict[str, Any] = {"steps": 50, "eta": 0.0}
    if ddim_config:
        for k, v in ddim_config.items():
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
    trace_steps: Collection[int] | None = None,
    sampler: str = "ancestral",
    dpm_config: dict[str, Any] | None = None,
    ddim_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """DDPM reverse sampling with optional denoising snapshots.

    ``sampler`` is ``\"ancestral\"`` (full DDPM), ``\"ddim\"``, or ``\"dpm\"``
    (DPM-Solver). When ``sampler==\"dpm\"``, optional ``dpm_config`` overrides
    defaults (steps, order, skip_type, solver_method, algorithm_type,
    lower_order_final, denoise_to_zero, solver_type). When ``sampler==\"ddim\"``,
    optional ``ddim_config`` overrides defaults (``steps``, ``eta``).

    Pass **at most one** of ``trace_every`` (log every k steps) or ``trace_steps``
    (explicit indices). If both are set, raises ``ValueError``.

    Trace key semantics:
      - **ancestral**: DDPM noise level ``t`` after each ``p_sample`` (and optionally
        the initial draw if ``num_timesteps`` is listed in ``trace_steps``).
      - **ddim**: inner loop index ``i`` (and optionally initial noise if
        ``num_timesteps`` is listed).
      - **dpm**: solver step keys as in :func:`sample_trajectory_dpm_solver`.
    """
    if trace_every is not None and trace_steps is not None:
        raise ValueError("Pass at most one of trace_every and trace_steps.")
    ts_set: frozenset[int] | None = None
    if trace_steps is not None:
        ts_set = frozenset(int(s) for s in trace_steps)
        if not ts_set:
            ts_set = None
    te_val = int(trace_every) if trace_every is not None and trace_every > 0 else None
    want_trace = ts_set is not None or te_val is not None

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
            trace_every=te_val,
            trace_steps=ts_set,
            **opts,
        )

    if sampler not in ("ancestral", "ddim"):
        raise ValueError(
            f"Unknown sampler {sampler!r}; expected 'ancestral', 'ddim', or 'dpm'"
        )

    shape = (batch_size, seq_len, num_agents, coord_dim)
    x = torch.randn(shape, device=device, dtype=dtype)
    traces: list[tuple[int, torch.Tensor]] = []

    if want_trace:
        if ts_set is not None:
            if int(model.schedule.num_timesteps) in ts_set:
                traces.append((int(model.schedule.num_timesteps), x.detach().clone()))
        else:
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

    model_kwargs: dict[str, Any] = {"context": context, "obs_mask": obs_mask}

    def model_fn(x_in: torch.Tensor, t_in: torch.Tensor, **mkw: Any) -> torch.Tensor:
        ctx = mkw["context"]
        m = mkw.get("obs_mask")
        if use_cfg:
            u = model._call_backbone(x_in, t_in, null_ctx, null_m)
            c = model._call_backbone(x_in, t_in, ctx, m)
            return u + gs * (c - u)
        return model._call_backbone(x_in, t_in, ctx, m)

    if sampler == "ddim":
        dopts = _merge_ddim_options(ddim_config)
        tseq = ddim_timestep_sequence(
            int(model.schedule.num_timesteps), int(dopts["steps"])
        )
        eta = float(dopts["eta"])
        for i in range(len(tseq) - 1):
            t_cur = tseq[i]
            t_nxt = tseq[i + 1]
            t = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)
            t_next = torch.full((batch_size,), t_nxt, device=device, dtype=torch.long)
            out = model.schedule.ddim_sample_stride(
                model_fn,
                x,
                t,
                t_next,
                eta=eta,
                model_kwargs=model_kwargs,
            )
            x = out["sample"]
            if want_trace:
                if ts_set is not None:
                    if i in ts_set:
                        traces.append((int(i), x.detach().clone()))
                elif i == 0 or (i % int(te_val) == 0):
                    traces.append((int(i), x.detach().clone()))
            if verbose:
                in_b = (x >= -2) & (x <= 2)
                percentage = in_b.all(dim=-1).float().mean() * 100
                print(
                    f"ddim step {i} (t={t_cur}->{t_nxt}): "
                    f"{percentage.item()}% within bounds"
                )
        return x, traces

    for step in reversed(range(model.schedule.num_timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        out = model.schedule.p_sample(
            model_fn,
            x,
            t,
            model_kwargs=model_kwargs,
        )
        x = out["sample"]
        if want_trace:
            if ts_set is not None:
                if step in ts_set:
                    traces.append((int(step), x.detach().clone()))
            elif step == 0 or (step % int(te_val) == 0):
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
    position_mode: str = "blended",
    guidance_scale_override: float | None = None,
    verbose: bool = False,
    sampler: str | None = None,
    dpm_config: dict[str, Any] | None = None,
    ddim_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Draw ``num_paths`` samples per row; return positions ``[B,P,T,A,C]``.

    ``position_mode``:
      - ``blended`` (default): observed steps match GT; masked steps roll out
        predicted deltas from the last observed coordinate (like blend videos).
      - ``pure``: ``position_0 + cumsum(predicted deltas)`` at every step; no
        mask blend.

    Returns the full horizon; callers may slice ``[metrics_start_t, T)`` for
    metrics.
    """
    mname = model_name(cfg)
    p = int(num_paths)
    if p < 1:
        raise ValueError("num_paths must be >= 1")

    sm, dpm_d, ddim_d = sampling_options_from_cfg(cfg)
    if sampler is not None:
        sm = str(sampler)
    if dpm_config is not None:
        dpm_d = {**dpm_d, **dpm_config}
    if ddim_config is not None:
        ddim_d = {**ddim_d, **ddim_config}

    if mname != "trajectory_filling_ddpm":
        raise ValueError(
            f"Unsupported model.name for sampling: {mname!r}; "
            "expected 'trajectory_filling_ddpm'"
        )

    pm = str(position_mode).strip().lower()
    if pm not in ("blended", "pure"):
        raise ValueError(
            f"position_mode must be 'blended' or 'pure', got {position_mode!r}"
        )

    bsz = next(iter(batch.values())).shape[0]
    ctx_key = str(module.context_key)
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
        ddim_config=ddim_d,
    )
    d_raw = denormalize_delta(
        deltas_hat,
        module._delta_shift,
        module._delta_scale,
    )
    if pm == "pure":
        x = pos0_rep.unsqueeze(1) + torch.cumsum(d_raw, dim=1)
    else:
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
