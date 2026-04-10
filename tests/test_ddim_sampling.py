"""Smoke tests for DDIM timestep spacing and deterministic sampling (eta=0)."""

from __future__ import annotations

import torch

from src.modules.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    ddim_timestep_sequence,
)


def test_ddim_timestep_sequence_is_strictly_decreasing_endpoints() -> None:
    s = ddim_timestep_sequence(100, 10)
    assert len(s) >= 2
    assert s[0] == 99
    assert s[-1] == 0
    for a, b in zip(s[:-1], s[1:], strict=True):
        assert a > b


def _zeros_model(x: torch.Tensor, t: torch.Tensor, **kwargs: object) -> torch.Tensor:
    return torch.zeros_like(x)


def test_ddim_stride_matches_ddim_sample_for_consecutive_indices() -> None:
    """``ddim_sample(t)`` uses ``alphas_cumprod_prev[t]`` == ``alphas_cumprod[t-1]`` for t>=1."""
    gd = GaussianDiffusion(
        timesteps=20,
        beta_schedule="linear",
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        coord_channels=2,
    )
    b, seq_len, n_agents, c_dim = 2, 4, 3, 2
    x = torch.randn(b, seq_len, n_agents, c_dim)
    tt = torch.full((b,), 7, dtype=torch.long)
    tn = torch.full((b,), 6, dtype=torch.long)
    o1 = gd.ddim_sample(_zeros_model, x, tt, eta=0.0, model_kwargs={})
    o2 = gd.ddim_sample_stride(_zeros_model, x, tt, tn, eta=0.0, model_kwargs={})
    assert torch.allclose(o1["sample"], o2["sample"])


def test_ddim_eta_zero_is_deterministic_across_runs() -> None:
    gd = GaussianDiffusion(
        timesteps=20,
        beta_schedule="linear",
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        coord_channels=2,
    )
    b, seq_len, n_agents, c_dim = 2, 4, 3, 2
    x0 = torch.randn(b, seq_len, n_agents, c_dim)
    seq = ddim_timestep_sequence(20, 8)
    outs: list[torch.Tensor] = []
    for _ in range(2):
        y = x0.clone()
        for i in range(len(seq) - 1):
            t_cur, t_nxt = seq[i], seq[i + 1]
            tt = torch.full((b,), t_cur, dtype=torch.long)
            t_next = torch.full((b,), t_nxt, dtype=torch.long)
            out = gd.ddim_sample_stride(
                _zeros_model,
                y,
                tt,
                t_next,
                eta=0.0,
                model_kwargs={},
            )
            y = out["sample"]
        outs.append(y)
    assert torch.allclose(outs[0], outs[1])


if __name__ == "__main__":
    test_ddim_timestep_sequence_is_strictly_decreasing_endpoints()
    test_ddim_stride_matches_ddim_sample_for_consecutive_indices()
    test_ddim_eta_zero_is_deterministic_across_runs()
    print("ok")
