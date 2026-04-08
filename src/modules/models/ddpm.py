from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from src.modules.diffusion.schedule import DDPMNoiseSchedule


class TrajectoryDDPMModel(nn.Module):
    """DDPM wrapper: epsilon backbone + noise schedule (training + sampling)."""

    def __init__(
        self,
        backbone: nn.Module,
        schedule: DDPMNoiseSchedule,
        parameterization: str = "eps",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.schedule = schedule
        self.parameterization = str(parameterization)
        if self.parameterization != "eps":
            raise ValueError(
                f"Only parameterization='eps' supported, got {parameterization!r}"
            )

    def _backbone_accepts_mask(self) -> bool:
        return "mask" in inspect.signature(self.backbone.forward).parameters

    def _call_backbone(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        obs_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._backbone_accepts_mask():
            return self.backbone(x, t, context, obs_mask)
        return self.backbone(x, t, context)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            context = torch.zeros(
                x_t.shape[0],
                x_t.shape[1],
                x_t.shape[2],
                x_t.shape[3],
                device=x_t.device,
                dtype=x_t.dtype,
            )
        return self._call_backbone(x_t, t, context, obs_mask)

    @torch.no_grad()
    def sample(
        self,
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
    ) -> torch.Tensor:
        """Ancestral DDPM sampling; returns x_0 shaped [B, S, N, C]."""
        shape = (batch_size, seq_len, num_agents, coord_dim)
        x = torch.randn(shape, device=device, dtype=dtype)
        if context is None:
            context = torch.zeros(shape, device=device, dtype=dtype)
        gs = float(guidance_scale)
        use_cfg = gs != 1.0
        if cfg_null_context is None:
            null_ctx = torch.zeros_like(context)
        else:
            null_ctx = cfg_null_context
        accepts_mask = self._backbone_accepts_mask()
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
        for step in reversed(range(self.schedule.num_timesteps)):
            t = torch.full(
                (batch_size,),
                step,
                device=device,
                dtype=torch.long,
            )
            if use_cfg:
                eps_u = self._call_backbone(x, t, null_ctx, null_m)
                eps_c = self._call_backbone(x, t, context, obs_mask)
                noise_pred = eps_u + gs * (eps_c - eps_u)
            else:
                noise_pred = self._call_backbone(x, t, context, obs_mask)
            x = self.schedule.p_sample_step(x, t, noise_pred)
            if verbose:
                in_b = (x >= -2) & (x <= 2)
                percentage = in_b.all(dim=-1).float().mean() * 100
                print(f"step {step}: {percentage.item()}% within bounds")
        return x
