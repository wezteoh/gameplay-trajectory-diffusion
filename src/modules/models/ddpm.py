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
        sampler: str = "ancestral",
        dpm_config: dict | None = None,
    ) -> torch.Tensor:
        """Reverse sampling; returns x_0 shaped [B, S, N, C].

        ``sampler`` is ``\"ancestral\"`` or ``\"dpm\"`` (DPM-Solver). When ``dpm``,
        pass optional ``dpm_config`` (steps, order, etc.); see
        :func:`src.inference.trajectory_sample.sample_with_trace`.
        """
        from src.inference.trajectory_sample import sample_with_trace

        x, _ = sample_with_trace(
            self,
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
            trace_every=None,
            sampler=sampler,
            dpm_config=dpm_config,
        )
        return x
