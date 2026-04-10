from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from src.modules.diffusion.gaussian_diffusion import GaussianDiffusion


class TrajectoryDDPMModel(nn.Module):
    """DDPM wrapper: backbone + Gaussian diffusion (training + sampling)."""

    def __init__(
        self,
        backbone: nn.Module,
        schedule: GaussianDiffusion,
        parameterization: str = "eps",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.schedule = schedule
        self.parameterization = str(parameterization)
        if self.parameterization not in ("eps", "x0"):
            raise ValueError(
                f"Only parameterization in ('eps','x0') supported, got {parameterization!r}"
            )

    @property
    def learn_sigma(self) -> bool:
        return bool(getattr(self.backbone, "learn_sigma", False))

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
        **kwargs: Any,
    ) -> torch.Tensor:
        if context is None:
            context = kwargs.get("context")
        if context is None:
            context = torch.zeros(
                x_t.shape[0],
                x_t.shape[1],
                x_t.shape[2],
                x_t.shape[3],
                device=x_t.device,
                dtype=x_t.dtype,
            )
        obs_mask = kwargs.get("obs_mask", obs_mask)
        return self._call_backbone(x_t, t, context, obs_mask)

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> Any:
        """Drop legacy ``schedule.*`` tensors except ``schedule.betas``; derived buffers are rebuilt."""
        filtered: dict[str, Any] = {}
        for k, v in state_dict.items():
            if k.startswith("schedule.") and k != "schedule.betas":
                continue
            filtered[k] = v
        return super().load_state_dict(filtered, strict=strict)
