# Timestep embedding pattern from DiT (Meta): https://github.com/facebookresearch/DiT
# Copyright (c) Meta Platforms, Inc. and affiliates. (BSD-style; see DiT LICENSE)
# References: GLIDE (OpenAI), DiT models.py TimestepEmbedder

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Scalar diffusion timesteps -> vector (sinusoidal + 2-layer MLP), DiT-style."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
