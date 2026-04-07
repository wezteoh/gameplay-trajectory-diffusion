"""Lightweight epsilon net for smoke tests and config defaults."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from src.modules.diffusion.timestep_embedder import TimestepEmbedder


class PlaceholderTrajectoryEpsilonNet(nn.Module):
    """MLP over flattened noisy trajectory + DiT-style timestep embedding."""

    def __init__(
        self,
        seq_len: int,
        num_agents: int,
        coord_dim: int,
        hidden_dim: int = 512,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        flat = self.seq_len * self.num_agents * self.coord_dim
        self.t_embedder = TimestepEmbedder(
            hidden_dim, frequency_embedding_size=frequency_embedding_size
        )
        self.net = nn.Sequential(
            nn.Linear(flat + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, s, n, c = x_t.shape
        if s != self.seq_len or n != self.num_agents or c != self.coord_dim:
            raise ValueError(
                f"Expected [*, {self.seq_len}, {self.num_agents}, {self.coord_dim}], "
                f"got {tuple(x_t.shape)}"
            )
        flat = rearrange(x_t, "b s n c -> b (s n c)")
        t_emb = self.t_embedder(t)
        h = torch.cat([flat, t_emb], dim=-1)
        out = self.net(h)
        return rearrange(
            out,
            "b (s n c) -> b s n c",
            s=self.seq_len,
            n=self.num_agents,
            c=self.coord_dim,
        )
