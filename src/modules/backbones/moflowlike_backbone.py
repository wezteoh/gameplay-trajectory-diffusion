"""Epsilon predictor for [B, S, N, C] (causal PointNet + agent Transformer)."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from src.modules.backbones.causal_pointnet_encoder import CausalPointNetEncoder
from src.modules.diffusion.timestep_embedder import TimestepEmbedder


class MoflowlikeBackbone(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_agents: int,
        coord_dim: int,
        team_size: int | None = None,
        agent_embedding_dim: int = 64,
        pointnet_hidden_dim: int = 128,
        pointnet_num_layers: int = 3,
        pointnet_num_pre_layers: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        transformer_dropout: float = 0.0,
        num_transformer_layers: int = 4,
        agentwise_mlp_hidden: tuple[int, ...] = (256, 256),
        d_shared_head_mlp: int = 512,
        t_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        if team_size is None:
            self.team_size = max(1, (self.num_agents - 1) // 2)
        else:
            self.team_size = int(team_size)
        if 2 * self.team_size + 1 != self.num_agents:
            raise ValueError(
                f"Expected num_agents == 2 * team_size + 1 (two teams + ball), "
                f"got num_agents={self.num_agents}, team_size={self.team_size}"
            )

        d_agent = int(agent_embedding_dim)
        self.team_one_query_embedding = nn.Embedding(1, d_agent)
        self.team_two_query_embedding = nn.Embedding(1, d_agent)
        self.ball_query_embedding = nn.Embedding(1, d_agent)

        self.pointnet_encoder = CausalPointNetEncoder(
            in_channels=self.coord_dim,
            hidden_dim=int(pointnet_hidden_dim),
            num_layers=int(pointnet_num_layers),
            num_pre_layers=int(pointnet_num_pre_layers),
            out_channels=None,
        )
        d_pt = int(pointnet_hidden_dim)

        self.t_embedder = TimestepEmbedder(
            int(d_model), frequency_embedding_size=int(t_embedding_size)
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(d_pt + d_agent + int(d_model), int(d_model)),
            nn.ReLU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=int(num_transformer_layers),
        )
        self.post_act = nn.ReLU()

        agent_layers: list[nn.Module] = []
        prev = int(d_model)
        for h in agentwise_mlp_hidden:
            agent_layers.extend([nn.Linear(prev, int(h)), nn.ReLU()])
            prev = int(h)
        self.agentwise_mlp = nn.Sequential(*agent_layers)

        out_flat = self.seq_len * self.num_agents * self.coord_dim
        self.shared_head_mlp = nn.Sequential(
            nn.Linear(self.num_agents * prev, int(d_shared_head_mlp)),
            nn.ReLU(),
            nn.Linear(int(d_shared_head_mlp), int(d_shared_head_mlp)),
            nn.ReLU(),
            nn.Linear(int(d_shared_head_mlp), out_flat),
        )

    def _agent_query_embedding(self, ref: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros(1, device=ref.device, dtype=torch.long)
        team_one = self.team_one_query_embedding(idx).squeeze(0)
        team_two = self.team_two_query_embedding(idx).squeeze(0)
        ball = self.ball_query_embedding(idx).squeeze(0)
        return torch.cat(
            [
                team_one.unsqueeze(0).expand(self.team_size, -1),
                team_two.unsqueeze(0).expand(self.team_size, -1),
                ball.unsqueeze(0),
            ],
            dim=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        context = rearrange(context, "b t a c -> b a t c")
        context = self.pointnet_encoder(context)

        agent_query = self._agent_query_embedding(x)
        agent_query = agent_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        t_emb = self.t_embedder(t).unsqueeze(1).expand(-1, x.shape[2], -1)
        x = torch.cat([context, agent_query, t_emb], dim=-1)
        x = self.mlp_z(x)

        x = self.transformer_encoder(x)
        x = self.post_act(x)
        x = self.agentwise_mlp(x)
        x = rearrange(x, "b a d -> b (a d)")
        x = self.shared_head_mlp(x)
        return rearrange(
            x,
            "b (t a c) -> b t a c",
            t=self.seq_len,
            a=self.num_agents,
            c=self.coord_dim,
        )
