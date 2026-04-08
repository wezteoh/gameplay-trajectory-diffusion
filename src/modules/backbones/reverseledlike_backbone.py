from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Linear, Module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D); pe: (1, max_len, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class st_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(
            channel_in, channel_out, dim_kernel, stride=1, padding=1
        )
        self.temporal_encoder = nn.GRU(
            channel_out, self.dim_embedding_key, 1, batch_first=True
        )

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        """
        X: b, T, 2

        return: b, F
        """
        X_t = torch.transpose(X, -2, -1)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, -2, -1)

        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.squeeze(0)

        return state_x


class social_transformer(nn.Module):
    def __init__(self, past_flat_dim: int, context_dim: int = 256):
        super().__init__()
        self.encode_past = nn.Linear(past_flat_dim, context_dim, bias=False)
        self.layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=2,
            dim_feedforward=context_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h, mask=None):
        """h: b, a, t, c. mask is unused (API compatibility)."""
        del mask
        h_feat = self.encode_past(h.reshape(h.size(0), h.size(1), -1))
        h_feat_ = self.transformer_encoder(h_feat)
        h_feat = h_feat + h_feat_

        return h_feat


class ReverseLEDLikeBackbone(Module):
    """Epsilon backbone for noisy [B, T, A, coord_dim] with conditioning context.

    Noisy trajectory channels coord_dim must be 2 (xy deltas). Context may use
    extra channels (e.g. coord_dim=2 but context_channels=4 for XY + delta).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_agents: int,
        coord_dim: int = 2,
        context_channels: int | None = None,
        context_dim: int = 256,
        n_temporal_layer: int = 2,
        d_model_temporal: int = 256,
        nhead_temporal: int = 4,
        d_ff_temporal: int = 1024,
        n_spatial_layer: int = 2,
        d_model_spatial: int = 256,
        nhead_spatial: int = 4,
        d_ff_spatial: int = 1024,
        num_timesteps: int = 1000,
        agent_embedding_dim: int = 64,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.team_size = (self.num_agents - 1) // 2
        self.coord_dim = int(coord_dim)
        if self.coord_dim != 2:
            raise ValueError(
                "ReverseLEDLikeBackbone only supports coord_dim=2 (xy); "
                f"got coord_dim={coord_dim}"
            )
        self.context_channels = int(context_channels) if context_channels is not None else self.coord_dim
        self.context_dim = int(context_dim)
        self.max_seq_len = max(1, int(max_seq_len))
        self.num_timesteps = max(1, int(num_timesteps))
        d_agent = int(agent_embedding_dim)

        self.team_one_query_embedding = nn.Embedding(1, d_agent)
        self.team_two_query_embedding = nn.Embedding(1, d_agent)
        self.ball_query_embedding = nn.Embedding(1, d_agent)

        pe_max = max(self.max_seq_len, 32)
        self.pos_emb = PositionalEncoding(
            d_model=int(d_model_temporal), dropout=0.1, max_len=pe_max
        )

        self.pre_encoder = nn.Sequential(
            nn.Linear(
                self.context_channels + 3 + agent_embedding_dim + 1,
                context_dim,
            ),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

        self.concat1 = ConcatSquashLinear(coord_dim, d_model_temporal, self.context_dim)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model_temporal,
            nhead=nhead_temporal,
            dim_feedforward=d_ff_temporal,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=int(n_temporal_layer)
        )

        self.concat2 = ConcatSquashLinear(
            d_model_temporal, d_model_spatial, self.context_dim
        )

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model_spatial,
            nhead=nhead_spatial,
            dim_feedforward=d_ff_spatial,
            batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=int(n_spatial_layer)
        )

        self.concat3 = ConcatSquashLinear(
            d_model_spatial, d_model_spatial, self.context_dim
        )
        d_spatial_half = max(1, int(d_model_spatial) // 2)
        self.concat4 = ConcatSquashLinear(
            d_model_spatial, d_spatial_half, self.context_dim
        )
        self.linear = ConcatSquashLinear(d_spatial_half, 2, self.context_dim)

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

    def forward(self, x, t, context, mask=None):
        """
        x: b, t, a, coord_dim (noisy state, 2)
        t: b
        context: b, t, a, context_channels (often 2; may be 4 with XY+delta)
        mask: b, t, a
        """
        x_len = x.size(1)
        batch_size = x.size(0)
        t_norm = t.float().view(batch_size, 1, 1, 1) / float(self.num_timesteps)
        time_emb = torch.cat(
            [t_norm, torch.sin(t_norm), torch.cos(t_norm)], dim=-1
        ).expand(batch_size, context.size(1), self.num_agents, 3)
        agent_emb = self._agent_query_embedding(x)
        agent_emb = agent_emb.reshape(1, 1, self.num_agents, agent_emb.size(-1)).expand(
            context.size(0), context.size(1), -1, -1
        )
        if mask is None:
            mask = torch.ones(
                batch_size,
                context.size(1),
                self.num_agents,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            mask = mask.to(device=x.device, dtype=x.dtype)
        mask = mask.unsqueeze(-1)
        context = torch.cat([context, time_emb, agent_emb, mask], dim=-1)
        ctx_emb = self.pre_encoder(context)  # b,t,a,d
        x = self.concat1(ctx_emb, x)

        x = rearrange(x, "b t a d -> (b a) t d")
        x = self.pos_emb(x)
        x = self.temporal_encoder(x)

        x = rearrange(x, "(b a) t d -> b t a d", a=self.num_agents)
        x = self.concat2(ctx_emb, x)

        x = rearrange(x, "b t a d -> (b t) a d")
        x = self.spatial_encoder(x)

        x = rearrange(x, "(b t) a d -> b t a d", t=x_len)
        x = self.concat3(ctx_emb, x)
        x = self.concat4(ctx_emb, x)
        x = self.linear(ctx_emb, x)
        return x
