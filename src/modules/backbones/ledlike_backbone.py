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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

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


class LEDLikeBackbone(Module):
    """Epsilon backbone for [B, T, A, C] trajectories; C must be 2."""

    def __init__(
        self,
        past_seq_len: int,
        future_seq_len: int,
        num_agents: int,
        coord_dim: int = 2,
        context_dim: int = 256,
        tf_layer: int = 2,
        num_timesteps: int = 1000,
    ):
        super().__init__()
        if int(coord_dim) != 2:
            raise ValueError(
                "LEDLikeBackbone only supports coord_dim=2 (xy); " f"got coord_dim={coord_dim}"
            )
        self.past_seq_len = int(past_seq_len)
        self.future_seq_len = int(future_seq_len)
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        self.context_dim = int(context_dim)
        self.num_timesteps = max(1, int(num_timesteps))

        past_flat = self.past_seq_len * self.coord_dim
        self.encoder_context = social_transformer(past_flat, self.context_dim)
        pe_max = max(self.future_seq_len, 32)
        self.pos_emb = PositionalEncoding(d_model=2 * self.context_dim, dropout=0.1, max_len=pe_max)
        ctx_in = self.context_dim + 3
        self.concat1 = ConcatSquashLinear(2, 2 * self.context_dim, ctx_in)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * self.context_dim,
            nhead=2,
            dim_feedforward=2 * self.context_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=int(tf_layer))
        self.concat3 = ConcatSquashLinear(2 * self.context_dim, self.context_dim, ctx_in)
        self.concat4 = ConcatSquashLinear(self.context_dim, self.context_dim // 2, ctx_in)
        self.linear = ConcatSquashLinear(self.context_dim // 2, 2, ctx_in)

    def forward(self, x, t, context):
        x = rearrange(x, "b t a c -> b a t c")
        batch_size = x.size(0)
        t_norm = t.float().view(batch_size, 1, 1) / float(self.num_timesteps)
        context = rearrange(context, "b t a d -> b a t d")
        context = self.encoder_context(context)

        time_emb = torch.cat([t_norm, torch.sin(t_norm), torch.cos(t_norm)], dim=-1).repeat(
            1, self.num_agents, 1
        )
        ctx_emb = torch.cat([time_emb, context], dim=-1).unsqueeze(2)

        x = self.concat1(ctx_emb, x)
        x = rearrange(x, "b a t d -> (b a) t d")
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        x = rearrange(x, "(b a) t d -> b a t d", a=self.num_agents)
        x = self.concat3(ctx_emb, x)
        x = self.concat4(ctx_emb, x)
        x = self.linear(ctx_emb, x)
        return rearrange(x, "b a t d -> b t a d")
