from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from src.modules.diffusion.timestep_embedder import TimestepEmbedder
from torch.nn import Module


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock1D(nn.Module):
    """DiT block with AdaLN-Zero modulation and 1D token attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x_attn = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_attn, x_attn, x_attn)[0]
        x_mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        return x


class DiTFinalLayer1D(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class PatchEmbed(nn.Module):
    """1D trajectory patch embedder over time: [B, T, C] -> [B, N, D]."""

    def __init__(
        self,
        max_seq_len: int,
        patch_size: int,
        in_chans: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        if max_seq_len % patch_size != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be divisible by patch_size ({patch_size})"
            )
        self.max_seq_len = int(max_seq_len)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.hidden_size = int(hidden_size)
        self.num_patches = self.max_seq_len // self.patch_size
        self.proj = nn.Conv1d(
            in_channels=self.in_chans,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x as [B, T, C], got shape {tuple(x.shape)}")
        b, t, c = x.shape
        if c != self.in_chans:
            raise ValueError(f"Expected channels={self.in_chans}, got channels={c}")
        if t < self.max_seq_len:
            x = torch.cat([x, x.new_zeros(b, self.max_seq_len - t, c)], dim=1)
        elif t > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


class FixedSinCosPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding over token index (no grad, no dropout)."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DITBackbone(Module):
    """1D trajectory DiT backbone over [B, T, A, C] with context and mask support."""

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
        patch_size: int = 1,
    ) -> None:
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.num_agents = int(num_agents)
        self.team_size = (self.num_agents - 1) // 2
        self.coord_dim = int(coord_dim)
        if self.coord_dim != 2:
            raise ValueError(
                "ReverseLEDDITBackbone only supports coord_dim=2 (xy); "
                f"got coord_dim={coord_dim}"
            )
        if self.num_agents != 2 * self.team_size + 1:
            raise ValueError(
                "DITBackbone expects num_agents = 2 * team_size + 1 (last agent is ball), "
                f"got num_agents={self.num_agents}"
            )
        self.context_channels = (
            int(context_channels) if context_channels is not None else self.coord_dim
        )
        self.context_with_mask_channels = self.context_channels + 1
        self.num_timesteps = max(1, int(num_timesteps))
        self.patch_size = int(patch_size)

        hidden_size = int(context_dim)
        if int(d_model_temporal) != hidden_size or int(d_model_spatial) != hidden_size:
            raise ValueError(
                "For ReverseLEDDITBackbone, context_dim, d_model_temporal, and "
                "d_model_spatial must match."
            )

        self.x_embedder = PatchEmbed(
            max_seq_len=self.max_seq_len,
            patch_size=self.patch_size,
            in_chans=self.coord_dim,
            hidden_size=hidden_size,
        )
        self.c_embedder = PatchEmbed(
            max_seq_len=self.max_seq_len,
            patch_size=self.patch_size,
            in_chans=self.context_with_mask_channels,
            hidden_size=hidden_size,
        )
        self.num_patches = self.x_embedder.num_patches
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_emb = FixedSinCosPositionalEncoding(
            d_model=hidden_size,
            max_len=max(32, self.num_patches),
        )

        # Keep agent query embeddings in model width directly (no projection).
        d_agent = hidden_size
        self.team_one_query_embedding = nn.Embedding(1, d_agent)
        self.team_two_query_embedding = nn.Embedding(1, d_agent)
        self.ball_query_embedding = nn.Embedding(1, d_agent)

        temporal_mlp_ratio = max(1.0, float(d_ff_temporal) / float(hidden_size))
        spatial_mlp_ratio = max(1.0, float(d_ff_spatial) / float(hidden_size))
        self.temporal_blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    hidden_size=hidden_size,
                    num_heads=int(nhead_temporal),
                    mlp_ratio=temporal_mlp_ratio,
                )
                for _ in range(int(n_temporal_layer))
            ]
        )
        self.agent_blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    hidden_size=hidden_size,
                    num_heads=int(nhead_spatial),
                    mlp_ratio=spatial_mlp_ratio,
                )
                for _ in range(int(n_spatial_layer))
            ]
        )
        self.final_layer = DiTFinalLayer1D(
            hidden_size=hidden_size,
            out_dim=self.patch_size * self.coord_dim,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        nn.init.normal_(self.team_one_query_embedding.weight, std=0.02)
        nn.init.normal_(self.team_two_query_embedding.weight, std=0.02)
        nn.init.normal_(self.ball_query_embedding.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in list(self.temporal_blocks) + list(self.agent_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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

    def _unpatchify(
        self, x: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """x: [B, A, N, patch_size*coord_dim] -> [B, T, A, coord_dim]."""
        b, a, n, d = x.shape
        expected = self.patch_size * self.coord_dim
        if d != expected:
            raise ValueError(f"Expected patch dim={expected}, got {d}")
        x = x.view(b, a, n, self.patch_size, self.coord_dim)
        x = x.reshape(b, a, n * self.patch_size, self.coord_dim)
        x = x[:, :, :target_len, :]
        return x.permute(0, 2, 1, 3).contiguous()

    def _apply_temporal_block(
        self,
        x: torch.Tensor,
        block: DiTBlock1D,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        b, a, n, d = x.shape
        x_flat = rearrange(x, "b a n d -> (b a) n d")
        c_flat = t_emb.repeat_interleave(a, dim=0)
        x_flat = block(x_flat, c_flat)
        return rearrange(x_flat, "(b a) n d -> b a n d", b=b, a=a)

    def _apply_agent_block(
        self,
        x: torch.Tensor,
        block: DiTBlock1D,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        b, a, n, d = x.shape
        x_flat = rearrange(x, "b a n d -> (b n) a d")
        c_flat = t_emb.repeat_interleave(n, dim=0)
        x_flat = block(x_flat, c_flat)
        return rearrange(x_flat, "(b n) a d -> b a n d", b=b, n=n)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, A, coord_dim]
            t: [B] diffusion timesteps
            context: [B, T, A, context_channels]
            mask: [B, T, A] optional observation mask concatenated to context
        Returns:
            eps prediction: [B, T, A, coord_dim]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x [B, T, A, C], got {tuple(x.shape)}")
        if context.ndim != 4:
            raise ValueError(
                f"Expected context [B, T, A, Cctx], got {tuple(context.shape)}"
            )
        b, t_len, a, c = x.shape
        if a != self.num_agents:
            raise ValueError(f"Expected num_agents={self.num_agents}, got {a}")
        if c != self.coord_dim:
            raise ValueError(f"Expected coord_dim={self.coord_dim}, got {c}")
        if context.shape[:3] != (b, t_len, a):
            raise ValueError(
                "Context must match x on [B, T, A], "
                f"got context shape {tuple(context.shape)}"
            )
        if context.shape[-1] != self.context_channels:
            raise ValueError(
                f"Expected context_channels={self.context_channels}, "
                f"got {context.shape[-1]}"
            )
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1 or t.shape[0] != b:
            raise ValueError(f"Expected t [B], got {tuple(t.shape)}")

        if mask is None:
            mask_in = torch.ones(
                b, t_len, a, device=context.device, dtype=context.dtype
            )
        else:
            if mask.shape != (b, t_len, a):
                raise ValueError(
                    f"Expected mask [B, T, A]={b, t_len, a}, got {tuple(mask.shape)}"
                )
            mask_in = mask.to(context.dtype)
        context = torch.cat([context, mask_in.unsqueeze(-1)], dim=-1)

        x_flat = rearrange(x, "b t a c -> (b a) t c")
        c_flat = rearrange(context, "b t a c -> (b a) t c")
        x_tok = self.x_embedder(x_flat)
        c_tok = self.c_embedder(c_flat)

        x_tok = rearrange(x_tok, "(b a) n d -> b a n d", b=b, a=a)
        c_tok = rearrange(c_tok, "(b a) n d -> b a n d", b=b, a=a)

        x_tok = x_tok + c_tok
        x_temporal = rearrange(x_tok, "b a n d -> (b a) n d")
        x_temporal = self.pos_emb(x_temporal)
        x_tok = rearrange(x_temporal, "(b a) n d -> b a n d", b=b, a=a)
        t_emb = self.t_embedder(t)
        agent_embed = self._agent_query_embedding(x).view(1, a, 1, -1)

        used_agent_stack = False
        pair_count = min(len(self.temporal_blocks), len(self.agent_blocks))
        for i in range(pair_count):
            x_tok = self._apply_temporal_block(x_tok, self.temporal_blocks[i], t_emb)
            if not used_agent_stack:
                x_tok = x_tok + agent_embed
                used_agent_stack = True
            x_tok = self._apply_agent_block(x_tok, self.agent_blocks[i], t_emb)

        for i in range(pair_count, len(self.temporal_blocks)):
            x_tok = self._apply_temporal_block(x_tok, self.temporal_blocks[i], t_emb)
        if len(self.agent_blocks) > pair_count and not used_agent_stack:
            x_tok = x_tok + agent_embed
            used_agent_stack = True
        for i in range(pair_count, len(self.agent_blocks)):
            x_tok = self._apply_agent_block(x_tok, self.agent_blocks[i], t_emb)

        x_final = rearrange(x_tok, "b a n d -> (b a) n d")
        c_final = t_emb.repeat_interleave(a, dim=0)
        x_final = self.final_layer(x_final, c_final)
        x_final = rearrange(x_final, "(b a) n d -> b a n d", b=b, a=a)
        return self._unpatchify(x_final, target_len=t_len)
