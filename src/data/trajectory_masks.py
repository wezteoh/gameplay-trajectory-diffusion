"""Train-time observation masks [T, A]: 1 = observed (GT in context), 0 = masked."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_BRANCH_KEYS = frozenset({"even", "agent"})


def _even_mask(
    seq_len: int,
    num_agents: int,
    prefix_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    t = int(seq_len)
    a = int(num_agents)
    m = torch.zeros(t, a, dtype=dtype)
    k = max(0, min(int(prefix_len), t))
    if k > 0:
        m[:k] = 1.0
    m[0, :] = 1.0
    return m


def _even_from_spec(
    seq_len: int,
    num_agents: int,
    spec: dict[str, Any],
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    mode = str(spec.get("mode", "fixed")).strip().lower()
    t = int(seq_len)
    if mode == "fixed":
        k = int(spec["visible_prefix_len"])
        return _even_mask(seq_len, num_agents, k, dtype)
    if mode == "random":
        low = int(spec["prefix_min"])
        high = int(spec["prefix_max"])
        if low > high:
            raise ValueError(f"prefix_min ({low}) > prefix_max ({high})")
        low_c = max(0, min(low, t))
        high_c = max(0, min(high, t))
        if low_c > high_c:
            raise ValueError(
                f"After clamping to seq_len={t}, prefix range is empty "
                f"({low_c}..{high_c})"
            )
        span = high_c - low_c + 1
        off = int(torch.randint(0, span, (1,), device="cpu", generator=gen).item())
        k = low_c + off
        return _even_mask(seq_len, num_agents, k, dtype)
    raise ValueError(f"even.mode must be 'fixed' or 'random', got {mode!r}")


def _agent_from_spec(
    seq_len: int,
    num_agents: int,
    spec: dict[str, Any],
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    mode = str(spec.get("mode", "fixed")).strip().lower()
    t = int(seq_len)
    a = int(num_agents)
    m = torch.ones(t, a, dtype=dtype)
    if mode == "fixed":
        n = int(spec["n_masked"])
    elif mode == "random":
        low = int(spec["n_masked_min"])
        high = int(spec["n_masked_max"])
        if low > high:
            raise ValueError(f"n_masked_min ({low}) > n_masked_max ({high})")
        low_c = max(0, min(low, a))
        high_c = max(0, min(high, a))
        if low_c > high_c:
            raise ValueError(
                f"After clamping to num_agents={a}, n_masked range is empty "
                f"({low_c}..{high_c})"
            )
        span = high_c - low_c + 1
        off = int(torch.randint(0, span, (1,), device="cpu", generator=gen).item())
        n = low_c + off
    else:
        raise ValueError(f"agent.mode must be 'fixed' or 'random', got {mode!r}")
    n = max(0, min(int(n), a))
    if n == 0 or t <= 1:
        return m
    perm = torch.randperm(a, device="cpu", generator=gen)
    idx = perm[:n]
    m[1:, idx] = 0.0
    return m


def _branch_kind_and_payload(node: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    keys = [k for k in _BRANCH_KEYS if k in node]
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one of {sorted(_BRANCH_KEYS)} in branch, got {keys}"
        )
    k = keys[0]
    payload = node[k]
    if not isinstance(payload, dict):
        raise ValueError(f"Branch {k!r} must map to a dict, got {type(payload)}")
    return k, payload


def build_mask_from_branch(
    node: dict[str, Any],
    seq_len: int,
    num_agents: int,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    """Build [T, A] mask from ``{\"even\": ...}`` or ``{\"agent\": ...}``."""
    kind, payload = _branch_kind_and_payload(node)
    if kind == "even":
        return _even_from_spec(seq_len, num_agents, payload, dtype, gen)
    return _agent_from_spec(seq_len, num_agents, payload, dtype, gen)


def _hybrid_from_spec(
    seq_len: int,
    num_agents: int,
    spec: dict[str, Any],
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    combine = str(spec.get("combine", "union")).strip().lower()
    if combine != "union":
        raise ValueError(f"hybrid.combine must be 'union', got {combine!r}")
    a_node = spec.get("a")
    b_node = spec.get("b")
    if not isinstance(a_node, dict) or not isinstance(b_node, dict):
        raise ValueError("hybrid.a and hybrid.b must be dicts")
    m1 = build_mask_from_branch(a_node, seq_len, num_agents, dtype, gen)
    m2 = build_mask_from_branch(b_node, seq_len, num_agents, dtype, gen)
    return torch.clamp(m1 + m2, max=1.0).to(dtype=dtype)


def build_mask_from_mixture_entry(
    entry: dict[str, Any],
    seq_len: int,
    num_agents: int,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    """One mixture row: exactly one of keys ``even``, ``agent``, ``hybrid``."""
    keys = [k for k in ("even", "agent", "hybrid") if k in entry]
    if len(keys) != 1:
        raise ValueError(
            f"Mixture entry must have exactly one of even|agent|hybrid, had {keys}"
        )
    key = keys[0]
    payload = entry[key]
    if not isinstance(payload, dict):
        raise ValueError(f"Mixture {key!r} payload must be a dict")
    if key == "even":
        return _even_from_spec(seq_len, num_agents, payload, dtype, gen)
    if key == "agent":
        return _agent_from_spec(seq_len, num_agents, payload, dtype, gen)
    return _hybrid_from_spec(seq_len, num_agents, payload, dtype, gen)


@dataclass(frozen=True)
class TrainMasking:
    """Normalized mixture for train-time mask sampling."""

    probs: torch.Tensor
    mixture_entries: tuple[dict[str, Any], ...]

    def sample_mask(
        self,
        seq_len: int,
        num_agents: int,
        dtype: torch.dtype,
        gen: torch.Generator,
    ) -> torch.Tensor:
        idx = int(
            torch.multinomial(self.probs, 1, replacement=True, generator=gen).item()
        )
        return build_mask_from_mixture_entry(
            self.mixture_entries[idx],
            seq_len,
            num_agents,
            dtype,
            gen,
        )


def parse_masking_config(raw: dict[str, Any]) -> tuple[TrainMasking, str | None]:
    """Parse ``data.params.masking`` (train mixture + optional val mask path)."""
    if not isinstance(raw, dict):
        raise TypeError(f"masking must be a dict, got {type(raw)}")
    train = raw.get("train")
    if not isinstance(train, dict):
        raise ValueError("masking.train is required and must be a dict")
    mix = train.get("mixture")
    if not isinstance(mix, list) or len(mix) == 0:
        raise ValueError("masking.train.mixture must be a non-empty list")
    weights: list[float] = []
    entries: list[dict[str, Any]] = []
    for i, row in enumerate(mix):
        if not isinstance(row, dict):
            raise ValueError(f"mixture[{i}] must be a dict")
        if "weight" not in row:
            raise ValueError(f"mixture[{i}] missing 'weight'")
        w = float(row["weight"])
        if w < 0:
            raise ValueError(f"mixture[{i}].weight must be >= 0, got {w}")
        weights.append(w)
        payload = {k: v for k, v in row.items() if k != "weight"}
        keys = [k for k in ("even", "agent", "hybrid") if k in payload]
        if len(keys) != 1:
            raise ValueError(
                f"mixture[{i}] must contain exactly one of even|agent|hybrid "
                f"(besides weight), got {keys}"
            )
        k = keys[0]
        entries.append({k: payload[k]})
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("mixture weights must sum to a positive value")
    probs = torch.tensor([w / s for w in weights], dtype=torch.float64)
    val_path = raw.get("val_mask_path")
    if val_path is None:
        vps: str | None = None
    else:
        vps = str(val_path).strip()
        if not vps:
            vps = None
    return TrainMasking(probs=probs, mixture_entries=tuple(entries)), vps


def masking_config_from_params(
    params: dict[str, Any],
) -> tuple[TrainMasking, str | None]:
    """Extract ``masking`` from datamodule kwargs (resolved Hydra dict)."""
    raw = params.get("masking")
    if raw is None:
        raise ValueError(
            "data.params.masking is required (keys: train.mixture, val_mask_path)"
        )
    if isinstance(raw, dict):
        return parse_masking_config(raw)
    raise TypeError(f"masking must be a dict, got {type(raw)}")
