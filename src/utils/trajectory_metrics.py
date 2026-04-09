"""ADE, FDE, JADE, JFDE metrics aligned with causaltraj ``compute_metrics``.

Tensor layout: ``samples`` is ``[B, P, T, A, C]``, ground truth ``gt`` is ``[B, T, A, C]``
(typically ``C=2``). Metrics match the same units as inputs (e.g. court feet after
denormalization). Callers may slice a suffix ``[..., t0:, ...]`` on both tensors before
``l2_distances`` to score only timesteps ``t0`` onward.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from einops import rearrange, repeat

Tensor = torch.Tensor

# Running aggregate: metric_name -> (sum, count) for correct multi-batch means.
MetricState = Dict[str, Tuple[float, float]]


def l2_distances(samples: Tensor, gt: Tensor) -> Tensor:
    """Pairwise L2 over coordinates: ``[B, P, T, A]``."""
    if samples.dim() != 5 or gt.dim() != 4:
        raise ValueError(
            f"Expected samples [B,P,T,A,C] and gt [B,T,A,C], got {samples.shape}, {gt.shape}"
        )
    gt_exp = repeat(gt, "b t a c -> b p t a c", p=samples.shape[1])
    return (samples - gt_exp).norm(p=2, dim=-1)


def _prefix_metrics_tensors(
    d: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return tensors before the final ``.mean()`` for one horizon prefix.

    ``d`` is ``[B, P, T', A]`` (already sliced). Order matches causaltraj reductions.
    """
    jade_path_agentwise = d.mean(dim=-2)
    jade_all = jade_path_agentwise.mean(dim=-1)

    jfde_path_agentwise = d[:, :, -1, :]
    jfde_all = jfde_path_agentwise.mean(dim=-1)

    ade_path_agentwise = d.mean(dim=-2)
    ade_agent_pathwise = rearrange(ade_path_agentwise, "b p a -> b a p")

    fde_path_agentwise = d[:, :, -1, :]
    fde_agent_pathwise = rearrange(fde_path_agentwise, "b p a -> b a p")
    return (
        jade_all,
        jfde_all,
        jade_all.min(dim=-1).values,
        jfde_all.min(dim=-1).values,
        ade_agent_pathwise.min(dim=-1).values,
        fde_agent_pathwise.min(dim=-1).values,
    )


def compute_metrics_from_distances(
    distances: Tensor,
    *,
    horizon_stride: int = 5,
) -> Dict[str, Tensor]:
    """Causaltraj-style metrics for each cumulative prefix horizon (single batch).

    ``distances`` is ``[B, P, T, A]``. Keys use ``{i + horizon_stride}frames`` like the
    reference when ``horizon_stride==5``.
    """
    if distances.dim() != 4:
        raise ValueError(f"Expected distances [B,P,T,A], got {distances.shape}")
    _, _, t, _ = distances.shape
    if horizon_stride <= 0:
        raise ValueError("horizon_stride must be positive")

    out: Dict[str, Tensor] = {}
    for i in range(0, t, horizon_stride):
        prefix = min(i + horizon_stride, t)
        d = distances[:, :, :prefix, :]
        jade_all, jfde_all, jade_min_t, jfde_min_t, ade_min_t, fde_min_t = (
            _prefix_metrics_tensors(d)
        )
        label = i + horizon_stride
        out[f"jade_mean_{label}frames"] = jade_all.mean()
        out[f"jade_min_{label}frames"] = jade_min_t.mean()
        out[f"jfde_mean_{label}frames"] = jfde_all.mean()
        out[f"jfde_min_{label}frames"] = jfde_min_t.mean()
        out[f"ade_min_{label}frames"] = ade_min_t.mean()
        out[f"fde_min_{label}frames"] = fde_min_t.mean()

    return out


def init_metric_state() -> MetricState:
    return {}


def update_metric_state(
    state: MetricState,
    distances: Tensor,
    *,
    horizon_stride: int = 5,
) -> None:
    """Add one batch of distances into global running sums (multi-batch safe)."""
    if distances.dim() != 4:
        raise ValueError(f"Expected distances [B,P,T,A], got {distances.shape}")
    b, p, t, a = distances.shape
    if horizon_stride <= 0:
        raise ValueError("horizon_stride must be positive")

    for i in range(0, t, horizon_stride):
        prefix = min(i + horizon_stride, t)
        d = distances[:, :, :prefix, :]
        jade_all, jfde_all, jade_min_t, jfde_min_t, ade_min_t, fde_min_t = (
            _prefix_metrics_tensors(d)
        )
        label = i + horizon_stride

        def add(key: str, sum_val: float, cnt: float) -> None:
            s, c = state.get(key, (0.0, 0.0))
            state[key] = (s + sum_val, c + cnt)

        add(f"jade_mean_{label}frames", jade_all.sum().item(), float(b * p))
        add(f"jade_min_{label}frames", jade_min_t.sum().item(), float(b))
        add(f"jfde_mean_{label}frames", jfde_all.sum().item(), float(b * p))
        add(f"jfde_min_{label}frames", jfde_min_t.sum().item(), float(b))
        add(f"ade_min_{label}frames", ade_min_t.sum().item(), float(b * a))
        add(f"fde_min_{label}frames", fde_min_t.sum().item(), float(b * a))


def finalize_metric_state(state: MetricState) -> Dict[str, float]:
    """Turn running (sum, count) into scalar means."""
    return {k: (s / c if c > 0 else float("nan")) for k, (s, c) in state.items()}


def compute_trajectory_metrics(
    samples: Tensor,
    gt: Tensor,
    *,
    horizon_stride: int = 5,
) -> Dict[str, Tensor]:
    """L2 distances then ``compute_metrics_from_distances``."""
    dist = l2_distances(samples, gt)
    return compute_metrics_from_distances(dist, horizon_stride=horizon_stride)
