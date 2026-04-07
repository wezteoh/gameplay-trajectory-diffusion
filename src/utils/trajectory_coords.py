"""Court XY in [0, W] x [0, H] <-> normalized [-2, 2] per axis."""

from __future__ import annotations

import torch


def normalize_court_xy(
    xy: torch.Tensor,
    court_width: float,
    court_height: float,
) -> torch.Tensor:
    """Map last dim (x, y) from [0, W] x [0, H] to [-2, 2] x [-2, 2]."""
    w = float(court_width)
    h = float(court_height)
    scale = torch.tensor(
        [4.0 / w, 4.0 / h],
        device=xy.device,
        dtype=xy.dtype,
    )
    bias = torch.tensor([-2.0, -2.0], device=xy.device, dtype=xy.dtype)
    return xy * scale + bias


def denormalize_court_xy(
    xy: torch.Tensor,
    court_width: float,
    court_height: float,
) -> torch.Tensor:
    """Inverse of normalize_court_xy."""
    w = float(court_width)
    h = float(court_height)
    scale = torch.tensor(
        [w / 4.0, h / 4.0],
        device=xy.device,
        dtype=xy.dtype,
    )
    bias = torch.tensor([2.0, 2.0], device=xy.device, dtype=xy.dtype)
    return (xy + bias) * scale


def denormalize_court_xy_numpy(
    xy,
    court_width: float,
    court_height: float,
):
    """NumPy array [... , 2], same mapping as denormalize_court_xy."""
    import numpy as np

    xy = np.asarray(xy, dtype=np.float64)
    w, h = float(court_width), float(court_height)
    scale = np.array([w / 4.0, h / 4.0], dtype=np.float64)
    bias = np.array([2.0, 2.0], dtype=np.float64)
    return (xy + bias) * scale
