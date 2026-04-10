"""Tests for trajectory observation-mask builders and dataset val-mask loading."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from src.data.nba_trajectory_filling import NBATrajectoryFillingDataset
from src.data.trajectory_masks import (
    build_mask_from_branch,
    build_mask_from_mixture_entry,
    parse_masking_config,
)


def test_even_fixed() -> None:
    gen = torch.Generator().manual_seed(0)
    m = build_mask_from_mixture_entry(
        {"even": {"mode": "fixed", "visible_prefix_len": 3}},
        seq_len=8,
        num_agents=4,
        dtype=torch.float32,
        gen=gen,
    )
    assert m.shape == (8, 4)
    assert torch.all(m[:3] == 1)
    assert torch.all(m[3:] == 0)
    assert torch.all(m[0] == 1)


def test_even_random_range() -> None:
    gen = torch.Generator().manual_seed(123)
    for _ in range(20):
        m = build_mask_from_mixture_entry(
            {"even": {"mode": "random", "prefix_min": 2, "prefix_max": 5}},
            seq_len=10,
            num_agents=3,
            dtype=torch.float32,
            gen=gen,
        )
        k = int(m[:, 0].sum().item())
        assert 2 <= k <= 5


def test_agent_fixed() -> None:
    gen = torch.Generator().manual_seed(1)
    m = build_mask_from_mixture_entry(
        {"agent": {"mode": "fixed", "n_masked": 2}},
        seq_len=6,
        num_agents=5,
        dtype=torch.float32,
        gen=gen,
    )
    sparse = torch.where((m[1:] == 0).all(dim=0))[0]
    full = torch.where((m[1:] == 1).all(dim=0))[0]
    assert sparse.numel() == 2
    assert full.numel() == 3
    assert torch.all(m[0] == 1)


def test_hybrid_union() -> None:
    gen = torch.Generator().manual_seed(2)
    m = build_mask_from_mixture_entry(
        {
            "hybrid": {
                "combine": "union",
                "a": {"even": {"mode": "fixed", "visible_prefix_len": 2}},
                "b": {"agent": {"mode": "fixed", "n_masked": 1}},
            }
        },
        seq_len=5,
        num_agents=3,
        dtype=torch.float32,
        gen=gen,
    )
    g2 = torch.Generator().manual_seed(2)
    m_even = build_mask_from_branch(
        {"even": {"mode": "fixed", "visible_prefix_len": 2}},
        5,
        3,
        torch.float32,
        g2,
    )
    m_ag = build_mask_from_branch(
        {"agent": {"mode": "fixed", "n_masked": 1}},
        5,
        3,
        torch.float32,
        g2,
    )
    expect = torch.clamp(m_even + m_ag, max=1.0)
    assert torch.allclose(m, expect)


def test_parse_mixture_and_sample_repro() -> None:
    raw = {
        "train": {
            "mixture": [
                {"weight": 1.0, "even": {"mode": "fixed", "visible_prefix_len": 4}},
            ]
        },
        "val_mask_path": None,
    }
    tm, vp = parse_masking_config(raw)
    assert vp is None
    gen = torch.Generator().manual_seed(7)
    m1 = tm.sample_mask(10, 11, torch.float32, gen)
    gen.manual_seed(7)
    m2 = tm.sample_mask(10, 11, torch.float32, gen)
    assert torch.allclose(m1, m2)


def test_val_mask_mmap_dataset_slice() -> None:
    T, A, C = 5, 3, 2
    n = 4
    traj = np.random.randn(n, T, A, C).astype(np.float32)
    masks = np.ones((n, T, A), dtype=np.float32)
    masks[:, 2:, :] = 0.0
    raw_tm, _ = parse_masking_config(
        {
            "train": {
                "mixture": [
                    {
                        "weight": 1.0,
                        "even": {"mode": "fixed", "visible_prefix_len": 1},
                    }
                ]
            }
        }
    )
    with tempfile.TemporaryDirectory() as td:
        tp = f"{td}/t.npy"
        np.save(tp, traj)
        ds = NBATrajectoryFillingDataset(
            npy_path=tp,
            full_seq_len=T,
            num_agents=A,
            coord_dim=C,
            court_width=94.0,
            court_height=50.0,
            trajectory_key="trajectory",
            context_key="context",
            mask_key="obs_mask",
            position_0_key="position_0",
            context_fill=[0.0, 0.0],
            delta_shift=[0.0, 0.0],
            delta_scale=[1.0, 1.0],
            train_masking=raw_tm,
            val_mask_mmap=masks,
            mask_rng_seed=None,
            is_train_split=False,
        )
        batch = ds[0]
        om = batch["obs_mask"]
        assert om.shape == (T, A)
        assert torch.allclose(om[:2], torch.ones(2, A))
        assert torch.allclose(om[2:], torch.zeros(T - 2, A))


def test_parse_errors() -> None:
    with pytest.raises(ValueError, match="mixture"):
        parse_masking_config({"train": {}})
    with pytest.raises(ValueError, match="exactly one"):
        parse_masking_config(
            {
                "train": {
                    "mixture": [
                        {"weight": 1.0, "even": {}, "agent": {}},
                    ]
                }
            }
        )
