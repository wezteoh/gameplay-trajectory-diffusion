"""NBA trajectories for masked coordinate filling; target = normalized frame deltas."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.trajectory_masks import TrainMasking, masking_config_from_params
from src.utils.trajectory_coords import normalize_court_xy, normalize_delta


def _resolve_path(root_dir: str | None, path: str) -> str:
    p = str(path).strip()
    if os.path.isabs(p):
        return os.path.normpath(p)
    base = str(root_dir).strip() if root_dir else os.getcwd()
    return os.path.normpath(os.path.join(base, p))


def _collate_filling(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def _masking_as_plain_dict(masking: Any) -> dict[str, Any]:
    if isinstance(masking, DictConfig):
        out = OmegaConf.to_container(masking, resolve=True)
        if not isinstance(out, dict):
            raise TypeError(f"masking must be a dict, got {type(out)}")
        return out
    if isinstance(masking, dict):
        return dict(masking)
    raise TypeError(f"masking must be a dict or DictConfig, got {type(masking)}")


class NBATrajectoryFillingDataset(Dataset):
    """Loads (N, T, A, 2).

    Yields deltas [T, A, 2] with index 0 zero; context is [T, A, 2] or [T, A, 4]
    when include_delta_in_context is True (XY + normalized deltas); mask length T.
    """

    def __init__(
        self,
        npy_path: str,
        full_seq_len: int,
        num_agents: int,
        coord_dim: int,
        court_width: float,
        court_height: float,
        trajectory_key: str,
        context_key: str,
        mask_key: str,
        position_0_key: str,
        context_fill: Sequence[float],
        delta_shift: Sequence[float],
        delta_scale: Sequence[float],
        train_masking: TrainMasking,
        val_mask_mmap: np.ndarray | None,
        mask_rng_seed: int | None,
        is_train_split: bool,
        include_delta_in_context: bool = False,
        delta_context_fill: Sequence[float] | None = None,
        max_samples: int | None = None,
    ) -> None:
        if int(coord_dim) != 2:
            msg = f"coord_dim must be 2 for court XY, got {coord_dim}"
            raise ValueError(msg)
        self.npy_path = str(npy_path)
        self.full_seq_len = int(full_seq_len)
        self.delta_len = self.full_seq_len
        if self.delta_len < 1:
            raise ValueError(f"full_seq_len must be >= 1, got {full_seq_len}")
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        self.court_width = float(court_width)
        self.court_height = float(court_height)
        self.trajectory_key = str(trajectory_key)
        self.context_key = str(context_key)
        self.mask_key = str(mask_key)
        self.position_0_key = str(position_0_key)
        self._train_masking = train_masking
        self._val_mask_mmap = val_mask_mmap
        self._is_train_split = bool(is_train_split)
        self._mask_rng_seed = mask_rng_seed
        self._mask_gen = torch.Generator()
        if mask_rng_seed is not None:
            self._mask_gen.manual_seed(int(mask_rng_seed))

        cf = list(context_fill)
        if len(cf) != 2:
            raise ValueError(f"context_fill must have length 2, got {cf}")
        self._context_fill = torch.tensor(cf, dtype=torch.float32)

        ds = list(delta_shift)
        sc = list(delta_scale)
        if len(ds) != 2 or len(sc) != 2:
            raise ValueError("delta_shift and delta_scale must have length 2")
        self._delta_shift = torch.tensor(ds, dtype=torch.float32)
        self._delta_scale = torch.tensor(sc, dtype=torch.float32)

        self.include_delta_in_context = bool(include_delta_in_context)
        if self.include_delta_in_context:
            if delta_context_fill is None:
                raise ValueError(
                    "delta_context_fill is required when "
                    "include_delta_in_context is True"
                )
            dcf = list(delta_context_fill)
            if len(dcf) != 2:
                raise ValueError(f"delta_context_fill must have length 2, got {dcf}")
            self._delta_context_fill = torch.tensor(dcf, dtype=torch.float32)
        else:
            self._delta_context_fill = None

        self._arr = np.load(self.npy_path, mmap_mode="r")
        if self._arr.ndim != 4:
            shp = self._arr.shape
            raise ValueError(f"Expected .npy (N,T,P,C), got shape {shp}")
        n, t, p, c = self._arr.shape
        if t != self.full_seq_len or p != self.num_agents or c != self.coord_dim:
            raise ValueError(
                f"Expected (*,{self.full_seq_len},{self.num_agents},"
                f"{self.coord_dim}), got {self._arr.shape}"
            )
        if max_samples is not None:
            cap = int(max_samples)
            if cap <= 0:
                self._arr = self._arr[:0]
            elif cap < n:
                self._arr = self._arr[:cap]

        if self._val_mask_mmap is not None:
            vm = self._val_mask_mmap
            if vm.ndim != 3:
                raise ValueError(f"val masks must be (N,T,A), got shape {vm.shape}")
            nv, tv, av = vm.shape
            if tv != self.full_seq_len or av != self.num_agents:
                raise ValueError(
                    f"val masks expected (*,{self.full_seq_len},{self.num_agents}), "
                    f"got {vm.shape}"
                )
            n_rows = int(self._arr.shape[0])
            if nv < n_rows:
                raise ValueError(
                    f"val_mask first dim ({nv}) < dataset length ({n_rows})"
                )
            if nv > n_rows:
                self._val_mask_mmap = vm[:n_rows]
            else:
                self._val_mask_mmap = vm

    def reseed_mask_generator_for_worker(self, worker_id: int) -> None:
        """Reseed mask RNG per dataloader worker (via ``worker_init_fn``)."""
        if self._mask_rng_seed is not None:
            self._mask_gen.manual_seed(
                int(self._mask_rng_seed) + int(worker_id) * 1_000_003
            )

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def _observation_mask_for_index(
        self, index: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        use_file = not self._is_train_split and self._val_mask_mmap is not None
        if use_file:
            row = np.asarray(self._val_mask_mmap[index], dtype=np.float32)
            return torch.tensor(row, dtype=dtype, device=device)
        return self._train_masking.sample_mask(
            self.full_seq_len,
            self.num_agents,
            dtype,
            self._mask_gen,
        ).to(device=device)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = np.asarray(self._arr[index], dtype=np.float32)
        x = torch.tensor(row)
        norm = normalize_court_xy(
            x,
            court_width=self.court_width,
            court_height=self.court_height,
        )
        raw_delta = torch.zeros(
            self.full_seq_len,
            self.num_agents,
            self.coord_dim,
            dtype=norm.dtype,
            device=norm.device,
        )
        if self.full_seq_len > 1:
            raw_delta[1:] = norm[1:] - norm[:-1]

        fill = self._context_fill.to(dtype=norm.dtype, device=norm.device)
        full_m = self._observation_mask_for_index(index, norm.dtype, norm.device)
        m_exp = full_m.unsqueeze(-1)
        ctx_xy = norm * m_exp + fill.view(1, 1, 2) * (1.0 - m_exp)

        shift = self._delta_shift.to(dtype=raw_delta.dtype, device=raw_delta.device)
        scale = self._delta_scale.to(dtype=raw_delta.dtype, device=raw_delta.device)
        target = normalize_delta(raw_delta, shift, scale)

        if self.include_delta_in_context:
            norm_delta = normalize_delta(raw_delta, shift, scale)
            dfill = self._delta_context_fill.to(
                dtype=norm_delta.dtype, device=norm_delta.device
            )
            delta_ctx = norm_delta * m_exp + dfill.view(1, 1, 2) * (1.0 - m_exp)
            ctx_full = torch.cat([ctx_xy, delta_ctx], dim=-1)
        else:
            ctx_full = ctx_xy
        return {
            self.trajectory_key: target,
            self.context_key: ctx_full,
            self.mask_key: full_m,
            self.position_0_key: norm[0],
        }


class NBATrajectoryFillingDataModule(pl.LightningDataModule):
    """Train from train_path; validate from val_path."""

    def __init__(
        self,
        train_batch_size: int,
        full_seq_len: int,
        num_agents: int,
        coord_dim: int,
        court_width: float,
        court_height: float,
        train_path: str,
        val_path: str,
        context_fill: Sequence[float],
        delta_shift: Sequence[float],
        delta_scale: Sequence[float],
        masking: dict[str, Any] | DictConfig,
        root_dir: str | None = None,
        trajectory_key: str = "trajectory",
        context_key: str = "context",
        mask_key: str = "obs_mask",
        position_0_key: str = "position_0",
        num_workers: int = 0,
        max_val_samples: int | None = None,
        val_batch_size: int | None = None,
        include_delta_in_context: bool = False,
        delta_context_fill: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = (
            int(val_batch_size) if val_batch_size is not None else self.train_batch_size
        )
        self.full_seq_len = int(full_seq_len)
        self.delta_len = self.full_seq_len
        if self.delta_len < 1:
            raise ValueError("full_seq_len must be >= 1")
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        self.court_width = float(court_width)
        self.court_height = float(court_height)
        self.train_path = _resolve_path(root_dir, train_path)
        self.val_path = _resolve_path(root_dir, val_path)
        self.context_fill = list(context_fill)
        self.delta_shift = list(delta_shift)
        self.delta_scale = list(delta_scale)
        masking_plain = _masking_as_plain_dict(masking)
        self._train_masking, raw_val_mask = masking_config_from_params(
            {"masking": masking_plain}
        )
        self._val_mask_path = (
            _resolve_path(root_dir, raw_val_mask) if raw_val_mask else None
        )
        rs = masking_plain.get("mask_rng_seed")
        self._mask_rng_seed = None if rs is None else int(rs)
        self.trajectory_key = str(trajectory_key)
        self.context_key = str(context_key)
        self.mask_key = str(mask_key)
        self.position_0_key = str(position_0_key)
        self.num_workers = int(num_workers)
        self.max_val_samples = max_val_samples
        self.include_delta_in_context = bool(include_delta_in_context)
        if delta_context_fill is not None:
            self.delta_context_fill = list(delta_context_fill)
        elif self.include_delta_in_context:
            self.delta_context_fill = [0.0, 0.0]
        else:
            self.delta_context_fill = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        val_mmap: np.ndarray | None = None
        if self._val_mask_path:
            val_mmap = np.load(self._val_mask_path, mmap_mode="r")
            if self.max_val_samples is not None:
                cap = int(self.max_val_samples)
                if cap > 0 and val_mmap.shape[0] > cap:
                    val_mmap = val_mmap[:cap]

        self._train_ds = NBATrajectoryFillingDataset(
            npy_path=self.train_path,
            full_seq_len=self.full_seq_len,
            num_agents=self.num_agents,
            coord_dim=self.coord_dim,
            court_width=self.court_width,
            court_height=self.court_height,
            trajectory_key=self.trajectory_key,
            context_key=self.context_key,
            mask_key=self.mask_key,
            position_0_key=self.position_0_key,
            context_fill=self.context_fill,
            delta_shift=self.delta_shift,
            delta_scale=self.delta_scale,
            train_masking=self._train_masking,
            val_mask_mmap=None,
            mask_rng_seed=self._mask_rng_seed,
            is_train_split=True,
            include_delta_in_context=self.include_delta_in_context,
            delta_context_fill=self.delta_context_fill,
        )

        self._val_ds = NBATrajectoryFillingDataset(
            npy_path=self.val_path,
            full_seq_len=self.full_seq_len,
            num_agents=self.num_agents,
            coord_dim=self.coord_dim,
            court_width=self.court_width,
            court_height=self.court_height,
            trajectory_key=self.trajectory_key,
            context_key=self.context_key,
            mask_key=self.mask_key,
            position_0_key=self.position_0_key,
            context_fill=self.context_fill,
            delta_shift=self.delta_shift,
            delta_scale=self.delta_scale,
            train_masking=self._train_masking,
            val_mask_mmap=val_mmap,
            mask_rng_seed=self._mask_rng_seed,
            is_train_split=False,
            include_delta_in_context=self.include_delta_in_context,
            delta_context_fill=self.delta_context_fill,
            max_samples=self.max_val_samples,
        )

    def _mask_worker_init_fn(self, worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        if isinstance(ds, NBATrajectoryFillingDataset):
            ds.reseed_mask_generator_for_worker(worker_id)

    def train_dataloader(self) -> DataLoader:
        kw: dict = {
            "batch_size": self.train_batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "collate_fn": _collate_filling,
        }
        if self.num_workers > 0 and self._mask_rng_seed is not None:
            kw["worker_init_fn"] = self._mask_worker_init_fn
        return DataLoader(self._train_ds, **kw)

    def val_dataloader(self) -> DataLoader:
        kw: dict = {
            "batch_size": self.val_batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": _collate_filling,
        }
        if self.num_workers > 0 and self._mask_rng_seed is not None:
            kw["worker_init_fn"] = self._mask_worker_init_fn
        return DataLoader(self._val_ds, **kw)
