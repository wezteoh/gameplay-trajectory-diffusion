"""Evaluate ADE/FDE/JADE/JFDE on a validation set from a Lightning checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import sample_trajectory_ddpm as sd  # noqa: E402
from src.inference.trajectory_sample import (  # noqa: E402
    ground_truth_normalized_filling,
    model_name,
    sample_batch_multi_path,
)
from src.utils.trajectory_coords import denormalize_court_xy  # noqa: E402
from src.utils.trajectory_metrics import (  # noqa: E402
    finalize_metric_state,
    init_metric_state,
    l2_distances,
    update_metric_state,
)
from train import _build_datamodule, _build_module  # noqa: E402

_DATA_PRESETS = tuple(sorted(p.stem for p in (_REPO_ROOT / "configs" / "data").glob("*.yaml")))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADE/FDE/JADE/JFDE on validation data from a checkpoint (court units).",
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Lightning .ckpt path.")
    p.add_argument(
        "--data",
        type=str,
        default="trajectory_nba_filling_test",
        choices=list(_DATA_PRESETS) if _DATA_PRESETS else None,
        help="Override configs/data/<name>.yaml (default: checkpoint config.yaml data).",
    )
    p.add_argument(
        "--num-paths",
        type=int,
        default=20,
        help="Stochastic samples per validation row (default: 20).",
    )
    p.add_argument(
        "--horizon-stride",
        type=int,
        default=5,
        help="Cumulative horizon stride for metric keys (default: 20).",
    )
    p.add_argument(
        "--metrics-start-t",
        type=int,
        default=10,
        help=(
            "First timestep index (0-based) included in ADE/JADE/FDE/JFDE. "
            "Metrics use only times [t, T). Default 0 = full sequence."
        ),
    )
    p.add_argument(
        "--prediction-mode",
        type=str,
        default="blended",
        choices=("blended", "pure"),
        help=(
            "blended: obs/GT blend for positions (masked steps roll out from last observed). "
            "pure: position_0 + cumsum(predicted deltas), no blend. "
            "Both use full horizon then --metrics-start-t slice for ADE/FDE."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of validation rows (after shuffle=False order). "
        "Default: full val set (already capped by data max_val_samples).",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    p.add_argument("--no-ema", action="store_true", help="Do not apply EMA weights.")
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override classifier-free guidance scale (filling).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: checkpoint config seed).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print DDPM sampling progress (slow).",
    )
    p.add_argument(
        "--sampling-method",
        type=str,
        default="dpm",
        choices=("ancestral", "dpm", "ddim"),
        help="Override cfg.sampling.method (ancestral, dpm, or ddim).",
    )
    p.add_argument(
        "--dpm-steps",
        type=int,
        default=40,
        help="Override cfg.sampling.dpm.steps for DPM-Solver.",
    )
    p.add_argument(
        "--dpm-order",
        type=int,
        default=3,
        help="Override cfg.sampling.dpm.order for DPM-Solver.",
    )
    p.add_argument(
        "--ddim-steps",
        type=int,
        default=500,
        help="Override cfg.sampling.ddim.steps when --sampling-method=ddim (default: from config or 50).",
    )
    p.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="Override cfg.sampling.ddim.eta when --sampling-method=ddim (0 = deterministic).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the per-sample progress bar (e.g. for logs/CI).",
    )
    return p.parse_args()


def _val_sample_total(val_loader: Any, max_samples: int | None) -> int | None:
    """Expected number of validation rows we will score (for progress total)."""
    try:
        n = len(val_loader.dataset)
    except TypeError:
        return None
    if max_samples is not None:
        return min(int(max_samples), int(n))
    return int(n)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    ckpt_path = Path(os.path.abspath(os.path.expanduser(args.checkpoint)))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg_path = sd._checkpoint_config_path(ckpt_path)
    cfg: DictConfig = sd._load_run_config(cfg_path)

    if args.data is not None:
        sd._merge_data_cfg(cfg, sd._load_data_preset(args.data))
    OmegaConf.resolve(cfg)
    sd._merge_sampling_cli(cfg, args)
    sd._validate_task_shapes(cfg)
    sd._ensure_nba_data_root(cfg)

    seed = int(cfg.seed) if args.seed is None else int(args.seed)
    pl.seed_everything(seed, workers=True)

    device = sd._resolve_device(cfg, force_cpu=bool(args.cpu))
    module = _build_module(cfg)
    sd._load_checkpoint(module, str(ckpt_path), map_location=device)
    module = module.to(device)
    sd._maybe_apply_ema(module, use_ema=not args.no_ema)
    module.eval()

    mname = model_name(cfg)
    if mname != "trajectory_filling_ddpm":
        raise ValueError(f"Unsupported model.name: {mname!r}; expected 'trajectory_filling_ddpm'")
    datamodule = _build_datamodule(cfg.data)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()

    state = init_metric_state()
    processed = 0
    max_s = args.max_samples
    total_expect = _val_sample_total(val_loader, max_s)
    pbar = tqdm(
        total=total_expect,
        unit="sample",
        desc="Eval metrics",
        leave=True,
        disable=bool(args.no_progress),
    )

    for batch in val_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        bs = next(iter(batch.values())).shape[0]
        if max_s is not None:
            remain = max_s - processed
            if remain <= 0:
                break
            if remain < bs:
                batch = {k: v[:remain] if torch.is_tensor(v) else v for k, v in batch.items()}
                bs = remain

        gt_n = ground_truth_normalized_filling(module, batch)
        t_g = int(gt_n.shape[1])
        ts = int(args.metrics_start_t)
        if ts < 0 or ts >= t_g:
            raise ValueError(f"--metrics-start-t must satisfy 0 <= t < T; got t={ts}, T={t_g}.")

        mode = str(args.prediction_mode).strip().lower()
        if mode not in ("blended", "pure"):
            raise ValueError(f"prediction_mode must be blended or pure; got {mode!r}")

        cw = float(module.court_width)
        ch = float(module.court_height)
        samples_norm = sample_batch_multi_path(
            module,
            cfg,
            batch,
            device,
            int(args.num_paths),
            position_mode=mode,
            guidance_scale_override=args.guidance_scale,
            verbose=bool(args.verbose),
        )
        t_s = int(samples_norm.shape[2])
        if t_s != t_g:
            raise ValueError(f"Prediction length T={t_s} != ground-truth T={t_g} for task {mname}.")
        if ts > 0:
            samples_norm = samples_norm[:, :, ts:, :, :]
        gt_slice = gt_n[:, ts:, :, :] if ts > 0 else gt_n
        samples_c = denormalize_court_xy(samples_norm, cw, ch)
        gt_c = denormalize_court_xy(gt_slice, cw, ch)

        dist = l2_distances(samples_c, gt_c)
        update_metric_state(
            state,
            dist,
            horizon_stride=int(args.horizon_stride),
        )
        processed += bs
        pbar.update(bs)

    pbar.close()

    if processed == 0:
        raise RuntimeError("No validation batches processed.")

    final = finalize_metric_state(state)
    print(json.dumps(final, indent=2, sort_keys=True))
    ts = int(args.metrics_start_t)
    print(f"metrics_start_t={ts}", flush=True)
    print(f"prediction_mode={str(args.prediction_mode).strip().lower()}", flush=True)
    print(f"rows_evaluated={processed}", flush=True)


if __name__ == "__main__":
    main()
