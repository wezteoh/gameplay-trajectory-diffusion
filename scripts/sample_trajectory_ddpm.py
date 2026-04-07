"""Generate trajectory DDPM samples from a Lightning checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train import _build_module  # noqa: E402
from src.utils.drawing import (  # noqa: E402
    create_frames_from_trajectory,
    create_video_from_frames,
)
from src.utils.trajectory_coords import denormalize_court_xy_numpy  # noqa: E402

_DATA_PRESETS = ("trajectory_synthetic", "trajectory_nba")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample trajectories from a saved DDPM checkpoint. "
        "Model/trainer/optimizer settings come from config.yaml in the "
        "checkpoint directory (written at train time).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a Lightning .ckpt file.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of trajectories to generate.",
    )
    p.add_argument(
        "--data",
        type=str,
        default=None,
        choices=list(_DATA_PRESETS),
        help=(
            "Data config to use (YAML under configs/data/). "
            "Must match the checkpoint's seq_len, num_agents, coord_dim. "
            "If omitted, the data section from the checkpoint "
            "config.yaml is used as-is."
        ),
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default="samples",
        help="Subdirectory under tmp/ to write outputs (default: samples).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: seed from checkpoint config).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    p.add_argument(
        "--no-ema",
        action="store_true",
        help="Do not copy EMA weights into the model before sampling.",
    )
    p.add_argument(
        "--no-npz",
        action="store_true",
        help="Skip writing trajectory_*.npz files.",
    )
    p.add_argument(
        "--save-videos",
        action="store_true",
        help="Also write trajectory_*.mp4 court visualizations.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step DDPM sampling progress.",
    )
    return p.parse_args()


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    d = checkpoint_path.resolve().parent
    return d / "config.yaml"


def _load_run_config(cfg_path: Path) -> DictConfig:
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Expected training config next to the checkpoint: {cfg_path!s}. "
            "Train with train.py so config.yaml is saved in the checkpoint dir."
        )
    return OmegaConf.load(cfg_path)


def _load_data_preset(name: str) -> DictConfig:
    path = _REPO_ROOT / "configs" / "data" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Data config not found: {path}")
    return OmegaConf.load(path)


def _merge_data_cfg(cfg: DictConfig, data: DictConfig) -> None:
    OmegaConf.resolve(data)
    cfg.data = data


def _validate_backbone_shapes(cfg: DictConfig) -> None:
    backbone = cfg.model.backbone
    d = cfg.data
    for key in ("seq_len", "num_agents", "coord_dim"):
        if int(backbone[key]) != int(d[key]):
            raise ValueError(
                f"data.{key}={d[key]} does not match model.backbone.{key}="
                f"{backbone[key]} from the checkpoint config. "
                "Choose a --data preset that matches the trained shapes."
            )


def _resolve_device(cfg: DictConfig, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    use_gpu = bool(cfg.hardware.use_gpu)
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "hardware.use_gpu=true in checkpoint config but CUDA is not available. "
            "Pass --cpu to sample on CPU."
        )
    if use_gpu:
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _normalize_output_subdir(subdir: str) -> str:
    s = str(subdir).strip().replace("\\", "/").strip("/")
    if not s:
        raise ValueError("--output-subdir must be non-empty.")
    if ".." in s.split("/"):
        raise ValueError("--output-subdir must not contain '..'.")
    return s


def _load_checkpoint(
    module: Any,
    checkpoint_path: str,
    map_location: torch.device | str,
) -> dict[str, Any]:
    ckpt = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            "Expected a Lightning checkpoint with 'state_dict', "
            f"got: {checkpoint_path!r}"
        )
    module.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt


def _maybe_apply_ema(module: Any, use_ema: bool) -> None:
    if not use_ema or getattr(module, "ema", None) is None:
        return
    module.ema.copy_to(module.model)


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(os.path.abspath(os.path.expanduser(args.checkpoint)))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg_path = _checkpoint_config_path(ckpt_path)
    cfg = _load_run_config(cfg_path)

    if args.data is not None:
        _merge_data_cfg(cfg, _load_data_preset(args.data))
    _validate_backbone_shapes(cfg)

    seed = int(cfg.seed) if args.seed is None else int(args.seed)
    pl.seed_everything(seed, workers=True)

    device = _resolve_device(cfg, force_cpu=bool(args.cpu))
    module = _build_module(cfg)
    _load_checkpoint(module, str(ckpt_path), map_location=device)
    module = module.to(device)
    _maybe_apply_ema(module, use_ema=not args.no_ema)
    module.eval()

    out_sub = _normalize_output_subdir(str(args.output_subdir))
    out_dir = os.path.join(str(_REPO_ROOT), "tmp", out_sub)
    os.makedirs(out_dir, exist_ok=True)

    n = int(args.num_samples)
    with torch.no_grad():
        x = module.model.sample(
            batch_size=n,
            seq_len=int(module.seq_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            verbose=bool(args.verbose),
        )

    x_np = x.float().cpu().numpy()
    court_w = float(module.court_width)
    court_h = float(module.court_height)

    if not args.no_npz:
        for i in range(n):
            court_xy = denormalize_court_xy_numpy(x_np[i], court_w, court_h)
            stem = f"trajectory_{i:04d}.npz"
            path = os.path.join(out_dir, stem)
            np.savez(
                path,
                trajectory_normalized=x_np[i].astype(np.float32),
                trajectory_court_xy=court_xy.astype(np.float32),
                seq_len=int(module.seq_len),
                num_agents=int(module.num_agents),
                coord_dim=int(module.coord_dim),
                court_width=court_w,
                court_height=court_h,
            )

    if args.save_videos:
        for i in range(n):
            court_xy = denormalize_court_xy_numpy(x_np[i], court_w, court_h)
            frames = create_frames_from_trajectory(court_xy, "basketball")
            video_path = os.path.join(out_dir, f"trajectory_{i:04d}.mp4")
            create_video_from_frames(frames, video_path, fps=10)

    cfg_dump = os.path.join(out_dir, "config.yaml")
    with open(cfg_dump, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Saved {n} sample(s) under {out_dir}/")
    if not args.no_npz:
        print(f"Wrote trajectory npz files and {cfg_dump}")


if __name__ == "__main__":
    main()
