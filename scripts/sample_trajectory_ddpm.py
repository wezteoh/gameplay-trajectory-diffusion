"""Generate trajectory DDPM samples from a Lightning checkpoint.

Supports all training task types: unconditional generation (trajectory_ddpm),
future completion (trajectory_completion_ddpm), and masked filling
(trajectory_filling_ddpm). Task behavior is taken from ``model.name`` in the
checkpoint's ``config.yaml``.
"""

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

from src.interfaces.trajectory_filling_ddpm import (  # noqa: E402
    _blend_traj_rollout_from_last_observed,
)
from src.utils.drawing import create_frames_from_trajectory, create_video_from_frames  # noqa: E402
from src.utils.trajectory_coords import denormalize_court_xy_numpy, denormalize_delta  # noqa: E402
from train import _build_datamodule, _build_module  # noqa: E402

_DATA_PRESETS = tuple(
    sorted(p.stem for p in (_REPO_ROOT / "configs" / "data").glob("*.yaml"))
)


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
        choices=list(_DATA_PRESETS) if _DATA_PRESETS else None,
        help=(
            "Data config stem (configs/data/<name>.yaml). "
            "Must match the checkpoint's shapes for that task. "
            "If omitted, the data section from the checkpoint config.yaml is used."
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
        "--save-evolution-videos",
        action="store_true",
        help=(
            "Also write evolution_*.mp4 showing denoising checkpoints for each sample."
        ),
    )
    p.add_argument(
        "--evolution-every",
        type=int,
        default=200,
        help=(
            "Capture interval (in denoising steps) for evolution videos; "
            "default 200."
        ),
    )
    p.add_argument(
        "--evolution-fps",
        type=int,
        default=10,
        help="FPS for evolution videos (default: 2).",
    )
    p.add_argument(
        "--filling-blend-videos",
        action="store_true",
        help=(
            "For trajectory_filling_ddpm only: also write trajectory_*_blend.mp4 "
            "(GT at observed steps + predicted deltas elsewhere), like validation."
        ),
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Override classifier-free guidance scale for conditional tasks "
            "(completion/filling). If omitted, uses value from checkpoint config."
        ),
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


def _model_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "model.name", default="trajectory_ddpm"))


def _validate_task_shapes(cfg: DictConfig) -> None:
    """Ensure resolved data.* matches model.backbone.* for the checkpoint's task."""
    mname = _model_name(cfg)
    bb = cfg.model.backbone
    d = cfg.data
    if mname == "trajectory_completion_ddpm":
        if int(bb.future_seq_len) != int(d.future_seq_len):
            raise ValueError(
                f"data.future_seq_len={d.future_seq_len} does not match "
                f"model.backbone.future_seq_len={bb.future_seq_len}."
            )
        if int(bb.past_seq_len) != int(d.observed_len):
            raise ValueError(
                f"data.observed_len={d.observed_len} does not match "
                f"model.backbone.past_seq_len={bb.past_seq_len}."
            )
        for key in ("num_agents", "coord_dim"):
            if int(bb[key]) != int(d[key]):
                raise ValueError(
                    f"data.{key}={d[key]} does not match model.backbone.{key}="
                    f"{bb[key]}."
                )
        return
    if mname == "trajectory_filling_ddpm":
        full_t = int(d.full_seq_len)
        delta_len = int(OmegaConf.select(d, "delta_len", default=full_t))
        if int(bb.max_seq_len) != delta_len:
            raise ValueError(
                f"data delta length ({delta_len}) does not match "
                f"model.backbone.max_seq_len={bb.max_seq_len}."
            )
        for key in ("num_agents", "coord_dim"):
            if int(bb[key]) != int(d[key]):
                raise ValueError(
                    f"data.{key}={d[key]} does not match model.backbone.{key}="
                    f"{bb[key]}."
                )
        return
    if mname != "trajectory_ddpm":
        raise ValueError(f"Unknown model.name for sampling: {mname!r}")
    for key in ("seq_len", "num_agents", "coord_dim"):
        if int(bb[key]) != int(d[key]):
            raise ValueError(
                f"data.{key}={d[key]} does not match model.backbone.{key}="
                f"{bb[key]} from the checkpoint config. "
                "Choose a --data preset that matches the trained shapes."
            )


def _ensure_nba_data_root(cfg: DictConfig) -> None:
    """Resolve train/val .npy paths relative to the repo root."""
    name = str(cfg.data.name)
    if name.startswith("trajectory_nba"):
        cfg.data.params.root_dir = str(_REPO_ROOT)


def _val_batch_first_n(
    datamodule: Any,
    n: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack validation batches until at least ``n`` rows, then trim."""
    datamodule.setup("validate")
    loader = datamodule.val_dataloader()
    chunks: list[dict[str, torch.Tensor]] = []
    total = 0
    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        chunks.append(batch)
        total += next(iter(batch.values())).shape[0]
        if total >= n:
            break
    if not chunks:
        raise RuntimeError("Validation dataloader returned no batches.")
    keys = chunks[0].keys()
    merged: dict[str, torch.Tensor] = {}
    for k in keys:
        merged[k] = torch.cat([c[k] for c in chunks], dim=0)[:n]
    return merged


@torch.no_grad()
def _sample_with_trace(
    model: Any,
    *,
    batch_size: int,
    seq_len: int,
    num_agents: int,
    coord_dim: int,
    device: torch.device,
    context: torch.Tensor | None = None,
    obs_mask: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    cfg_null_context: torch.Tensor | None = None,
    cfg_null_mask: torch.Tensor | None = None,
    trace_every: int | None = None,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Sampling with optional intermediate snapshots as (step, x_t)."""
    shape = (batch_size, seq_len, num_agents, coord_dim)
    x = torch.randn(shape, device=device, dtype=dtype)
    traces: list[tuple[int, torch.Tensor]] = []

    if trace_every is not None and trace_every > 0:
        traces.append((int(model.schedule.num_timesteps), x.detach().clone()))

    if context is None:
        context = torch.zeros(shape, device=device, dtype=dtype)
    gs = float(guidance_scale)
    use_cfg = gs != 1.0
    null_ctx = (
        torch.zeros_like(context) if cfg_null_context is None else cfg_null_context
    )
    accepts_mask = model._backbone_accepts_mask()
    if use_cfg and accepts_mask:
        if cfg_null_mask is not None:
            null_m = cfg_null_mask
        elif obs_mask is not None:
            null_m = torch.zeros_like(obs_mask, dtype=dtype, device=device)
        else:
            null_m = torch.zeros(
                batch_size, seq_len, num_agents, device=device, dtype=dtype
            )
    else:
        null_m = None

    if verbose:
        print("sampling")
    for step in reversed(range(model.schedule.num_timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        if use_cfg:
            eps_u = model._call_backbone(x, t, null_ctx, null_m)
            eps_c = model._call_backbone(x, t, context, obs_mask)
            noise_pred = eps_u + gs * (eps_c - eps_u)
        else:
            noise_pred = model._call_backbone(x, t, context, obs_mask)
        x = model.schedule.p_sample_step(x, t, noise_pred)
        if trace_every is not None and trace_every > 0:
            if step == 0 or (step % int(trace_every) == 0):
                traces.append((int(step), x.detach().clone()))
        if verbose:
            in_b = (x >= -2) & (x <= 2)
            percentage = in_b.all(dim=-1).float().mean() * 100
            print(f"step {step}: {percentage.item()}% within bounds")
    return x, traces


def _sample_for_task(
    module: Any,
    cfg: DictConfig,
    device: torch.device,
    n: int,
    verbose: bool,
    filling_blend: bool = False,
    guidance_scale_override: float | None = None,
    trace_every: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    """Run sampling; returns tensor for visualization (normalized positions) and metadata."""
    mname = _model_name(cfg)
    meta: dict[str, Any] = {"task": mname}

    if mname == "trajectory_ddpm":
        x, traces = _sample_with_trace(
            module.model,
            batch_size=n,
            seq_len=int(module.seq_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            verbose=verbose,
            trace_every=trace_every,
        )
        meta["seq_len"] = int(module.seq_len)
        meta["evolution_traces"] = traces
        return x, meta

    _ensure_nba_data_root(cfg)
    dm = _build_datamodule(cfg.data)
    batch = _val_batch_first_n(dm, n, device)

    if mname == "trajectory_completion_ddpm":
        ctx_key = str(module.context_key)
        past = batch[ctx_key]
        gs = (
            float(module.guidance_scale)
            if guidance_scale_override is None
            else float(guidance_scale_override)
        )
        future_hat, future_traces = _sample_with_trace(
            module.model,
            batch_size=n,
            seq_len=int(module.future_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            context=past,
            guidance_scale=gs,
            dtype=past.dtype,
            verbose=verbose,
            trace_every=trace_every,
        )
        anchor = past[:, -1]
        future_abs = future_hat + anchor.unsqueeze(1)
        x = torch.cat([past, future_abs], dim=1)
        evo: list[tuple[int, torch.Tensor]] = []
        for step, ft in future_traces:
            evo.append((step, torch.cat([past, ft + anchor.unsqueeze(1)], dim=1)))
        meta["observed_len"] = int(module.observed_len)
        meta["future_len"] = int(module.future_len)
        meta["full_seq_len"] = int(module.full_seq_len)
        meta["seq_len"] = int(module.full_seq_len)
        meta["evolution_traces"] = evo
        return x, meta

    if mname == "trajectory_filling_ddpm":
        ctx_key = str(module.context_key)
        mask_key = str(module.mask_key)
        pos_key = str(module.position_0_key)
        context = batch[ctx_key]
        obs_mask = batch[mask_key].to(dtype=context.dtype)
        pos0 = batch[pos_key]
        dtype = context.dtype
        gs = (
            float(module.guidance_scale)
            if guidance_scale_override is None
            else float(guidance_scale_override)
        )
        fill = module._context_fill_brc.to(device=device, dtype=dtype).expand(
            n, int(module.seq_len), int(module.num_agents), int(module.context_channels)
        )
        null_mask = torch.zeros(
            n,
            int(module.seq_len),
            int(module.num_agents),
            device=device,
            dtype=dtype,
        )
        deltas_hat, delta_traces = _sample_with_trace(
            module.model,
            batch_size=n,
            seq_len=int(module.seq_len),
            num_agents=int(module.num_agents),
            coord_dim=int(module.coord_dim),
            device=device,
            context=context,
            obs_mask=obs_mask,
            guidance_scale=gs,
            dtype=dtype,
            verbose=verbose,
            cfg_null_context=fill,
            cfg_null_mask=null_mask,
            trace_every=trace_every,
        )
        d_raw = denormalize_delta(
            deltas_hat,
            module._delta_shift,
            module._delta_scale,
        )
        x = pos0.unsqueeze(1) + torch.cumsum(d_raw, dim=1)
        evo: list[tuple[int, torch.Tensor]] = []
        for step, d_hat in delta_traces:
            d_raw_i = denormalize_delta(
                d_hat,
                module._delta_shift,
                module._delta_scale,
            )
            evo.append((step, pos0.unsqueeze(1) + torch.cumsum(d_raw_i, dim=1)))
        meta["seq_len"] = int(module.seq_len)
        meta["blend_extra"] = None
        meta["evolution_traces"] = evo
        if filling_blend:
            x0_true = batch[str(module.trajectory_key)]
            d_gt_raw = denormalize_delta(
                x0_true,
                module._delta_shift,
                module._delta_scale,
            )
            traj_gt = pos0.unsqueeze(1) + torch.cumsum(d_gt_raw, dim=1)
            meta["blend_extra"] = _blend_traj_rollout_from_last_observed(
                pos0, d_raw, traj_gt, obs_mask
            )
        return x, meta

    raise ValueError(f"Unsupported model.name: {mname!r}")


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
    OmegaConf.resolve(cfg)
    _validate_task_shapes(cfg)

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
    evo_every = int(args.evolution_every)
    if args.save_evolution_videos and evo_every <= 0:
        raise ValueError(
            "--evolution-every must be > 0 when --save-evolution-videos is set."
        )
    evo_trace_every = evo_every if args.save_evolution_videos else None
    evo_fps = int(args.evolution_fps)
    if args.save_evolution_videos and evo_fps <= 0:
        raise ValueError(
            "--evolution-fps must be > 0 when --save-evolution-videos is set."
        )
    task = _model_name(cfg)
    with torch.no_grad():
        x, sample_meta = _sample_for_task(
            module,
            cfg,
            device,
            n,
            verbose=bool(args.verbose),
            filling_blend=bool(args.filling_blend_videos),
            guidance_scale_override=args.guidance_scale,
            trace_every=evo_trace_every,
        )

    x_np = x.float().cpu().numpy()
    court_w = float(module.court_width)
    court_h = float(module.court_height)

    if not args.no_npz:
        for i in range(n):
            court_xy = denormalize_court_xy_numpy(x_np[i], court_w, court_h)
            stem = f"trajectory_{i:04d}.npz"
            path = os.path.join(out_dir, stem)
            sl = int(sample_meta["seq_len"])
            save_kw: dict[str, Any] = dict(
                task=str(task),
                trajectory_normalized=x_np[i].astype(np.float32),
                trajectory_court_xy=court_xy.astype(np.float32),
                seq_len=sl,
                num_agents=int(module.num_agents),
                coord_dim=int(module.coord_dim),
                court_width=court_w,
                court_height=court_h,
            )
            if task == "trajectory_completion_ddpm":
                save_kw["observed_len"] = int(sample_meta["observed_len"])
                save_kw["future_len"] = int(sample_meta["future_len"])
            np.savez(path, **save_kw)

    blend_np = None
    if sample_meta.get("blend_extra") is not None:
        blend_np = sample_meta["blend_extra"].float().cpu().numpy()

    if args.save_videos:
        for i in range(n):
            court_xy = denormalize_court_xy_numpy(x_np[i], court_w, court_h)
            frames = create_frames_from_trajectory(court_xy, "basketball")
            video_path = os.path.join(out_dir, f"trajectory_{i:04d}.mp4")
            create_video_from_frames(frames, video_path, fps=10)
            if (
                blend_np is not None
                and args.filling_blend_videos
                and task == "trajectory_filling_ddpm"
            ):
                court_b = denormalize_court_xy_numpy(blend_np[i], court_w, court_h)
                frames_b = create_frames_from_trajectory(court_b, "basketball")
                blend_path = os.path.join(out_dir, f"trajectory_{i:04d}_blend.mp4")
                create_video_from_frames(frames_b, blend_path, fps=10)

    if args.save_evolution_videos:
        traces = sample_meta.get("evolution_traces") or []
        if traces:
            step_labels = [int(s) for s, _ in traces]
            print(f"Evolution checkpoints: {step_labels}")
            for i in range(n):
                for step, x_step in traces:
                    court_xy_step = denormalize_court_xy_numpy(
                        x_step[i].float().cpu().numpy(),
                        court_w,
                        court_h,
                    )
                    step_frames = create_frames_from_trajectory(court_xy_step, "basketball")
                    if not step_frames:
                        continue
                    evo_path = os.path.join(
                        out_dir,
                        f"evolution_{i:04d}_step_{int(step):04d}.mp4",
                    )
                    create_video_from_frames(step_frames, evo_path, fps=evo_fps)

    cfg_dump = os.path.join(out_dir, "config.yaml")
    with open(cfg_dump, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Task: {task}. Saved {n} sample(s) under {out_dir}/")
    if not args.no_npz:
        print(f"Wrote trajectory npz files and {cfg_dump}")


if __name__ == "__main__":
    main()
