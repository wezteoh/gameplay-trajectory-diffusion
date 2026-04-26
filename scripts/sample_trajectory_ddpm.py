"""Generate masked trajectory-filling DDPM samples from a Lightning checkpoint.

Expects ``model.name: trajectory_filling_ddpm`` in the checkpoint ``config.yaml``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference.trajectory_sample import model_name as _model_name  # noqa: E402
from src.inference.trajectory_sample import (  # noqa: E402
    sample_with_trace,
    sampling_options_from_cfg,
)
from src.interfaces.trajectory_filling_ddpm import (  # noqa: E402
    _blend_traj_rollout_from_last_observed,
)
from src.utils.drawing import (  # noqa: E402
    create_frames_from_trajectory,
    create_video_from_frames,
    render_scene_to_rgb,
)
from src.utils.trajectory_coords import (  # noqa: E402
    denormalize_court_xy_numpy,
    denormalize_delta,
    normalize_court_xy,
    normalize_delta,
)
from train import _build_datamodule, _build_module  # noqa: E402

_DATA_PRESETS = tuple(sorted(p.stem for p in (_REPO_ROOT / "configs" / "data").glob("*.yaml")))


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
        "--save-npz",
        action="store_true",
        help="Write trajectory_*.npz files (off by default).",
    )
    p.add_argument(
        "--save-videos",
        action="store_true",
        help="Also write trajectory_*.gif court visualizations.",
    )
    p.add_argument(
        "--save-evolution-videos",
        action="store_true",
        help=("Also write evolution_*.gif showing denoising checkpoints for each sample."),
    )
    p.add_argument(
        "--evolution-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated step indices to snapshot (required with evolution "
            "videos/images). Meanings: ancestral = DDPM t after p_sample; ddim = "
            "inner step index i; dpm = solver step key. Include training T "
            "(e.g. 1000) for the initial noise frame."
        ),
    )
    p.add_argument(
        "--evolution-fps",
        type=int,
        default=10,
        help="FPS for evolution videos (default: 10).",
    )
    p.add_argument(
        "--save-evolution-images",
        action="store_true",
        help=(
            "Also write evolution_*_step_*.png (full-court faded polylines) per "
            "denoising checkpoint; uses --evolution-steps like videos."
        ),
    )
    p.add_argument(
        "--evolution-image-dpi",
        type=int,
        default=96,
        help=(
            "Matplotlib figure DPI for all court renders from this script: "
            "trajectory GIFs, blend GIFs, evolution GIFs, and evolution PNGs "
            "(default: 96)."
        ),
    )
    p.add_argument(
        "--filling-blend-videos",
        action="store_true",
        help=(
            "Also write trajectory_*_blend.gif (GT at observed steps + predicted "
            "deltas elsewhere), like validation."
        ),
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help=(
            "Override classifier-free guidance scale. "
            "If omitted, uses value from checkpoint config."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step DDPM sampling progress.",
    )
    p.add_argument(
        "--sampling-method",
        type=str,
        default="ancestral",
        choices=("ancestral", "dpm", "ddim"),
        help="Override cfg.sampling.method (ancestral, dpm, or ddim). "
        "Default: from checkpoint config.",
    )
    p.add_argument(
        "--dpm-steps",
        type=int,
        default=None,
        help="Override cfg.sampling.dpm.steps when --sampling-method=dpm.",
    )
    p.add_argument(
        "--dpm-order",
        type=int,
        default=None,
        help="Override cfg.sampling.dpm.order when using DPM-Solver.",
    )
    p.add_argument(
        "--ddim-steps",
        type=int,
        default=None,
        help="Override cfg.sampling.ddim.steps when --sampling-method=ddim.",
    )
    p.add_argument(
        "--ddim-eta",
        type=float,
        default=None,
        help="Override cfg.sampling.ddim.eta (0 = deterministic DDIM).",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Directory of conditioning examples: each immediate subdirectory must "
            "contain traj.npy (court XY (T,A,2) or (1,T,A,2)) and mask.npy "
            "((T,A) or (T,A,1)). Runs sampling per subdir; outputs go under "
            "tmp/<output-subdir>/<subdir_name>/."
        ),
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


def _validate_task_shapes(cfg: DictConfig) -> None:
    """Ensure resolved data.* matches model.backbone.* for trajectory filling."""
    mname = _model_name(cfg)
    if mname != "trajectory_filling_ddpm":
        raise ValueError(
            f"Unsupported model.name for sampling: {mname!r}; " "expected 'trajectory_filling_ddpm'"
        )
    bb = cfg.model.backbone
    d = cfg.data
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
                f"data.{key}={d[key]} does not match model.backbone.{key}=" f"{bb[key]}."
            )


def _merge_sampling_cli(cfg: DictConfig, args: argparse.Namespace) -> None:
    """Apply optional CLI overrides onto ``cfg.sampling``."""
    if args.sampling_method is not None:
        if OmegaConf.select(cfg, "sampling") is None:
            cfg.sampling = OmegaConf.create({})
        cfg.sampling.method = str(args.sampling_method)
    if args.dpm_steps is not None or args.dpm_order is not None:
        if OmegaConf.select(cfg, "sampling") is None:
            cfg.sampling = OmegaConf.create({})
        if OmegaConf.select(cfg, "sampling.dpm") is None:
            cfg.sampling.dpm = OmegaConf.create({})
        if args.dpm_steps is not None:
            cfg.sampling.dpm.steps = int(args.dpm_steps)
        if args.dpm_order is not None:
            cfg.sampling.dpm.order = int(args.dpm_order)
    if args.ddim_steps is not None or args.ddim_eta is not None:
        if OmegaConf.select(cfg, "sampling") is None:
            cfg.sampling = OmegaConf.create({})
        if OmegaConf.select(cfg, "sampling.ddim") is None:
            cfg.sampling.ddim = OmegaConf.create({})
        if args.ddim_steps is not None:
            cfg.sampling.ddim.steps = int(args.ddim_steps)
        if args.ddim_eta is not None:
            cfg.sampling.ddim.eta = float(args.ddim_eta)


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
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
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


def _cond_subdirs(input_dir: Path) -> list[Path]:
    """Sorted immediate child directories of ``input_dir``."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")
    subs = [p for p in input_dir.iterdir() if p.is_dir()]
    return sorted(subs, key=lambda p: p.name)


def _cond_traj_mask_paths(subdir: Path) -> tuple[Path, Path]:
    """Return ``(traj.npy, mask.npy)`` paths under ``subdir``."""
    traj_p = subdir / "traj.npy"
    mask_p = subdir / "mask.npy"
    missing = [name for name, p in (("traj.npy", traj_p), ("mask.npy", mask_p)) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{subdir}: missing required file(s): {', '.join(missing)}")
    return traj_p, mask_p


def _prepare_pos_mask_arrays(
    positions: np.ndarray,
    mask: np.ndarray,
    full_t: int,
    num_agents: int,
    coord_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if pos.ndim == 4 and pos.shape[0] == 1:
        pos = pos[0]
    if pos.ndim != 3 or pos.shape[-1] != coord_dim:
        raise ValueError(
            f"Positions must be (T,A,{coord_dim}) or (1,T,A,{coord_dim}), got {pos.shape}"
        )
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m.squeeze(-1)
    if m.ndim != 2:
        raise ValueError(f"Mask must be (T,A) or (T,A,1), got {m.shape}")
    if pos.shape[:2] != m.shape or pos.shape[0] != full_t or pos.shape[1] != num_agents:
        raise ValueError(
            f"Expected positions {full_t}x{num_agents}x{coord_dim} and matching mask "
            f"(T,A); got pos {pos.shape}, mask {m.shape}"
        )
    return pos, m


def _filling_single_example_from_arrays(
    court_xy_tap2: np.ndarray,
    obs_mask_ta: np.ndarray,
    module: Any,
    cfg: DictConfig,
) -> dict[str, torch.Tensor]:
    """One row like ``NBATrajectoryFillingDataset.__getitem__`` (no batch dim)."""
    dp = cfg.data.params
    full_t = int(cfg.data.full_seq_len)
    num_agents = int(cfg.data.num_agents)
    coord_dim = int(cfg.data.coord_dim)
    pos, m_arr = _prepare_pos_mask_arrays(court_xy_tap2, obs_mask_ta, full_t, num_agents, coord_dim)
    x = torch.from_numpy(pos)
    norm = normalize_court_xy(
        x,
        court_width=float(module.court_width),
        court_height=float(module.court_height),
    )
    raw_delta = torch.zeros(
        full_t,
        num_agents,
        coord_dim,
        dtype=norm.dtype,
        device=norm.device,
    )
    if full_t > 1:
        raw_delta[1:] = norm[1:] - norm[:-1]

    cf = OmegaConf.select(dp, "context_fill", default=[0.0, 0.0])
    fill = torch.tensor(list(cf), dtype=torch.float32).to(dtype=norm.dtype, device=norm.device)
    full_m = torch.from_numpy(m_arr).to(dtype=norm.dtype, device=norm.device)
    m_exp = full_m.unsqueeze(-1)
    ctx_xy = norm * m_exp + fill.view(1, 1, 2) * (1.0 - m_exp)

    shift = module._delta_shift.to(device=norm.device, dtype=raw_delta.dtype)
    scale = module._delta_scale.to(device=norm.device, dtype=raw_delta.dtype)
    target = normalize_delta(raw_delta, shift, scale)

    if bool(OmegaConf.select(dp, "include_delta_in_context", default=False)):
        dcf = OmegaConf.select(dp, "delta_context_fill", default=[0.0, 0.0])
        dfill = torch.tensor(list(dcf), dtype=torch.float32).to(
            dtype=norm.dtype, device=norm.device
        )
        norm_delta = normalize_delta(raw_delta, shift, scale)
        delta_ctx = norm_delta * m_exp + dfill.view(1, 1, 2) * (1.0 - m_exp)
        ctx_full = torch.cat([ctx_xy, delta_ctx], dim=-1)
    else:
        ctx_full = ctx_xy

    cc = int(ctx_full.shape[-1])
    if cc != int(module.context_channels):
        raise ValueError(
            f"Built context with {cc} channels but model.context_channels="
            f"{module.context_channels} (check include_delta_in_context vs checkpoint)."
        )
    tk = str(module.trajectory_key)
    ck = str(module.context_key)
    mk = str(module.mask_key)
    pk = str(module.position_0_key)
    return {
        tk: target,
        ck: ctx_full,
        mk: full_m,
        pk: norm[0],
    }


def _repeat_batch_dim0(
    batch1: dict[str, torch.Tensor],
    n: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Repeat each tensor along batch dim 0 ``n`` times."""
    out: dict[str, torch.Tensor] = {}
    for k, v in batch1.items():
        out[k] = v.unsqueeze(0).expand(n, *v.shape).contiguous().to(device)
    return out


def _sample_for_task(
    module: Any,
    cfg: DictConfig,
    device: torch.device,
    n: int,
    verbose: bool,
    filling_blend: bool = False,
    guidance_scale_override: float | None = None,
    trace_steps: list[int] | None = None,
    batch: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    """Run sampling.

    Returns tensor for visualization (normalized positions) and metadata.
    """
    mname = _model_name(cfg)
    if mname != "trajectory_filling_ddpm":
        raise ValueError(f"Unsupported model.name: {mname!r}; expected 'trajectory_filling_ddpm'")
    meta: dict[str, Any] = {"task": mname}
    sm, dpm_d, ddim_d = sampling_options_from_cfg(cfg)

    if batch is None:
        _ensure_nba_data_root(cfg)
        dm = _build_datamodule(cfg.data)
        batch = _val_batch_first_n(dm, n, device)
    else:
        batch = _repeat_batch_dim0(batch, n, device)

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
    deltas_hat, delta_traces = sample_with_trace(
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
        trace_steps=trace_steps,
        sampler=sm,
        dpm_config=dpm_d,
        ddim_config=ddim_d,
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
        meta["blend_extra"] = _blend_traj_rollout_from_last_observed(pos0, d_raw, traj_gt, obs_mask)
    return x, meta


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


def _parse_evolution_steps(s: str | None) -> list[int]:
    """Parse comma-separated non-negative integers for evolution checkpoints."""
    if s is None or not str(s).strip():
        return []
    out: list[int] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        v = int(p, 10)
        if v < 0:
            raise ValueError(f"evolution step must be >= 0, got {v}")
        out.append(v)
    return out


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
            "Expected a Lightning checkpoint with 'state_dict', " f"got: {checkpoint_path!r}"
        )
    module.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt


def _maybe_apply_ema(module: Any, use_ema: bool) -> None:
    if not use_ema or getattr(module, "ema", None) is None:
        return
    module.ema.copy_to(module.model)


def _load_filling_single_from_subdir(
    subdir: Path,
    module: Any,
    cfg: DictConfig,
) -> dict[str, torch.Tensor]:
    traj_path, mask_path = _cond_traj_mask_paths(subdir)
    pos = np.load(traj_path)
    mask = np.load(mask_path)
    return _filling_single_example_from_arrays(pos, mask, module, cfg)


def _save_sample_results(
    args: argparse.Namespace,
    cfg: DictConfig,
    module: Any,
    task: str,
    out_dir: str,
    n: int,
    x: torch.Tensor,
    sample_meta: dict[str, Any],
) -> None:
    """Write npz, videos, evolution artifacts, blend, and config.yaml under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    court_dpi = int(args.evolution_image_dpi)
    x_np = x.float().cpu().numpy()
    court_w = float(module.court_width)
    court_h = float(module.court_height)

    if args.save_npz:
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
            np.savez(path, **save_kw)

    blend_np = None
    if sample_meta.get("blend_extra") is not None:
        blend_np = sample_meta["blend_extra"].float().cpu().numpy()

    if args.save_videos:
        for i in range(n):
            court_xy = denormalize_court_xy_numpy(x_np[i], court_w, court_h)
            frames = create_frames_from_trajectory(court_xy, "basketball", dpi=court_dpi)
            video_path = os.path.join(out_dir, f"trajectory_{i:04d}.gif")
            create_video_from_frames(frames, video_path, fps=10)
            if blend_np is not None and args.filling_blend_videos:
                court_b = denormalize_court_xy_numpy(blend_np[i], court_w, court_h)
                frames_b = create_frames_from_trajectory(court_b, "basketball", dpi=court_dpi)
                blend_path = os.path.join(out_dir, f"trajectory_{i:04d}_blend.gif")
                create_video_from_frames(frames_b, blend_path, fps=10)

    want_evo_traces = bool(args.save_evolution_videos or args.save_evolution_images)
    if want_evo_traces:
        traces = sample_meta.get("evolution_traces") or []
        if traces:
            step_labels = [int(s) for s, _ in traces]
            print(f"Evolution checkpoints: {step_labels}")
            evo_fps = int(args.evolution_fps)
            n_png = 0
            for i in range(n):
                for step, x_step in traces:
                    court_xy_step = denormalize_court_xy_numpy(
                        x_step[i].float().cpu().numpy(),
                        court_w,
                        court_h,
                    )
                    if args.save_evolution_videos:
                        step_frames = create_frames_from_trajectory(
                            court_xy_step,
                            "basketball",
                            basketball_ball_trace=False,
                            dpi=court_dpi,
                        )
                        if step_frames:
                            evo_path = os.path.join(
                                out_dir,
                                f"evolution_{i:04d}_step_{int(step):04d}.gif",
                            )
                            create_video_from_frames(step_frames, evo_path, fps=evo_fps)
                    if args.save_evolution_images:
                        rgb = render_scene_to_rgb(court_xy_step, dpi=court_dpi)
                        png_path = os.path.join(
                            out_dir,
                            f"evolution_{i:04d}_step_{int(step):04d}.png",
                        )
                        iio.imwrite(png_path, rgb)
                        n_png += 1
            if args.save_evolution_images and n_png:
                print(f"Wrote {n_png} evolution PNG(s) (fadeline court views).")

    cfg_dump = os.path.join(out_dir, "config.yaml")
    with open(cfg_dump, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


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
    _merge_sampling_cli(cfg, args)
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
    base_out = os.path.join(str(_REPO_ROOT), "tmp", out_sub)

    n = int(args.num_samples)
    want_evo_traces = bool(args.save_evolution_videos or args.save_evolution_images)
    evo_step_list: list[int] | None = None
    if want_evo_traces:
        evo_step_list = _parse_evolution_steps(args.evolution_steps)
        if not evo_step_list:
            raise ValueError(
                "Non-empty --evolution-steps is required when "
                "--save-evolution-videos or --save-evolution-images is set "
                "(comma-separated integers, e.g. 1000,750,500,250,0)."
            )
    evo_fps = int(args.evolution_fps)
    if args.save_evolution_videos and evo_fps <= 0:
        raise ValueError("--evolution-fps must be > 0 when --save-evolution-videos is set.")
    court_dpi = int(args.evolution_image_dpi)
    if court_dpi < 1 and (
        args.save_videos or args.save_evolution_videos or args.save_evolution_images
    ):
        raise ValueError(
            "--evolution-image-dpi must be >= 1 when saving videos or evolution images."
        )
    task = _model_name(cfg)

    input_dir_arg = args.input_dir
    if input_dir_arg:
        input_dir = Path(os.path.abspath(os.path.expanduser(str(input_dir_arg))))
        subdirs = _cond_subdirs(input_dir)
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found under --input-dir {input_dir}")
        with torch.no_grad():
            for sd in subdirs:
                batch_single = _load_filling_single_from_subdir(sd, module, cfg)
                x, sample_meta = _sample_for_task(
                    module,
                    cfg,
                    device,
                    n,
                    verbose=bool(args.verbose),
                    filling_blend=bool(args.filling_blend_videos),
                    guidance_scale_override=args.guidance_scale,
                    trace_steps=evo_step_list,
                    batch=batch_single,
                )
                out_dir = os.path.join(base_out, sd.name)
                _save_sample_results(
                    args,
                    cfg,
                    module,
                    task,
                    out_dir,
                    n,
                    x,
                    sample_meta,
                )
                print(f"Task: {task}. Saved {n} sample(s) under {out_dir}/")
        print(
            f"Directory mode: processed {len(subdirs)} conditioning folder(s) under " f"{base_out}/"
        )
        if args.save_npz:
            print("Wrote trajectory npz files and config.yaml per folder.")
        else:
            print("Wrote config.yaml per folder.")
    else:
        out_dir = base_out
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():
            x, sample_meta = _sample_for_task(
                module,
                cfg,
                device,
                n,
                verbose=bool(args.verbose),
                filling_blend=bool(args.filling_blend_videos),
                guidance_scale_override=args.guidance_scale,
                trace_steps=evo_step_list,
                batch=None,
            )
        _save_sample_results(args, cfg, module, task, out_dir, n, x, sample_meta)
        cfg_dump = os.path.join(out_dir, "config.yaml")
        print(f"Task: {task}. Saved {n} sample(s) under {out_dir}/")
        if args.save_npz:
            print(f"Wrote trajectory npz files and {cfg_dump}")
        else:
            print(f"Wrote {cfg_dump}")


if __name__ == "__main__":
    main()
