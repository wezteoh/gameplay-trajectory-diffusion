import os
from typing import Any, Literal

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from hydra.utils import instantiate

from src.data.nba_trajectory_filling import NBATrajectoryFillingDataModule
from src.interfaces.trajectory_filling_ddpm import TrajectoryFillingDDPMInterface
from src.modules.diffusion.schedule import DDPMNoiseSchedule
from src.modules.models.ddpm import TrajectoryDDPMModel


def _build_datamodule(data_cfg: DictConfig) -> Any:
    params = OmegaConf.to_container(data_cfg.params, resolve=True)
    if data_cfg.name == "trajectory_nba_filling":
        return NBATrajectoryFillingDataModule(**params)
    raise ValueError(f"Unsupported dataset: {data_cfg.name}")


def _sampling_kwargs(cfg: DictConfig) -> tuple[str, dict[str, Any]]:
    """Read ``cfg.sampling`` for validation / checkpoint sampling (ancestral vs DPM-Solver)."""
    s = OmegaConf.select(cfg, "sampling", default=None)
    if s is None:
        return "ancestral", {}
    method = str(OmegaConf.select(s, "method", default="ancestral"))
    raw = OmegaConf.select(s, "dpm", default={})
    dpm = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    if not isinstance(dpm, dict):
        dpm = {}
    return method, dpm


def _build_module(cfg: DictConfig) -> pl.LightningModule:
    model_cfg = cfg.model
    model_name = str(
        OmegaConf.select(model_cfg, "name", default="trajectory_filling_ddpm")
    )
    if model_name != "trajectory_filling_ddpm":
        raise ValueError(
            f"Unsupported model name: {model_name!r}; expected 'trajectory_filling_ddpm'"
        )

    data_cfg = cfg.data
    optim = OmegaConf.to_container(cfg.trainer.optim, resolve=True)
    traj_key = str(
        OmegaConf.select(
            model_cfg,
            "trajectory_key",
            default=OmegaConf.select(
                data_cfg, "params.trajectory_key", default="trajectory"
            ),
        )
    )

    include_delta_ctx = bool(
        OmegaConf.select(
            data_cfg, "params.include_delta_in_context", default=False
        )
    )
    context_channels = 4 if include_delta_ctx else int(data_cfg.coord_dim)
    backbone = instantiate(
        model_cfg.backbone,
        _recursive_=False,
        context_channels=context_channels,
    )
    schedule = DDPMNoiseSchedule(
        timesteps=int(model_cfg.timesteps),
        beta_schedule=str(model_cfg.beta_schedule),
        linear_start=float(model_cfg.linear_start),
        linear_end=float(model_cfg.linear_end),
        cosine_s=float(model_cfg.cosine_s),
    )
    ddpm_model = TrajectoryDDPMModel(
        backbone=backbone,
        schedule=schedule,
        parameterization=str(model_cfg.parameterization),
    )
    court_w = float(OmegaConf.select(data_cfg, "court_width", default=94.0))
    court_h = float(OmegaConf.select(data_cfg, "court_height", default=50.0))
    sampling_method, sampling_dpm = _sampling_kwargs(cfg)
    vl = OmegaConf.select(cfg, "trainer.val_logging", default=None)
    if vl is None:
        val_logging_enabled = bool(cfg.wandb.enabled)
        val_num_samples = 1
        log_every_n_val_epochs = int(
            OmegaConf.select(
                cfg,
                "wandb.log_sample_video_every_n_epochs",
                default=1,
            )
        )
    else:
        val_logging_enabled = bool(vl.enabled)
        val_num_samples = int(vl.num_samples)
        log_every_n_val_epochs = int(vl.log_every_n_val_epochs)

    ema_cfg = OmegaConf.select(cfg, "trainer.ema", default=None)
    if ema_cfg is None:
        ema_enabled = True
        ema_decay = 0.9999
        ema_use_num_updates = True
    else:
        ema_enabled = bool(OmegaConf.select(ema_cfg, "enabled", default=True))
        ema_decay = float(OmegaConf.select(ema_cfg, "decay", default=0.9999))
        ema_use_num_updates = bool(
            OmegaConf.select(ema_cfg, "use_num_updates", default=True)
        )

    ctx_key = str(
        OmegaConf.select(
            model_cfg,
            "context_key",
            default=OmegaConf.select(
                data_cfg, "params.context_key", default="context"
            ),
        )
    )
    mask_key = str(
        OmegaConf.select(
            model_cfg,
            "mask_key",
            default=OmegaConf.select(
                data_cfg, "params.mask_key", default="obs_mask"
            ),
        )
    )
    pos0_key = str(
        OmegaConf.select(
            model_cfg,
            "position_0_key",
            default=OmegaConf.select(
                data_cfg, "params.position_0_key", default="position_0"
            ),
        )
    )
    p_uncond = float(OmegaConf.select(model_cfg, "p_uncond", default=0.1))
    guidance_scale = float(
        OmegaConf.select(model_cfg, "guidance_scale", default=1.0)
    )
    log_blend_trajectory_video = bool(
        OmegaConf.select(model_cfg, "log_blend_trajectory_video", default=True)
    )
    full_t = int(data_cfg.full_seq_len)
    delta_len = int(OmegaConf.select(data_cfg, "delta_len", default=full_t))
    if delta_len != full_t:
        raise ValueError(
            f"data.delta_len ({delta_len}) must equal full_seq_len ({full_t})"
        )
    fill = OmegaConf.select(data_cfg, "params.context_fill", default=[0.0, 0.0])
    d_delta_fill = OmegaConf.select(
        data_cfg, "params.delta_context_fill", default=[0.0, 0.0]
    )
    d_shift = OmegaConf.select(data_cfg, "params.delta_shift", default=[0.0, 0.0])
    d_scale = OmegaConf.select(data_cfg, "params.delta_scale", default=[1.0, 1.0])
    return TrajectoryFillingDDPMInterface(
        model=ddpm_model,
        seq_len=delta_len,
        num_agents=int(data_cfg.num_agents),
        coord_dim=int(data_cfg.coord_dim),
        trajectory_key=traj_key,
        context_key=ctx_key,
        mask_key=mask_key,
        position_0_key=pos0_key,
        context_fill=list(fill),
        delta_context_fill=list(d_delta_fill),
        context_channels=context_channels,
        delta_shift=list(d_shift),
        delta_scale=list(d_scale),
        learning_rate=float(optim["learning_rate"]),
        betas=tuple(optim.get("betas", (0.9, 0.999))),
        weight_decay=float(optim.get("weight_decay", 0.0)),
        parameterization=str(model_cfg.parameterization),
        loss_type=str(model_cfg.loss_type),
        court_width=court_w,
        court_height=court_h,
        val_logging_enabled=val_logging_enabled,
        val_num_samples=val_num_samples,
        log_every_n_val_epochs=log_every_n_val_epochs,
        ema_enabled=ema_enabled,
        ema_decay=ema_decay,
        ema_use_num_updates=ema_use_num_updates,
        p_uncond=p_uncond,
        guidance_scale=guidance_scale,
        log_blend_trajectory_video=log_blend_trajectory_video,
        sampling_method=sampling_method,
        sampling_dpm=sampling_dpm,
    )


def _apply_hardware_options(
    cfg: DictConfig,
    trainer_kwargs: dict[str, Any],
) -> None:
    use_gpu = bool(cfg.hardware.use_gpu)
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "hardware.use_gpu=true but CUDA is not available. "
                "Set hardware.use_gpu=false to train on CPU."
            )
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = int(cfg.hardware.gpu_devices)
        return

    trainer_kwargs["accelerator"] = "cpu"
    trainer_kwargs["devices"] = 1


def _module_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _print_model_tree(
    module: nn.Module,
    max_depth: int = 4,
    prefix: str = "",
    depth: int = 0,
    name: str | None = None,
) -> None:
    param_count = _module_param_count(module)
    display_name = name or module.__class__.__name__
    if depth == 0:
        print(f"\n===== Model (top {max_depth} levels) =====")
    cls_name = module.__class__.__name__
    print(f"{prefix}{display_name} ({cls_name}): {param_count:,} params")
    if depth < max_depth - 1:
        for child_name, child in module.named_children():
            _print_model_tree(
                child,
                max_depth,
                prefix + "  ",
                depth + 1,
                child_name,
            )


def _print_model_summary(module: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in module.parameters())
    total_param_size_mb = (total_params * 4) / (1024**2)

    _print_model_tree(module, max_depth=4)
    print("===== Parameter Summary =====")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Approx parameter size (fp32): {total_param_size_mb:.2f} MB\n")


def _save_checkpoint_dir_config(cfg: DictConfig, ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


def _wandb_project_checkpoint_segment(project: str) -> str:
    s = str(project).strip()
    if not s:
        return "unnamed_project"
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    for c in '<>:"|?*':
        s = s.replace(c, "_")
    return s or "unnamed_project"


def _resolve_logging_backend(
    cfg: DictConfig,
) -> Literal["wandb", "tensorboard", "none"]:
    backend = OmegaConf.select(cfg, "logging.backend", default=None)
    if backend is None:
        return "wandb" if bool(cfg.wandb.enabled) else "none"
    b = str(backend).strip().lower()
    if b not in ("wandb", "tensorboard", "none"):
        raise ValueError(
            f"Unknown logging.backend {backend!r}; "
            "expected 'wandb', 'tensorboard', or 'none'"
        )
    return b  # type: ignore[return-value]


@hydra.main(
    config_path="configs",
    config_name="train_trajectory_filling_ddpm",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = _build_datamodule(cfg.data)
    module = _build_module(cfg)

    trainer_kwargs = OmegaConf.to_container(
        cfg.trainer.lightning,
        resolve=True,
    )
    _apply_hardware_options(cfg, trainer_kwargs)

    backend = _resolve_logging_backend(cfg)
    project_seg = _wandb_project_checkpoint_segment(str(cfg.wandb.project))
    logger: TensorBoardLogger | WandbLogger | bool = False
    checkpoint_relpath = "no_wandb"

    if backend == "wandb":
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            save_dir=cfg.wandb.save_dir,
            log_model=cfg.wandb.log_model,
        )
        run_id = logger.experiment.id
        checkpoint_relpath = os.path.join(project_seg, run_id)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    elif backend == "tensorboard":
        tb_save_dir = str(
            OmegaConf.select(
                cfg,
                "logging.tensorboard.save_dir",
                default="./tensorboard_logs",
            )
        )
        tb_name = OmegaConf.select(cfg, "logging.tensorboard.name", default=None)
        tb_kwargs: dict[str, Any] = {"save_dir": tb_save_dir}
        if tb_name is not None and str(tb_name).strip():
            tb_kwargs["name"] = str(tb_name)
        logger = TensorBoardLogger(**tb_kwargs)
        checkpoint_relpath = os.path.join(project_seg, f"tb_v{logger.version}")

    monitor, mode = "val/loss", "min"
    filename = "traj-ddpm-{epoch}-{step}"
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join("checkpoints", checkpoint_relpath),
            monitor=monitor,
            mode=mode,
            save_top_k=2,
            save_last=True,
            filename=filename,
            auto_insert_metric_name=True,
        )
    ]
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    ckpt_dir = os.path.join("checkpoints", checkpoint_relpath)
    _save_checkpoint_dir_config(cfg, ckpt_dir)

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)
    _print_model_summary(module)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
