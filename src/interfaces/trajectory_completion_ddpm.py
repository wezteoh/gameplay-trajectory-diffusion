from __future__ import annotations

import os
from typing import Any, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim import Optimizer

import wandb
from src.modules.ema import LitEma
from src.modules.losses.ddpm import ddpm_training_loss
from src.modules.models.ddpm import TrajectoryDDPMModel
from src.utils.drawing import (
    create_frames_from_trajectory,
    create_video_from_frames,
    frames_to_tb_video_tensor,
)
from src.utils.trajectory_coords import denormalize_court_xy_numpy


class TrajectoryCompletionDDPMInterface(pl.LightningModule):
    """DDPM on future segment [B, F, N, C] with past context [B, O, N, C]."""

    def __init__(
        self,
        model: TrajectoryDDPMModel,
        future_len: int,
        observed_len: int,
        full_seq_len: int,
        num_agents: int,
        coord_dim: int,
        trajectory_key: str = "trajectory",
        context_key: str = "context",
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        parameterization: str = "eps",
        loss_type: str = "l2",
        court_width: float = 94.0,
        court_height: float = 50.0,
        val_logging_enabled: bool = True,
        val_num_samples: int = 1,
        log_every_n_val_epochs: int = 1,
        ema_enabled: bool = True,
        ema_decay: float = 0.9999,
        ema_use_num_updates: bool = True,
        p_uncond: float = 0.1,
        guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        if ema_enabled:
            self.ema = LitEma(
                self.model,
                decay=float(ema_decay),
                use_num_upates=bool(ema_use_num_updates),
            )
        else:
            self.ema = None
        self.future_len = int(future_len)
        self.observed_len = int(observed_len)
        self.full_seq_len = int(full_seq_len)
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        self.trajectory_key = str(trajectory_key)
        self.context_key = str(context_key)
        self.learning_rate = float(learning_rate)
        self.betas = tuple(betas)
        self.weight_decay = float(weight_decay)
        self.parameterization = str(parameterization)
        self.loss_type = str(loss_type)
        self.court_width = float(court_width)
        self.court_height = float(court_height)
        self.val_logging_enabled = bool(val_logging_enabled)
        self.val_num_samples = max(0, int(val_num_samples))
        self.log_every_n_val_epochs = int(log_every_n_val_epochs)
        self.p_uncond = float(p_uncond)
        self.guidance_scale = float(guidance_scale)

    def _check_batch_shape(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
    ) -> None:
        if x0.dim() != 4:
            raise ValueError(f"Expected x0 [B,F,N,C], got shape {tuple(x0.shape)}")
        if context.dim() != 4:
            raise ValueError(
                f"Expected context [B,O,N,C], got shape {tuple(context.shape)}"
            )
        b, f, n, c = x0.shape
        b2, o, n2, c2 = context.shape
        if b != b2:
            raise ValueError(f"Batch mismatch x0 {b} vs context {b2}")
        if f != self.future_len or o != self.observed_len:
            raise ValueError(
                f"Expected future_len={self.future_len}, "
                f"observed_len={self.observed_len}, got x0 [*,{f},*,*], "
                f"context [*,{o},*,*]"
            )
        if n != n2 or c != c2 or n != self.num_agents or c != self.coord_dim:
            raise ValueError(
                f"Expected N={self.num_agents}, C={self.coord_dim}, "
                f"got x0 [*,*,{n},{c}], context [*,*,{n2},{c2}]"
            )

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        x0 = batch[self.trajectory_key]
        context = batch[self.context_key]
        self._check_batch_shape(x0, context)
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(
            0,
            int(self.model.schedule.num_timesteps),
            (b,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        x_t = self.model.schedule.q_sample(x0, t, noise)
        ctx_in = context
        if self.training and self.p_uncond > 0.0:
            mask = torch.rand(b, device=device) < self.p_uncond
            if mask.any():
                ctx_in = context.clone()
                ctx_in[mask] = 0.0
        pred = self.model(x_t, t, ctx_in)
        loss = ddpm_training_loss(
            pred,
            x0,
            noise,
            parameterization=self.parameterization,
            loss_type=self.loss_type,
        )
        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        if self.global_step % 200 == 0:
            self.log(
                f"{prefix}/x0_abs_mean",
                x0.detach().abs().mean(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        return loss

    def on_validation_epoch_start(self) -> None:
        if self.ema is not None:
            self.ema.store(list(self.model.parameters()))
            self.ema.copy_to(self.model)

    def on_validation_epoch_end(self) -> None:
        try:
            if not self.val_logging_enabled:
                return
            if not isinstance(self.logger, (WandbLogger, TensorBoardLogger)):
                return
            if self.log_every_n_val_epochs <= 0:
                return
            if self.current_epoch % self.log_every_n_val_epochs != 0:
                return
            trainer = self.trainer
            if trainer is None or not trainer.is_global_zero:
                return
            if self.val_num_samples <= 0:
                return
            self._log_sample_trajectory_videos()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model.parameters())

    @torch.no_grad()
    def _log_sample_trajectory_videos(self) -> None:
        was_training = self.training
        self.eval()
        dm = self.trainer.datamodule
        if dm is None:
            return
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        vs = self.val_num_samples
        past = batch[self.context_key][:vs]
        n = past.shape[0]
        future_hat = self.model.sample(
            batch_size=n,
            seq_len=self.future_len,
            num_agents=self.num_agents,
            coord_dim=self.coord_dim,
            device=self.device,
            context=past,
            guidance_scale=self.guidance_scale,
            dtype=past.dtype,
        )
        anchor = past[:, -1]
        future_abs = future_hat + anchor.unsqueeze(1)
        x = torch.cat([past, future_abs], dim=1)
        if was_training:
            self.train()
        use_wandb = isinstance(self.logger, WandbLogger)
        use_tb = isinstance(self.logger, TensorBoardLogger)
        if use_wandb:
            os.makedirs("tmp", exist_ok=True)
        log_payload: dict[str, wandb.Video] = {}
        for i in range(x.shape[0]):
            court_xy = denormalize_court_xy_numpy(
                x[i].float().cpu().numpy(),
                self.court_width,
                self.court_height,
            )
            frames = create_frames_from_trajectory(court_xy, "basketball")
            if x.shape[0] == 1:
                key = "samples/trajectory_completion"
            else:
                key = f"samples/trajectory_completion_{i}"
            if use_wandb:
                stem = (
                    f"sample_completion_e{self.current_epoch}_"
                    f"s{self.global_step}_{i}.mp4"
                )
                video_path = os.path.join("tmp", stem)
                create_video_from_frames(frames, video_path, fps=10)
                log_payload[key] = wandb.Video(
                    video_path,
                    format="mp4",
                )
            if use_tb:
                vid = frames_to_tb_video_tensor(frames)
                self.logger.experiment.add_video(
                    key,
                    vid,
                    global_step=int(self.global_step),
                    fps=10,
                )
        if use_wandb and log_payload:
            self.logger.experiment.log(log_payload, step=self.global_step)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_closure: Callable[..., Any] | None = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema is not None:
            self.ema(self.model)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return opt
