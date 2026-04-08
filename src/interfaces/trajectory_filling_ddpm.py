from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.modules.ema import LitEma
from src.modules.losses.ddpm import ddpm_training_loss
from src.modules.models.ddpm import TrajectoryDDPMModel
from src.utils.drawing import (
    create_frames_from_trajectory,
    create_video_from_frames,
    frames_to_tb_video_tensor,
)
from src.utils.trajectory_coords import denormalize_court_xy_numpy, denormalize_delta
from torch.optim import Optimizer


def _blend_traj_rollout_from_last_observed(
    pos0: torch.Tensor,
    d_raw: torch.Tensor,
    traj_gt: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """Build positions where observed steps use GT; masked steps roll out predicted deltas.

    For each batch and agent, scan time in order. At observed ``(t)``, output is
    ``traj_gt[t]``. At masked ``(t)``, output is ``traj_gt[t_prev] + sum(d_raw[t_prev+1:t+1])``
    where ``t_prev`` is the latest timestep ``< t`` with mask 1 (if none, ``pos0`` is
    the anchor and the segment sum starts from index 0).
    """
    B, T, N, _ = d_raw.shape
    cum_d = torch.cumsum(d_raw, dim=1)
    out = torch.empty_like(traj_gt)
    for b in range(B):
        for a in range(N):
            last_obs = -1
            for t in range(T):
                if obs_mask[b, t, a] >= 0.5:
                    out[b, t, a] = traj_gt[b, t, a]
                    last_obs = t
                else:
                    if last_obs < 0:
                        base = pos0[b, a]
                        seg = cum_d[b, t, a]
                    else:
                        base = traj_gt[b, last_obs, a]
                        seg = cum_d[b, t, a] - cum_d[b, last_obs, a]
                    out[b, t, a] = base + seg
    return out


class TrajectoryFillingDDPMInterface(pl.LightningModule):
    """DDPM on normalized deltas [B, S, N, C] with masked coordinate context + mask."""

    def __init__(
        self,
        model: TrajectoryDDPMModel,
        seq_len: int,
        num_agents: int,
        coord_dim: int,
        trajectory_key: str = "trajectory",
        context_key: str = "context",
        mask_key: str = "obs_mask",
        position_0_key: str = "position_0",
        context_fill: Sequence[float] = (0.0, 0.0),
        delta_context_fill: Sequence[float] = (0.0, 0.0),
        context_channels: int = 2,
        delta_shift: Sequence[float] = (0.0, 0.0),
        delta_scale: Sequence[float] = (1.0, 1.0),
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
        log_blend_trajectory_video: bool = True,
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
        self.seq_len = int(seq_len)
        self.num_agents = int(num_agents)
        self.coord_dim = int(coord_dim)
        cc = int(context_channels)
        if cc not in (2, 4):
            raise ValueError(f"context_channels must be 2 or 4, got {cc}")
        self.context_channels = cc
        self.trajectory_key = str(trajectory_key)
        self.context_key = str(context_key)
        self.mask_key = str(mask_key)
        self.position_0_key = str(position_0_key)
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
        self.log_blend_trajectory_video = bool(log_blend_trajectory_video)
        self.p_uncond = float(p_uncond)
        self.guidance_scale = float(guidance_scale)

        cf = list(context_fill)
        if len(cf) != 2:
            raise ValueError(f"context_fill must have length 2, got {cf}")
        if self.context_channels == 2:
            fill_vec = cf
        else:
            dcf = list(delta_context_fill)
            if len(dcf) != 2:
                raise ValueError(
                    f"delta_context_fill must have length 2 when context_channels=4, "
                    f"got {dcf}"
                )
            fill_vec = cf + dcf
        fill = torch.tensor(fill_vec, dtype=torch.float32).view(1, 1, 1, -1)
        self.register_buffer("_context_fill_brc", fill)
        self.register_buffer(
            "_delta_shift",
            torch.tensor(list(delta_shift), dtype=torch.float32),
        )
        self.register_buffer(
            "_delta_scale",
            torch.tensor(list(delta_scale), dtype=torch.float32),
        )

    def _check_batch_shape(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> None:
        if x0.dim() != 4:
            raise ValueError(f"Expected x0 [B,S,N,C], got shape {tuple(x0.shape)}")
        if context.dim() != 4:
            raise ValueError(
                f"Expected context [B,S,N,C], got shape {tuple(context.shape)}"
            )
        if obs_mask.dim() != 3:
            raise ValueError(
                f"Expected obs_mask [B,S,N], got shape {tuple(obs_mask.shape)}"
            )
        b, s, n, c = x0.shape
        b2, s2, n2, c2 = context.shape
        b3, s3, n3 = obs_mask.shape
        if not (b == b2 == b3 and s == s2 == s3 and n == n2 == n3):
            raise ValueError(
                f"Shape mismatch x0 {tuple(x0.shape)} context {tuple(context.shape)} "
                f"mask {tuple(obs_mask.shape)}"
            )
        if s != self.seq_len or n != self.num_agents or c != self.coord_dim:
            raise ValueError(
                f"Expected x0 [*, {self.seq_len}, {self.num_agents}, {self.coord_dim}], "
                f"got [*, {s}, {n}, {c}]"
            )
        if c2 != self.context_channels:
            raise ValueError(
                f"Expected context [*, {self.seq_len}, {self.num_agents}, "
                f"{self.context_channels}], got [*, {s2}, {n2}, {c2}]"
            )

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        x0 = batch[self.trajectory_key]
        context = batch[self.context_key]
        obs_mask = batch[self.mask_key].to(dtype=x0.dtype)
        self._check_batch_shape(x0, context, obs_mask)
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
        mask_in = obs_mask
        if self.training and self.p_uncond > 0.0:
            uncond = torch.rand(b, device=device) < self.p_uncond
            if uncond.any():
                ctx_in = context.clone()
                mask_in = obs_mask.clone()
                k = int(uncond.sum().item())
                fill_e = self._context_fill_brc.to(
                    device=device, dtype=ctx_in.dtype
                ).expand(k, self.seq_len, self.num_agents, self.context_channels)
                ctx_in[uncond] = fill_e
                mask_in[uncond] = 0.0
        pred = self.model(x_t, t, ctx_in, obs_mask=mask_in)
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
            if (self.current_epoch + 1) % self.log_every_n_val_epochs != 0:
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
        context = batch[self.context_key][:vs]
        obs_mask = batch[self.mask_key][:vs].to(dtype=context.dtype)
        pos0 = batch[self.position_0_key][:vs]
        n = context.shape[0]
        dtype = context.dtype
        fill = self._context_fill_brc.to(device=self.device, dtype=dtype).expand(
            n, self.seq_len, self.num_agents, self.context_channels
        )
        null_mask = torch.zeros(
            n, self.seq_len, self.num_agents, device=self.device, dtype=dtype
        )
        deltas_hat = self.model.sample(
            batch_size=n,
            seq_len=self.seq_len,
            num_agents=self.num_agents,
            coord_dim=self.coord_dim,
            device=self.device,
            context=context,
            obs_mask=obs_mask,
            guidance_scale=self.guidance_scale,
            dtype=dtype,
            cfg_null_context=fill,
            cfg_null_mask=null_mask,
        )
        d_raw = denormalize_delta(
            deltas_hat,
            self._delta_shift,
            self._delta_scale,
        )
        traj = pos0.unsqueeze(1) + torch.cumsum(d_raw, dim=1)
        traj_blend: torch.Tensor | None = None
        if self.log_blend_trajectory_video:
            x0_true = batch[self.trajectory_key][:vs]
            d_gt_raw = denormalize_delta(
                x0_true,
                self._delta_shift,
                self._delta_scale,
            )
            traj_gt = pos0.unsqueeze(1) + torch.cumsum(d_gt_raw, dim=1)
            traj_blend = _blend_traj_rollout_from_last_observed(
                pos0, d_raw, traj_gt, obs_mask
            )
        if was_training:
            self.train()
        use_wandb = isinstance(self.logger, WandbLogger)
        use_tb = isinstance(self.logger, TensorBoardLogger)
        if use_wandb:
            os.makedirs("tmp", exist_ok=True)
        log_payload: dict[str, wandb.Video] = {}
        for i in range(traj.shape[0]):
            court_xy_pred = denormalize_court_xy_numpy(
                traj[i].float().cpu().numpy(),
                self.court_width,
                self.court_height,
            )
            frames_pred = create_frames_from_trajectory(court_xy_pred, "basketball")
            frames_blend = None
            if traj_blend is not None:
                court_xy_blend = denormalize_court_xy_numpy(
                    traj_blend[i].float().cpu().numpy(),
                    self.court_width,
                    self.court_height,
                )
                frames_blend = create_frames_from_trajectory(
                    court_xy_blend, "basketball"
                )
            if traj.shape[0] == 1:
                key = "samples/trajectory_filling"
                key_blend = "samples/trajectory_filling_blend"
            else:
                key = f"samples/trajectory_filling_{i}"
                key_blend = f"samples/trajectory_filling_{i}_blend"
            if use_wandb:
                # Stable names so each sampling cycle overwrites prior tmp mp4s.
                stem = f"sample_filling_{i}.mp4"
                video_path = os.path.join("tmp", stem)
                create_video_from_frames(frames_pred, video_path, fps=10)
                log_payload[key] = wandb.Video(
                    video_path,
                    format="mp4",
                )
                if frames_blend is not None:
                    stem_blend = f"sample_filling_blend_{i}.mp4"
                    video_path_blend = os.path.join("tmp", stem_blend)
                    create_video_from_frames(frames_blend, video_path_blend, fps=10)
                    log_payload[key_blend] = wandb.Video(
                        video_path_blend,
                        format="mp4",
                    )
            if use_tb:
                vid = frames_to_tb_video_tensor(frames_pred)
                self.logger.experiment.add_video(
                    key,
                    vid,
                    global_step=int(self.global_step),
                    fps=10,
                )
                if frames_blend is not None:
                    vid_blend = frames_to_tb_video_tensor(frames_blend)
                    self.logger.experiment.add_video(
                        key_blend,
                        vid_blend,
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
