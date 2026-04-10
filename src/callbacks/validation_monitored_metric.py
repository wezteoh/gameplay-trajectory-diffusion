"""Fail fast when a checkpoint monitor is missing from validation metrics."""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MonitoredMetricValidationCallback(Callback):
    """Fail validation if the checkpoint ``monitor`` key is missing.

    The check runs in ``on_validation_end`` (not ``on_validation_epoch_end``) so it
    sees metrics logged from :meth:`~lightning.pytorch.core.LightningModule.log` in
    the module's ``on_validation_epoch_end``. PyTorch Lightning runs callback
    ``on_validation_epoch_end`` before the module hook, so ``callback_metrics``
    would still be missing keys such as trajectory metrics computed there.
    """

    def __init__(self, monitor: str) -> None:
        super().__init__()
        self.monitor = str(monitor)

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if getattr(trainer, "sanity_checking", False):
            return
        if getattr(trainer, "fast_dev_run", False):
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        if self.monitor in metrics:
            return
        msg = (
            f"Checkpoint monitor {self.monitor!r} is missing from validation metrics. "
            "Check the metric name (short names get a val/ prefix), whether val/mse "
            "or val/vb are logged for your loss settings, and whether "
            "trainer.val_trajectory_metrics is enabled with every_n_val_epochs=1 when "
            "monitoring JADE/ADE-style keys (and horizon_stride vs. sequence length)."
        )
        raise RuntimeError(msg)
