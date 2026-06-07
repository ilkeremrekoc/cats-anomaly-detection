from __future__ import annotations

import pytorch_lightning as pl


class MetricHistoryCallback(pl.Callback):
    """Collect train and validation loss values after each epoch."""

    def __init__(self) -> None:
        """Initialize lists for loss history."""
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Store the latest training loss at the end of each epoch."""
        metric = trainer.callback_metrics.get("train_loss")
        if metric is not None:
            # Move tensor to CPU and convert to plain float for plotting.
            self.train_losses.append(float(metric.detach().cpu().item()))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Store the latest validation loss at the end of each epoch."""
        metric = trainer.callback_metrics.get("val_loss")
        if metric is not None:
            # Move tensor to CPU and convert to plain float for plotting.
            self.val_losses.append(float(metric.detach().cpu().item()))
