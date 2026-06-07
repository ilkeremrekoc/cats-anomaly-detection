from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from cats_anomaly_detection.models import TCNTransformerAutoencoder


class AnomalyLightningModule(pl.LightningModule):
    """Lightning module for anomaly detection with reconstruction.

    The model learns to rebuild normal windows. We use MSE loss for training,
    and we also use reconstruction error as the anomaly score.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Set up the model, config values, and MSE loss."""
        super().__init__()
        self.cfg = cfg
        # Persist config fields into checkpoints and loggers for reproducibility.
        self.save_hyperparameters(logger=True)

        self.model = TCNTransformerAutoencoder(
            input_dim=int(cfg.model.input_dim),
            hidden_dim=int(cfg.model.hidden_dim),
            transformer_heads=int(cfg.model.transformer_heads),
            transformer_layers=int(cfg.model.transformer_layers),
            tcn_levels=int(cfg.model.tcn_levels),
            tcn_kernel_size=int(cfg.model.tcn_kernel_size),
            dropout=float(cfg.model.dropout),
        )
        self.loss_fn = nn.MSELoss()
        self.val_reconstruction_errors: list[np.ndarray] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return reconstructed windows for the input batch."""
        return self.model(inputs)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared step for train, val, and test with loss logging."""
        features, targets = batch
        reconstructions = self(features)
        loss = self.loss_fn(reconstructions, targets)
        self.log(stage, loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute and log training loss."""
        return self._shared_step(batch, stage="train_loss")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute validation loss and accumulate reconstruction errors for epoch-level metrics."""
        features, targets = batch
        reconstructions = self(features)
        loss = self.loss_fn(reconstructions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Accumulate reconstruction errors for epoch-end metric computation.
        errors = torch.mean(
            (reconstructions - targets) ** 2, dim=(1, 2)
        )  # Mean over time and features
        self.val_reconstruction_errors.append(errors.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of each epoch."""
        if not self.val_reconstruction_errors:
            return

        # Concatenate all batch errors into a single array.
        all_errors = np.concatenate(self.val_reconstruction_errors, axis=0)
        self.val_reconstruction_errors.clear()

        # Get validation labels from the datamodule.
        if self.trainer and self.trainer.datamodule:
            val_labels = self.trainer.datamodule.val_window_labels()

            # Only compute metrics if we have matching number of samples and multiple classes.
            if (
                val_labels is not None
                and len(val_labels) == len(all_errors)
                and len(np.unique(val_labels)) > 1
            ):
                # Threshold at 95th percentile for binary predictions.
                threshold = float(np.percentile(all_errors, 95.0))
                predictions = (all_errors > threshold).astype(np.int32)

                # Compute and log metrics.
                val_precision = float(precision_score(val_labels, predictions, zero_division=0))
                val_recall = float(recall_score(val_labels, predictions, zero_division=0))
                val_f1 = float(f1_score(val_labels, predictions, zero_division=0))
                val_pr_auc = float(average_precision_score(val_labels, all_errors))
                val_roc_auc = float(roc_auc_score(val_labels, all_errors))

                self.log("val_precision", val_precision, on_epoch=True, logger=True)
                self.log("val_recall", val_recall, on_epoch=True, logger=True)
                self.log("val_f1_score", val_f1, on_epoch=True, logger=True)
                self.log("val_pr_auc", val_pr_auc, on_epoch=True, logger=True)
                self.log("val_roc_auc", val_roc_auc, on_epoch=True, logger=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log test loss."""
        return self._shared_step(batch, stage="test_loss")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create an AdamW optimizer from training config values."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.training.learning_rate),
            weight_decay=float(self.cfg.training.weight_decay),
        )
