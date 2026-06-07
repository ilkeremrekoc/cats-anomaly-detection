from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytorch_lightning as pl
import requests
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from cats_anomaly_detection.data import TimeSeriesDataModule
from cats_anomaly_detection.training.callbacks import MetricHistoryCallback
from cats_anomaly_detection.training.lightning_module import AnomalyLightningModule
from cats_anomaly_detection.utils.git_utils import get_git_commit_hash
from cats_anomaly_detection.utils.plots import (
    collect_reconstruction_errors,
    plot_anomaly_score_timeline,
    plot_f1_threshold_curve,
    plot_loss_curves,
    plot_reconstruction_histogram,
)
from cats_anomaly_detection.utils.reproducibility import seed_everything

"""Training orchestration: setup, execution, logging, and visualization."""


def is_mlflow_reachable(tracking_uri: str) -> bool:
    """Check if MLflow server can be reached."""
    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.scheme not in {"http", "https"}:
        return False

    try:
        # Test connectivity and response status.
        response = requests.get(tracking_uri, timeout=2.0)  # Small timeout to avoid long waits.
        return response.status_code < 500
    except requests.RequestException:
        # Network error or other request issue.
        return False


def create_logger(cfg: DictConfig) -> MLFlowLogger | CSVLogger:
    """Create MLflow logger when possible, otherwise use CSV logger."""
    tracking_uri = str(cfg.logging.tracking_uri)
    if is_mlflow_reachable(tracking_uri):
        mlflow_logger = MLFlowLogger(
            experiment_name=str(cfg.logging.experiment_name),
            tracking_uri=tracking_uri,
            run_name=str(cfg.logging.run_name),
            save_dir=str(cfg.logging.save_dir),
            log_model=False,
        )
        try:
            # Force an early connection so trainer setup does not fail later.
            _ = mlflow_logger.experiment
            return mlflow_logger
        except Exception as exc:
            print(f"MLflow logger could not connect ({exc}). Falling back to CSVLogger.")

    print("MLflow server is not reachable, falling back to CSVLogger.")
    return CSVLogger(save_dir="logs", name="fallback")


def run_training(cfg: DictConfig) -> None:
    """Run training, test, and save plots."""
    # Set random seeds for reproducibility.
    seed_everything(int(cfg.seed))

    # Initialize data module and model.
    data_module = TimeSeriesDataModule(cfg)
    model = AnomalyLightningModule(cfg)

    # Determine logging backend: MLflow if reachable, else fallback to CSV.
    logger = create_logger(cfg)
    use_mlflow = isinstance(logger, MLFlowLogger)

    # Setup callbacks: track metrics and save best checkpoint.
    metric_history = MetricHistoryCallback()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        dirpath="models",
    )

    # Configure and run trainer.
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=str(cfg.training.accelerator),
        devices=int(cfg.training.devices),
        precision=int(cfg.training.precision),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        logger=logger,
        callbacks=[checkpoint_callback, metric_history],
        log_every_n_steps=10,
    )

    # Train and test the model.
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    # Log hyperparameters and git commit to MLflow.
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if use_mlflow and isinstance(logger, MLFlowLogger):
        logger.log_hyperparams(resolved_config)
        logger.experiment.log_param(logger.run_id, "git_commit", get_git_commit_hash())

    # Generate plots and save to disk.
    plot_dir = Path(str(cfg.logging.plot_dir))
    plot_loss_curves(metric_history.train_losses, metric_history.val_losses, plot_dir)

    # Compute reconstruction errors and create evaluation plots.
    validation_loader = data_module.val_dataloader()
    reconstruction_errors = collect_reconstruction_errors(model=model, dataloader=validation_loader)
    plot_reconstruction_histogram(reconstruction_errors, plot_dir)
    plot_anomaly_score_timeline(reconstruction_errors, plot_dir)

    # Plot F1-score curve.
    val_labels = data_module.val_window_labels()
    if val_labels is not None and len(np.unique(val_labels)) > 1:
        plot_f1_threshold_curve(reconstruction_errors, val_labels, plot_dir)

    print(f"Training complete. Plots saved to: {plot_dir}")
