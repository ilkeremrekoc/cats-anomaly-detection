from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from mlflow import log_metric, set_experiment, set_tracking_uri, start_run
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cats_anomaly_detection.data import TimeSeriesDataModule
from cats_anomaly_detection.training.lightning_module import AnomalyLightningModule
from cats_anomaly_detection.utils.plots import (
    collect_reconstruction_errors,
    plot_anomaly_score_timeline,
    plot_reconstruction_histogram,
)
from cats_anomaly_detection.utils.reproducibility import seed_everything


def is_mlflow_reachable(tracking_uri: str) -> bool:
    """Return True when the MLflow endpoint is reachable over HTTP."""
    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.scheme not in {"http", "https"}:
        return False

    try:
        response = requests.get(tracking_uri, timeout=2.0)
        return response.status_code < 500
    except requests.RequestException:
        return False


def run_inference(cfg: DictConfig) -> None:
    """Run inference, compute anomaly metrics, and save outputs/plots."""

    # Keep run deterministic for reproducible inference results.
    seed_everything(int(cfg.seed))

    data_module = TimeSeriesDataModule(cfg)
    # Load and prepare test data.
    data_module.prepare_data()
    data_module.setup(stage="test")

    # Build model and switch to eval mode.
    model = AnomalyLightningModule(cfg)
    model.eval()

    # Load latest checkpoint if it exists.
    checkpoint_dir = Path("models")
    checkpoint_files = sorted(checkpoint_dir.glob("*.ckpt"))
    if checkpoint_files:
        checkpoint_path = str(checkpoint_files[-1])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

    test_loader = data_module.test_dataloader()
    # Reconstruction error is the anomaly score.
    reconstruction_errors = collect_reconstruction_errors(model=model, dataloader=test_loader)

    # Mark high-error windows as anomalies using a fixed percentile threshold.
    percentile_threshold = np.percentile(reconstruction_errors, 95.0)
    anomaly_flags = (reconstruction_errors > percentile_threshold).astype(int)

    # Save raw inference outputs for later analysis.
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "test_reconstruction_errors.npy", reconstruction_errors)
    np.save(output_dir / "test_anomaly_flags.npy", anomaly_flags)

    metrics: dict[str, float] = {
        "mean_reconstruction_error": float(np.mean(reconstruction_errors)),
    }

    test_labels = data_module.test_window_labels()
    if test_labels is not None:
        metrics["precision"] = float(precision_score(test_labels, anomaly_flags, zero_division=0))
        metrics["recall"] = float(recall_score(test_labels, anomaly_flags, zero_division=0))
        metrics["f1_score"] = float(f1_score(test_labels, anomaly_flags, zero_division=0))

        # These are score-based metrics and use raw reconstruction error scores.
        unique_labels = np.unique(test_labels)
        if len(unique_labels) > 1:
            metrics["pr_auc"] = float(average_precision_score(test_labels, reconstruction_errors))
            metrics["roc_auc"] = float(roc_auc_score(test_labels, reconstruction_errors))

    # Save human-readable metric summary.
    metric_lines = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    (output_dir / "metrics.txt").write_text("\n".join(metric_lines), encoding="utf-8")

    tracking_uri = str(cfg.logging.tracking_uri)
    # Log inference metrics to MLflow only when server is reachable.
    if is_mlflow_reachable(tracking_uri):
        set_tracking_uri(tracking_uri)
        set_experiment(str(cfg.logging.experiment_name))
        with start_run(run_name=f"{str(cfg.logging.run_name)}-inference"):
            for key, value in metrics.items():
                log_metric(f"inference_{key}", float(value))

    # Save evaluation plots to disk.
    plot_dir = Path(str(cfg.logging.plot_dir))
    plot_reconstruction_histogram(reconstruction_errors, plot_dir)
    plot_anomaly_score_timeline(reconstruction_errors, plot_dir)

    print(f"Inference complete. Saved outputs to: {output_dir}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
