from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

"""Plotting utilities for training visualization and model evaluation."""


def _save_figure(plot_path: Path) -> None:
    """Helper to save matplotlib figure, create parent dirs if needed, and close."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def plot_loss_curves(train_losses: list[float], val_losses: list[float], plot_dir: Path) -> None:
    """Plot training and validation loss curves across epochs."""
    epochs = np.arange(1, max(len(train_losses), len(val_losses)) + 1)
    plt.figure(figsize=(9, 5))
    if train_losses:
        plt.plot(epochs[: len(train_losses)], train_losses, label="train_loss")
    if val_losses:
        plt.plot(epochs[: len(val_losses)], val_losses, label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    _save_figure(plot_dir / "loss_curve.png")


def collect_reconstruction_errors(model: torch.nn.Module, dataloader: DataLoader) -> np.ndarray:
    """Compute per-window reconstruction error (MSE) on the dataloader.

    Returns array of shape (num_windows,) with reconstruction error for each window.
    """
    model.eval()
    errors: list[np.ndarray] = []

    with torch.no_grad():
        for features, _ in dataloader:
            reconstructed = model(features)
            # Mean squared error across time and feature dimensions, per window.
            batch_error = torch.mean((reconstructed - features) ** 2, dim=(1, 2))
            errors.append(batch_error.detach().cpu().numpy())

    return np.concatenate(errors)


def plot_reconstruction_histogram(errors: np.ndarray, plot_dir: Path) -> None:
    """Plot histogram of reconstruction errors to visualize error distribution."""
    plt.figure(figsize=(9, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title("Reconstruction Error Histogram")
    plt.xlabel("Error")
    plt.ylabel("Count")
    _save_figure(plot_dir / "reconstruction_error_histogram.png")


def plot_anomaly_score_timeline(errors: np.ndarray, plot_dir: Path) -> None:
    """Plot reconstruction errors over time to visualize anomaly scores."""
    plt.figure(figsize=(10, 4))
    plt.plot(errors)
    plt.title("Anomaly Score Timeline (Reconstruction Error)")
    plt.xlabel("Window Index")
    plt.ylabel("Anomaly Score")
    _save_figure(plot_dir / "anomaly_score_timeline.png")


def plot_f1_threshold_curve(errors: np.ndarray, labels: np.ndarray, plot_dir: Path) -> None:
    """Plot F1-score at different reconstruction error threshold percentiles.

    Shows how F1 varies when anomalies are defined as the top 1%, 5%, 10%, and 15%
    of windows by reconstruction error.
    """
    percentiles = [85, 90, 95, 99]
    f1_scores = []

    for percentile in percentiles:
        # Threshold at the given percentile of reconstruction errors.
        threshold = float(np.percentile(errors, percentile))
        # Predict anomaly if reconstruction error exceeds threshold.
        predictions = (errors > threshold).astype(np.int32)
        f1 = float(f1_score(labels, predictions, zero_division=0))
        f1_scores.append(f1)

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, f1_scores, marker="o", linewidth=2, markersize=8, color="steelblue")
    plt.title("F1-Score at Different Thresholds")
    plt.xlabel("Threshold Percentile")
    plt.ylabel("F1-Score")
    plt.xticks(percentiles)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    _save_figure(plot_dir / "f1_threshold_curve.png")
