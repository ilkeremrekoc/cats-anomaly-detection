from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from cats_anomaly_detection.data.download import ensure_data_available
from cats_anomaly_detection.data.preprocessing import (
    extract_features_and_labels,
    fit_transform_features,
    load_dataframe,
    make_sliding_windows,
    make_window_labels,
    split_windows,
)


@dataclass
class WindowSplits:
    """Container for train/validation/test window arrays."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class LabelSplits:
    """Container for train/validation/test window labels."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class TimeSeriesDataModule(pl.LightningDataModule):
    """Lightning data module for windowed multivariate anomaly detection."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize configuration for data loading and splitting."""

        super().__init__()
        self.cfg = cfg
        self.window_splits: WindowSplits | None = None
        self.label_splits: LabelSplits | None = None
        self.batch_size: int = int(cfg.data.batch_size)
        self.num_workers: int = int(cfg.data.num_workers)

    def prepare_data(self) -> None:
        """Ensure the raw dataset is available locally before setup."""

        ensure_data_available(
            raw_data_path=str(self.cfg.data.raw_data_path),
            download_url=str(self.cfg.data.download_url),
        )

    def setup(self, stage: str | None = None) -> None:
        """Load, preprocess, window, and split features."""

        raw_data_path = Path(str(self.cfg.data.raw_data_path))
        raw_dataframe = load_dataframe(
            raw_data_path=raw_data_path,
        )
        # Labels may be unavailable for some datasets; downstream code handles None.
        feature_frame, labels = extract_features_and_labels(raw_dataframe)
        # Fit scaling before windowing.
        scaled_matrix, _ = fit_transform_features(feature_frame)

        # Convert the sequence into windows for sequence modeling.
        windows = make_sliding_windows(
            data=scaled_matrix,
            window_size=int(self.cfg.data.window_size),
            stride=int(self.cfg.data.stride),
        )

        train_windows, val_windows, test_windows = split_windows(
            windows=windows,
            train_ratio=float(self.cfg.data.train_ratio),
            val_ratio=float(self.cfg.data.val_ratio),
        )
        self.window_splits = WindowSplits(train=train_windows, val=val_windows, test=test_windows)

        if labels is not None:
            # Build window-level labels using the same window and stride settings.
            window_labels = make_window_labels(
                labels=labels,
                window_size=int(self.cfg.data.window_size),
                stride=int(self.cfg.data.stride),
            )
            # Keep label splits aligned with feature-window splits by using identical ratios.
            train_labels, val_labels, test_labels = split_windows(
                windows=window_labels,
                train_ratio=float(self.cfg.data.train_ratio),
                val_ratio=float(self.cfg.data.val_ratio),
            )
            self.label_splits = LabelSplits(
                train=train_labels.astype(np.int32),
                val=val_labels.astype(np.int32),
                test=test_labels.astype(np.int32),
            )
        else:
            self.label_splits = None

    @staticmethod
    def _to_loader(
        windows: np.ndarray, batch_size: int, num_workers: int, shuffle: bool
    ) -> DataLoader:
        """Create an autoencoder dataloader where inputs and targets are identical."""

        tensor_windows = torch.from_numpy(windows)
        # Reconstruction training uses the same tensor as both input and target.
        dataset = TensorDataset(tensor_windows, tensor_windows)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""

        if self.window_splits is None:
            raise RuntimeError("Data module not prepared. Call setup first.")
        return self._to_loader(self.window_splits.train, self.batch_size, self.num_workers, True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""

        if self.window_splits is None:
            raise RuntimeError("Data module not prepared. Call setup first.")
        return self._to_loader(self.window_splits.val, self.batch_size, self.num_workers, False)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""

        if self.window_splits is None:
            raise RuntimeError("Data module not prepared. Call setup first.")
        return self._to_loader(self.window_splits.test, self.batch_size, self.num_workers, False)

    def test_window_labels(self) -> np.ndarray | None:
        """Return test window labels when available, otherwise None."""

        if self.label_splits is None:
            return None
        return self.label_splits.test

    def val_window_labels(self) -> np.ndarray | None:
        """Return validation window labels when available, otherwise None."""

        if self.label_splits is None:
            return None
        return self.label_splits.val
