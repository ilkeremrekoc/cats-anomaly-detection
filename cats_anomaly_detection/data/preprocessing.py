from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataframe(raw_data_path: Path) -> pd.DataFrame:
    """Read parquet dataset and drop metadata columns."""
    return pd.read_parquet(raw_data_path)


def extract_features_and_labels(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Return model features and optional binary labels from raw dataframe."""
    labels: np.ndarray | None = None
    if "y" in dataframe.columns:
        labels = dataframe["y"].to_numpy(dtype=np.int32)
        labels = (labels > 0).astype(np.int32)

    feature_frame = dataframe.drop(columns=["y", "category"], errors="ignore")
    return feature_frame, labels


def fit_transform_features(dataframe: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Fit standard scaler and transform feature matrix."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(dataframe.to_numpy(dtype=np.float32))
    return scaled_values.astype(np.float32), scaler


def make_sliding_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Convert sequence to overlapping windows for sequence autoencoding."""
    window_list: list[np.ndarray] = []
    final_start = len(data) - window_size

    for start_index in range(0, final_start + 1, stride):
        window_list.append(data[start_index : start_index + window_size])

    if not window_list:
        raise ValueError("No windows were created. Increase data size or reduce window_size.")

    return np.stack(window_list).astype(np.float32)


def make_window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Create one binary label per window using max label within each window."""
    label_list: list[int] = []
    final_start = len(labels) - window_size

    for start_index in range(0, final_start + 1, stride):
        window_target = int(np.max(labels[start_index : start_index + window_size]))
        label_list.append(window_target)

    if not label_list:
        raise ValueError("No labels were created. Increase data size or reduce window_size.")

    return np.asarray(label_list, dtype=np.int32)


def split_windows(
    windows: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split windows into train/validation/test sets by temporal order."""
    total_windows = len(windows)
    train_end = int(total_windows * train_ratio)
    val_end = train_end + int(total_windows * val_ratio)

    train_windows = windows[:train_end]
    val_windows = windows[train_end:val_end]
    test_windows = windows[val_end:]

    if len(train_windows) == 0 or len(val_windows) == 0 or len(test_windows) == 0:
        raise ValueError("Train, val, and test splits must all be non-empty.")

    return train_windows, val_windows, test_windows
