from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def try_dvc_pull(target_path: Path) -> bool:
    """Attempt to fetch data from DVC remote."""
    try:
        subprocess.run(
            ["dvc", "pull", str(target_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_data(download_url: str, output_path: Path) -> Path:
    """Download a parquet file and save it to output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_data = pd.read_parquet(download_url)
    source_data.to_parquet(output_path, index=False)
    return output_path


def ensure_data_available(raw_data_path: str, download_url: str) -> Path:
    """Resolve dataset with this priority: local file, DVC pull, public download."""
    dataset_path = Path(raw_data_path)
    if dataset_path.exists():
        return dataset_path

    if try_dvc_pull(dataset_path) and dataset_path.exists():
        return dataset_path

    return download_data(
        download_url=download_url,
        output_path=dataset_path,
    )
