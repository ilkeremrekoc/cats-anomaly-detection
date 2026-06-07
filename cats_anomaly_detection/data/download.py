from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd


def try_dvc_pull(target_path: Path) -> bool:
    """Attempt to fetch data from DVC remote."""
    dvc_target = target_path.with_name(f"{target_path.name}.dvc")

    if not dvc_target.exists():
        print(f"No DVC file found for {target_path}. Skipping DVC pull.")
        return False

    try:
        print(f"Trying DVC pull for {dvc_target} (timeout: 30s)...")
        start_time = time.perf_counter()
        subprocess.run(
            ["dvc", "pull", str(dvc_target)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = time.perf_counter() - start_time
        print(f"DVC pull finished in {elapsed:.1f}s.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("DVC pull failed or timed out. Falling back to public download.")
        return False


def download_data(download_url: str, output_path: Path) -> Path:
    """Download a parquet file and save it to output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading data from public source to {output_path}...")
    start_time = time.perf_counter()
    source_data = pd.read_parquet(download_url)
    source_data.to_parquet(output_path, index=False)
    elapsed = time.perf_counter() - start_time
    print(f"Download finished in {elapsed:.1f}s.")
    return output_path


def ensure_data_available(raw_data_path: str, download_url: str) -> Path:
    """Resolve dataset with this priority: local file, DVC pull, public download."""
    dataset_path = Path(raw_data_path)
    if dataset_path.exists():
        print(f"Using local data at {dataset_path}.")
        return dataset_path
    else:
        print(f"Local data not found at {dataset_path}.")

    if try_dvc_pull(dataset_path) and dataset_path.exists():
        print(f"Using data restored by DVC at {dataset_path}.")
        return dataset_path
    else:
        print("DVC restore did not produce the dataset file.")
        print("Falling back to public download.")

    return download_data(
        download_url=download_url,
        output_path=dataset_path,
    )
