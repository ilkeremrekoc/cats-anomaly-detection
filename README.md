# Multivariate Anomaly Detection in Complex Systems

The goal of this project is to detect anomalies in complex industrial systems using multivariate
time-series data. The model learns normal behavior from 17 variables (control commands,
environment signals, and telemetry). After training, it flags unusual behavior early so failures can
be investigated before they become critical.

In simple terms: the model tries to reconstruct normal sequences. If reconstruction error is high,
that point is likely anomalous.

## Project Concept

### Input and Output Data Format

Input: 17-dimensional vector per timestamp. Features include 4 control commands, 3 environmental
stimuli and 10 sensor/telemetry readings.

Output: An anomaly score per timestamp.

### Metrics

- Precision: out of predicted anomalies, how many are truly anomalies.
- Recall: out of real anomalies, how many the model finds.
- F1-score: balance between precision and recall.
- PR-AUC: important for imbalanced data where anomalies are rare.
- ROC-AUC: checks ranking quality across different thresholds.
- Mean reconstruction error: average error level on validation/test windows.

### Validation

Since the dataset’s first million observations are guaranteed to be normal, I will use that for
training. This is especially important because during anomaly detection, we want the model to
learn normal processes.

The rest of the dataset will be separated into %20 Validation and %80 Test sets.

To ensure reproducibility, I will use DVC (Data Versioning Control) to track data versions and a
fixed random seed for all PyTorch data loaders and split indices.

### Data

I use the CATS (Controlled Anomalies Time Series) dataset by Solenix Engineering GmbH.

Link: https://zenodo.org/records/8338435

Features: 17 multivariate features (4 control signals, 3 environment stimuli, 10 sensor values).

The CSV version is very large (~1.5 GB), so I use the parquet version (~500 MB).
To handle size efficiently, preprocessing uses sliding windows and batched loading.

### Modeling Approach

- Sequence autoencoder with a causal dilated TCN backbone and a Transformer encoder.
- Training objective: reconstruction MSE on normal windows.
- Detection rule: higher reconstruction error means higher anomaly probability.

## Setup

### 1. Install dependencies

```bash
uv sync --group dev
```

### 2. Install quality hooks

```bash
uv run pre-commit install
uv run pre-commit run -a
```

## Train

### Core commands

```bash
uv run cats-train train
uv run cats-train infer
```

### Hydra overrides

```bash
uv run cats-train train training.max_epochs=20 data.window_size=64 training.learning_rate=5e-4
```

### MLflow

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

### DVC data and artifact flow

```bash
# Only needed if you add/replace the raw dataset file
uv run dvc add data/raw/cats_features.parquet

# Run pipeline stages from dvc.yaml
uv run dvc repro

# Push raw dataset pointer/cache to data remote
uv run dvc push data/raw/cats_features.parquet.dvc -r data_remote

# Push training/inference artifacts to model remote
uv run dvc push models plots outputs -r models_remote
```

### Simple training flow

```bash
uv sync --group dev
```

If you want to see metrics in MLflow UI during training, start MLflow first in another terminal:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

To get the data manually `dvc pull`. This is optional. During training, the pipeline first checks
local data, then tries DVC pull, and finally falls back to public download if needed.

```bash
uv run dvc pull data/raw/cats_features.parquet.dvc -r data_remote
```

Train

```bash
uv run cats-train train
```

## Notes

- DVC remotes are configured in `.dvc/config` (`data_remote` and `models_remote`).
- Training writes plots to `plots/` and model checkpoints to `models/`.
- Run commands from the `cats-anomaly-detection` project root.
- If MLflow is not running, training still works and logs fall back to local CSV logs.
- If local data and DVC pull are both unavailable, internet is needed for the public download fallback.
- Before inference, make sure a checkpoint exists in `models/` (either run training or pull model artifacts from `models_remote`).
- The first run can take longer because data download/preprocessing/cache may happen.
- Inference saves arrays and metrics to `outputs/`.
- When MLflow is available, the git commit hash is logged.
