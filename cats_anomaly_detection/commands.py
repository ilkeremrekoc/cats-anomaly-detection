from __future__ import annotations

from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from cats_anomaly_detection.training.inference import run_inference
from cats_anomaly_detection.training.runner import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(overrides: tuple[str, ...]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides))


class Commands:
    """CLI commands"""

    def train(self, *overrides: str) -> None:
        """Run model training with overrides."""
        cfg = load_config(overrides)
        run_training(cfg)

    def infer(self, *overrides: str) -> None:
        """Run reconstruction-error scoring on the test split."""
        cfg = load_config(overrides)
        run_inference(cfg)


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
