"""
Train the convolutional autoencoder on normal images.

Run manually after capturing enough "normal" frames.
"""
from __future__ import annotations


import sys
from pathlib import Path

# Allow running this script from any working directory (e.g., double-clicking a .bat file).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from pathlib import Path

from src.config import ensure_directories, load_config
from src.logging_utils import JsonlLogger
from src.train import train_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoencoder for anomaly detection.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    logger = JsonlLogger(Path(cfg["paths"]["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="train_autoencoder")

    result = train_autoencoder(cfg)
    logger.log("training_complete", **result)

    print("Training complete.")
    print(f"Model saved to: {result['model_path']}")
    print(f"Threshold (p{result['percentile']}): {result['threshold']:.6f}")
    print(f"Stats: {result['stats_path']}")


if __name__ == "__main__":
    main()
