"""
Run unattended monitoring when state/mode.txt == MONITOR
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
from src.monitor import run_monitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor camera for anomalies using trained autoencoder.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    logger = JsonlLogger(Path(cfg["paths"]["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="monitor_script")

    run_monitor(cfg)


if __name__ == "__main__":
    main()
