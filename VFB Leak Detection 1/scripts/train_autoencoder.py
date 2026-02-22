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
from src.notify import EmailConfig, Notifier, SlackConfig
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

    # Best-effort completion notification (Slack preferred, email fallback if enabled).
    try:
        slack_cfg = cfg["notify"]["slack"]
        email_cfg = cfg["notify"]["email"]
        notifier = Notifier(
            slack=SlackConfig(
                enabled=bool(slack_cfg.get("enabled", True)),
                webhook_url=str(slack_cfg.get("webhook_url", "") or "").strip(),
            ),
            email=EmailConfig(
                enabled=bool(email_cfg.get("enabled", False)),
                smtp_host=str(email_cfg.get("smtp_host", "")),
                smtp_port=int(email_cfg.get("smtp_port", 587)),
                smtp_user=str(email_cfg.get("smtp_user", "")),
                smtp_password=str(email_cfg.get("smtp_password", "")),
                from_addr=str(email_cfg.get("from_addr", "")),
                to_addrs=list(email_cfg.get("to_addrs", [])),
                use_tls=bool(email_cfg.get("use_tls", True)),
            ),
        )
        notifier.notify(
            "Leak monitor: training complete",
            (
                "Autoencoder training completed successfully.\n"
                f"- Model: {result['model_path']}\n"
                f"- Threshold (p{result['percentile']}): {result['threshold']:.6f}\n"
                f"- Stats: {result['stats_path']}"
            ),
        )
    except Exception as e:
        logger.log("notify_error", component="train_autoencoder", error=str(e), title="training_complete")

    print("Training complete.")
    print(f"Model saved to: {result['model_path']}")
    print(f"Threshold (p{result['percentile']}): {result['threshold']:.6f}")
    print(f"Stats: {result['stats_path']}")


if __name__ == "__main__":
    main()
