"""
Set the operating mode (CAPTURE / MONITOR / OFF) by writing state/mode.txt.

This is friendlier than manually editing the file.

Examples:
  python .\scripts\set_mode.py CAPTURE
  python .\scripts\set_mode.py MONITOR
  python .\scripts\set_mode.py OFF
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import ensure_directories, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Set mode (CAPTURE / MONITOR / OFF).")
    parser.add_argument("mode", type=str, help="CAPTURE, MONITOR, or OFF")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    mode = str(args.mode).strip().upper()
    if mode not in {"CAPTURE", "MONITOR", "OFF"}:
        raise SystemExit("Mode must be one of: CAPTURE, MONITOR, OFF")

    mode_file = Path(cfg["paths"]["state_mode_file"])
    mode_file.parent.mkdir(parents=True, exist_ok=True)
    mode_file.write_text(mode + "\n", encoding="utf-8")

    print(f"Mode set to: {mode}")
    print(f"Mode file: {mode_file.resolve()}")


if __name__ == "__main__":
    main()
