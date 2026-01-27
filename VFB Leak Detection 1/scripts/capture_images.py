"""
Capture frames from the USB camera to build a 'normal' dataset.

Only captures when state/mode.txt == CAPTURE
"""
from __future__ import annotations


import sys
from pathlib import Path

# Allow running this script from any working directory (e.g., double-clicking a .bat file).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import time
from pathlib import Path

from src.camera import Camera, CameraConfig
from src.config import ensure_directories, load_config, read_mode_file
from src.data import save_bgr_image
from src.logging_utils import JsonlLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture normal images from USB camera.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    paths = cfg["paths"]
    cam_cfg = cfg["camera"]
    cap_cfg = cfg["capture"]

    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="capture_images")

    camera = Camera(CameraConfig(
        index=int(cam_cfg["index"]),
        backend=str(cam_cfg.get("backend", "DSHOW")),
        width=int(cam_cfg.get("width", 1280)),
        height=int(cam_cfg.get("height", 720)),
        fps=int(cam_cfg.get("fps", 30)),
    ))

    mode_file = paths["state_mode_file"]
    out_dir = paths["data_normal_dir"]
    interval = float(cap_cfg.get("interval_seconds", 1.5))
    jpg_quality = int(cap_cfg.get("jpg_quality", 95))

    last_mode = None
    last_capture_ts = 0.0

    try:
        while True:
            mode = read_mode_file(mode_file)
            if mode != last_mode:
                logger.log("mode_change", mode=mode)
                last_mode = mode

            if mode != "CAPTURE":
                time.sleep(1.0)
                continue

            now = time.time()
            if now - last_capture_ts < interval:
                time.sleep(0.05)
                continue

            ok, frame = camera.read()
            if not ok or frame is None:
                logger.log("camera_error", message="Failed to read frame during CAPTURE.")
                time.sleep(1.0)
                continue

            saved = save_bgr_image(frame, out_dir, prefix="normal", jpg_quality=jpg_quality)
            logger.log("captured_frame", image_path=str(saved.resolve()))
            last_capture_ts = now

    except KeyboardInterrupt:
        logger.log("shutdown", component="capture_images", reason="KeyboardInterrupt")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
