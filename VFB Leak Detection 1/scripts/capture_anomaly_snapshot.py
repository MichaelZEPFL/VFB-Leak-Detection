"""
Capture one camera frame and save anomaly-style artifacts for testing.

This command is useful for evaluating camera framing and model diagnostics
without waiting for a real anomaly trigger.

Outputs are saved to data/anomalies/ using the same artifact style as monitor.py:
- anomaly_*.jpg (ROI-cropped full-resolution snapshot)
- input_resized_*.jpg (model input resolution)
- reconstruction_*.jpg
- error_heatmap_*.jpg
- compare_*.jpg (input | reconstruction | heatmap)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script from any working directory (e.g., double-clicking a .bat file).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import tensorflow as tf

from src.camera import Camera, CameraConfig
from src.config import ensure_directories, load_config
from src.data import apply_roi, now_timestamp, save_bgr_image
from src.logging_utils import JsonlLogger
from src.model import score_from_reconstruction
from src.monitor import _load_threshold, _make_error_heatmap_bgr, _preprocess_frame_bgr, _rgb01_to_bgr_uint8


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one anomaly-style snapshot + heatmap from the camera.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--warmup_reads", type=int, default=5, help="Warm-up camera reads before capturing.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    paths = cfg["paths"]
    img_cfg = cfg["image"]
    cam_cfg = cfg["camera"]
    roi_cfg = dict(cfg.get("roi", {}) or {})
    scoring_cfg = dict(cfg.get("scoring", {}) or {})

    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="capture_anomaly_snapshot")

    model_dir = Path(paths["model_dir"])
    model_path = model_dir / "autoencoder.keras"
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path.resolve()}\n"
            "Train first using: python .\\scripts\\train_autoencoder.py"
        )

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {model_path.resolve()}\nError: {e}") from e

    threshold = None
    percentile = None
    try:
        threshold, percentile, _ = _load_threshold(model_dir)
    except Exception:
        # Threshold is optional for this utility; we still save artifacts.
        threshold = None
        percentile = None

    camera = Camera(
        CameraConfig(
            index=int(cam_cfg["index"]),
            backend=str(cam_cfg.get("backend", "DSHOW")),
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
            fps=int(cam_cfg.get("fps", 30)),
        )
    )

    try:
        for _ in range(max(0, int(args.warmup_reads))):
            camera.read()

        ok, frame_bgr = camera.read()
        if not ok or frame_bgr is None:
            raise SystemExit("Failed to read frame from camera.")

        frame_roi = apply_roi(frame_bgr, roi_cfg)
        anomalies_dir = Path(paths["data_anomalies_dir"])
        anomalies_dir.mkdir(parents=True, exist_ok=True)

        event_id = now_timestamp()
        jpg_quality = int(cfg["capture"].get("jpg_quality", 95))

        x = _preprocess_frame_bgr(frame_roi, int(img_cfg["width"]), int(img_cfg["height"]))
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        xhat_tf = model(x_tf, training=False)

        score = float(
            score_from_reconstruction(
                x_tf,
                xhat_tf,
                method=str(scoring_cfg.get("method", "mean")),
                topk_percent=float(scoring_cfg.get("topk_percent", 1.0)),
                topk_min_pixels=int(scoring_cfg.get("topk_min_pixels", 100)),
            ).numpy()[0]
        )

        extra_suffix = f"manual_score{score:.6f}"
        saved_paths: list[str] = []

        p_anom = save_bgr_image(
            frame_roi,
            anomalies_dir,
            prefix="anomaly",
            jpg_quality=jpg_quality,
            extra_suffix=extra_suffix,
            timestamp=event_id,
        )
        saved_paths.append(str(p_anom.resolve()))

        xin = x[0]
        xhat = xhat_tf.numpy()[0]

        xin_bgr = _rgb01_to_bgr_uint8(xin)
        p_input = save_bgr_image(
            xin_bgr,
            anomalies_dir,
            prefix="input_resized",
            jpg_quality=jpg_quality,
            extra_suffix=extra_suffix,
            timestamp=event_id,
        )
        saved_paths.append(str(p_input.resolve()))

        recon_bgr = _rgb01_to_bgr_uint8(xhat)
        p_recon = save_bgr_image(
            recon_bgr,
            anomalies_dir,
            prefix="reconstruction",
            jpg_quality=jpg_quality,
            extra_suffix=extra_suffix,
            timestamp=event_id,
        )
        saved_paths.append(str(p_recon.resolve()))

        heat_bgr = _make_error_heatmap_bgr(xin, xhat)
        p_heat = save_bgr_image(
            heat_bgr,
            anomalies_dir,
            prefix="error_heatmap",
            jpg_quality=jpg_quality,
            extra_suffix=extra_suffix,
            timestamp=event_id,
        )
        saved_paths.append(str(p_heat.resolve()))

        compare_bgr = np.concatenate([xin_bgr, recon_bgr, heat_bgr], axis=1)
        p_compare = save_bgr_image(
            compare_bgr,
            anomalies_dir,
            prefix="compare",
            jpg_quality=jpg_quality,
            extra_suffix=extra_suffix,
            timestamp=event_id,
        )
        saved_paths.append(str(p_compare.resolve()))

        logger.log(
            "manual_anomaly_snapshot",
            component="capture_anomaly_snapshot",
            event_id=event_id,
            score=score,
            threshold=threshold,
            percentile=percentile,
            scoring_method=str(scoring_cfg.get("method", "mean")),
            topk_percent=float(scoring_cfg.get("topk_percent", 1.0)),
            topk_min_pixels=int(scoring_cfg.get("topk_min_pixels", 100)),
            saved_paths=saved_paths,
        )

        print("Manual anomaly-style snapshot complete.")
        print(f"Event ID: {event_id}")
        print(f"Score: {score:.6f}")
        if threshold is not None and percentile is not None:
            print(f"Threshold (p{percentile}): {threshold:.6f}")
            print(f"Would trigger score alert: {score > threshold}")
        print("Saved files:")
        for p in saved_paths:
            print(f"- {p}")

    finally:
        camera.release()


if __name__ == "__main__":
    main()
