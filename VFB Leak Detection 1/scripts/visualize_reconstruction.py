"""
Visualize autoencoder reconstructions + error heatmaps.

This script helps humans validate what the model considers "normal" and why a frame
was flagged as anomalous.

Examples (run from project root):

  # Visualize a few images from a directory (default: data/normal)
  python .\scripts\visualize_reconstruction.py --num 16

  # Grab frames from the USB camera instead of disk
  python .\scripts\visualize_reconstruction.py --use_camera --num 8

Outputs are written to: reports/reconstruction_viz/

Each output image is a side-by-side panel:
  [input_resized | reconstruction | error_heatmap]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script from any working directory (e.g., desktop shortcuts).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

# Headless-safe plotting (we only save images/files)
import matplotlib
matplotlib.use("Agg")  # noqa: E402

from src.camera import Camera, CameraConfig
from src.config import ensure_directories, load_config
from src.data import list_image_files, now_timestamp
from src.logging_utils import JsonlLogger


def _rgb01_to_bgr_uint8(rgb01: np.ndarray) -> np.ndarray:
    rgb_u8 = np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def _make_error_heatmap_bgr(x_rgb01: np.ndarray, xhat_rgb01: np.ndarray) -> np.ndarray:
    # Per-pixel error averaged across channels.
    err = np.mean(np.square(x_rgb01 - xhat_rgb01), axis=2)  # [H,W]

    # Normalize robustly for visualization.
    p99 = float(np.percentile(err, 99.0))
    denom = (p99 if p99 > 1e-12 else float(err.max())) + 1e-12
    err_norm = np.clip(err / denom, 0.0, 1.0)

    err_u8 = np.clip(err_norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(err_u8, cv2.COLORMAP_JET)


def _preprocess_bgr(frame_bgr: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    """BGR uint8 -> RGB float32 [1,H,W,3] in [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_w, image_h), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def _read_image_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _sample_paths(paths: List[Path], n: int, seed: int, random_sample: bool) -> List[Path]:
    if n <= 0:
        return []
    if not paths:
        return []
    if n >= len(paths):
        return paths
    if random_sample:
        rng = random.Random(seed)
        return rng.sample(paths, n)
    return paths[:n]


def _annotate_panel(panel_bgr: np.ndarray, *, title: str, score: float) -> np.ndarray:
    out = panel_bgr.copy()
    text = f"{title} | score={score:.6f}"
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _make_panel(
    *,
    model: tf.keras.Model,
    frame_bgr: np.ndarray,
    image_w: int,
    image_h: int,
) -> Tuple[np.ndarray, float]:
    x = _preprocess_bgr(frame_bgr, image_w, image_h)  # [1,H,W,3] RGB01
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    xhat_tf = model(x_tf, training=False)

    score_tf = tf.reduce_mean(tf.square(x_tf - xhat_tf), axis=[1, 2, 3])
    score = float(score_tf.numpy()[0])

    xin = x[0]
    xhat = xhat_tf.numpy()[0]

    xin_bgr = _rgb01_to_bgr_uint8(xin)
    xhat_bgr = _rgb01_to_bgr_uint8(xhat)
    heat_bgr = _make_error_heatmap_bgr(xin, xhat)

    panel = np.concatenate([xin_bgr, xhat_bgr, heat_bgr], axis=1)
    return panel, score


def _make_montage(panels: List[np.ndarray], cols: int) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to montage.")
    cols = max(1, int(cols))
    tile_h, tile_w = panels[0].shape[:2]
    rows = int(math.ceil(len(panels) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r = i // cols
        c = i % cols
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = p
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reconstructions + error heatmaps.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input_dir", type=str, default="", help="Directory of images to visualize (default: data/normal)")
    parser.add_argument("--use_camera", action="store_true", help="Capture frames from the USB camera instead of disk.")
    parser.add_argument("--num", type=int, default=16, help="Number of samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --random is set).")
    parser.add_argument("--random", action="store_true", help="Randomly sample images from input_dir.")
    parser.add_argument("--out_dir", type=str, default="reports/reconstruction_viz", help="Output directory for visualizations.")
    parser.add_argument("--grid_cols", type=int, default=4, help="Columns in the montage grid.")
    parser.add_argument("--no_grid", action="store_true", help="Do not write a montage grid image.")
    parser.add_argument("--no_individual", action="store_true", help="Do not save individual panel images (only the montage).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    paths = cfg["paths"]
    img_cfg = cfg["image"]
    cam_cfg = cfg["camera"]

    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="visualize_reconstruction")

    model_path = Path(paths["model_dir"]) / "autoencoder.keras"
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path.resolve()}\n"
            "Train first using: python .\\scripts\\train_autoencoder.py"
        )

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {model_path.resolve()}\nError: {e}") from e

    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_w = int(img_cfg["width"])
    image_h = int(img_cfg["height"])

    panels: List[np.ndarray] = []
    saved_files: List[str] = []
    run_id = now_timestamp()

    if args.use_camera:
        camera = Camera(CameraConfig(
            index=int(cam_cfg["index"]),
            backend=str(cam_cfg.get("backend", "DSHOW")),
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
            fps=int(cam_cfg.get("fps", 30)),
        ))
        try:
            # Warm-up reads (common for webcams)
            for _ in range(5):
                camera.read()

            for i in range(int(args.num)):
                ok, frame = camera.read()
                if not ok or frame is None:
                    logger.log("camera_error", component="visualize_reconstruction", message="Failed to read frame.")
                    continue

                panel, score = _make_panel(model=model, frame_bgr=frame, image_w=image_w, image_h=image_h)
                panel = _annotate_panel(panel, title=f"camera_i{i:02d}", score=score)
                panels.append(panel)

                if not args.no_individual:
                    out_path = out_dir / f"panel_{run_id}_camera_i{i:02d}_score{score:.6f}.jpg"
                    cv2.imwrite(str(out_path), panel)
                    saved_files.append(str(out_path.resolve()))
        finally:
            camera.release()
    else:
        input_dir = Path(args.input_dir) if args.input_dir else Path(paths["data_normal_dir"])
        if not input_dir.is_absolute():
            input_dir = (PROJECT_ROOT / input_dir).resolve()
        image_paths = list_image_files(input_dir)
        if not image_paths:
            raise SystemExit(f"No images found in: {input_dir.resolve()}")

        sample_paths = _sample_paths(image_paths, int(args.num), int(args.seed), bool(args.random))
        for i, p in enumerate(sample_paths):
            img_bgr = _read_image_bgr(p)
            if img_bgr is None:
                logger.log("read_error", component="visualize_reconstruction", path=str(p), message="cv2.imread returned None")
                continue

            panel, score = _make_panel(model=model, frame_bgr=img_bgr, image_w=image_w, image_h=image_h)
            panel = _annotate_panel(panel, title=p.name, score=score)
            panels.append(panel)

            if not args.no_individual:
                out_path = out_dir / f"panel_{run_id}_i{i:02d}_{p.stem}_score{score:.6f}.jpg"
                cv2.imwrite(str(out_path), panel)
                saved_files.append(str(out_path.resolve()))

    if not panels:
        raise SystemExit("No panels were generated (camera/images may have failed).")

    montage_path = None
    if not args.no_grid:
        montage = _make_montage(panels, cols=int(args.grid_cols))
        montage_path = out_dir / f"montage_{run_id}_n{len(panels)}.jpg"
        cv2.imwrite(str(montage_path), montage)

    logger.log(
        "reconstruction_viz_complete",
        component="visualize_reconstruction",
        out_dir=str(out_dir.resolve()),
        saved_files=[Path(p).name for p in saved_files],
        montage_file=(montage_path.name if montage_path else None),
    )

    print("Reconstruction visualization complete.")
    print(f"Output directory: {out_dir.resolve()}")
    if montage_path:
        print(f"Montage: {montage_path.resolve()}")
    print(f"Panels saved: {len(saved_files)}")


if __name__ == "__main__":
    main()
