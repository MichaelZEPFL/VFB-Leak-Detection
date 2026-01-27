"""
Threshold calibration + histogram plot.

Computes reconstruction errors for a folder of "normal" images and produces:
- Error histogram plot with percentile threshold line
- Summary JSON with key statistics

This is useful to:
- sanity-check the chosen threshold
- detect drift (lighting/camera changes) by comparing plots across days

Examples:

  # Use normal images in data/normal and current trained model
  python .\scripts\threshold_calibration.py

  # Use a separate folder of validation normals and overwrite model/threshold.json
  python .\scripts\threshold_calibration.py --input_dir data\val_normals --write_threshold

Outputs go to: reports/threshold_calibration/
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.config import ensure_directories, load_config
from src.data import list_image_files, now_timestamp
from src.logging_utils import JsonlLogger


def _preprocess_path(path: Path, image_w: int, image_h: int) -> Optional[np.ndarray]:
    """Load image -> RGB float32 [H,W,3] in [0,1]. Returns None on read failure."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_w, image_h), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return x


def _compute_errors(
    model: tf.keras.Model,
    image_paths: List[Path],
    *,
    image_w: int,
    image_h: int,
    batch_size: int,
) -> np.ndarray:
    xs: List[np.ndarray] = []
    errors: List[float] = []

    def flush() -> None:
        nonlocal xs, errors
        if not xs:
            return
        batch = np.stack(xs, axis=0)  # [B,H,W,3]
        x_tf = tf.convert_to_tensor(batch, dtype=tf.float32)
        xhat_tf = model(x_tf, training=False)
        batch_err = tf.reduce_mean(tf.square(x_tf - xhat_tf), axis=[1, 2, 3]).numpy().tolist()
        errors.extend(batch_err)
        xs = []

    for p in image_paths:
        x = _preprocess_path(p, image_w, image_h)
        if x is None:
            continue
        xs.append(x)
        if len(xs) >= batch_size:
            flush()

    flush()
    return np.array(errors, dtype=np.float32)


def _load_existing_threshold(model_dir: Path) -> Optional[Tuple[float, float]]:
    p = model_dir / "threshold.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        thr = float(obj["threshold"])
        pct = float(obj.get("percentile", 99.5))
        if not np.isfinite(thr) or not np.isfinite(pct):
            return None
        return thr, pct
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reconstruction-error histogram + threshold plot.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input_dir", type=str, default="", help="Directory of normal images (default: data/normal)")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile for threshold (default: config threshold.percentile)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference when computing errors.")
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on number of images (0 = all).")
    parser.add_argument("--out_dir", type=str, default="reports/threshold_calibration", help="Directory to write plots and stats.")
    parser.add_argument("--write_threshold", action="store_true", help="Overwrite model/threshold.json with computed threshold (backs up existing file).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    paths = cfg["paths"]
    img_cfg = cfg["image"]
    model_dir = Path(paths["model_dir"])
    model_path = model_dir / "autoencoder.keras"

    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="threshold_calibration")

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

    image_w = int(img_cfg["width"])
    image_h = int(img_cfg["height"])

    input_dir = Path(args.input_dir) if args.input_dir else Path(paths["data_normal_dir"])
    if not input_dir.is_absolute():
        input_dir = (PROJECT_ROOT / input_dir).resolve()

    image_paths = list_image_files(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir.resolve()}")

    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]

    percentile = float(args.percentile) if args.percentile is not None else float(cfg["threshold"].get("percentile", 99.5))

    errors = _compute_errors(
        model,
        image_paths,
        image_w=image_w,
        image_h=image_h,
        batch_size=int(args.batch_size),
    )

    if errors.size == 0:
        raise SystemExit("No reconstruction errors computed (all images may have failed to load).")

    computed_thr = float(np.percentile(errors, percentile))

    existing = _load_existing_threshold(model_dir)
    existing_thr = existing[0] if existing else None
    existing_pct = existing[1] if existing else None

    run_id = now_timestamp()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=60)
    plt.axvline(computed_thr, linewidth=2, label=f"computed p{percentile}: {computed_thr:.6f}")
    if existing_thr is not None:
        plt.axvline(existing_thr, linestyle="--", linewidth=2, label=f"existing p{existing_pct}: {existing_thr:.6f}")
    plt.title("Reconstruction error histogram (normal images)")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / f"threshold_hist_{run_id}_n{len(errors)}.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    stats = {
        "run_id": run_id,
        "input_dir": str(input_dir.resolve()),
        "n_images": int(len(errors)),
        "percentile": percentile,
        "computed_threshold": computed_thr,
        "mean": float(errors.mean()),
        "std": float(errors.std()),
        "min": float(errors.min()),
        "max": float(errors.max()),
        "p95": float(np.percentile(errors, 95.0)),
        "p99": float(np.percentile(errors, 99.0)),
        "p99_5": float(np.percentile(errors, 99.5)),
        "existing_threshold": existing_thr,
        "existing_percentile": existing_pct,
        "model_path": str(model_path.resolve()),
        "plot_path": str(plot_path.resolve()),
    }

    stats_path = out_dir / f"threshold_stats_{run_id}.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if args.write_threshold:
        thr_file = model_dir / "threshold.json"
        if thr_file.exists():
            backup = model_dir / f"threshold_backup_{run_id}.json"
            try:
                backup.write_text(thr_file.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                # If backup fails, do not overwrite.
                raise SystemExit(f"Refusing to overwrite threshold.json because backup failed: {backup.resolve()}")
        new_obj = {"percentile": percentile, "threshold": computed_thr}
        thr_file.write_text(json.dumps(new_obj, indent=2), encoding="utf-8")

    logger.log(
        "threshold_calibration_complete",
        component="threshold_calibration",
        input_dir=str(input_dir.resolve()),
        percentile=percentile,
        computed_threshold=computed_thr,
        plot=str(plot_path.resolve()),
        stats=str(stats_path.resolve()),
        wrote_threshold=bool(args.write_threshold),
    )

    print("Threshold calibration complete.")
    print(f"Input dir: {input_dir.resolve()}")
    print(f"Computed threshold p{percentile}: {computed_thr:.6f}")
    if existing_thr is not None:
        print(f"Existing threshold p{existing_pct}: {existing_thr:.6f}")
    print(f"Plot: {plot_path.resolve()}")
    print(f"Stats: {stats_path.resolve()}")
    if args.write_threshold:
        print(f"Updated threshold.json in: {model_dir.resolve()}")


if __name__ == "__main__":
    main()
