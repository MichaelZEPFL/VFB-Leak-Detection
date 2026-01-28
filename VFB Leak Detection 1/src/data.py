"""
Data and filesystem helpers.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Iterable, List, Optional
import os

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def apply_roi(frame_bgr: np.ndarray, roi_cfg: Optional[dict[str, Any]]) -> np.ndarray:
    """
    Apply a rectangular ROI crop to a BGR frame.

    roi_cfg schema:
      - enabled: bool
      - units: "relative" (fractions of width/height) or "pixels"
      - x, y, width, height: numbers

    Returns:
        Cropped frame if ROI is enabled and valid, otherwise the original frame.
    """
    if not roi_cfg or not bool(roi_cfg.get("enabled", False)):
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    units = str(roi_cfg.get("units", "relative")).lower()

    def _clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(val, hi))

    try:
        if units == "pixels":
            x = int(round(float(roi_cfg.get("x", 0))))
            y = int(round(float(roi_cfg.get("y", 0))))
            roi_w = int(round(float(roi_cfg.get("width", w))))
            roi_h = int(round(float(roi_cfg.get("height", h))))
        else:
            x = int(round(float(roi_cfg.get("x", 0.0)) * w))
            y = int(round(float(roi_cfg.get("y", 0.0)) * h))
            roi_w = int(round(float(roi_cfg.get("width", 1.0)) * w))
            roi_h = int(round(float(roi_cfg.get("height", 1.0)) * h))
    except Exception:
        return frame_bgr

    x = _clamp(x, 0, max(0, w - 1))
    y = _clamp(y, 0, max(0, h - 1))
    roi_w = _clamp(roi_w, 1, w - x)
    roi_h = _clamp(roi_h, 1, h - y)

    if roi_w <= 0 or roi_h <= 0:
        return frame_bgr

    return frame_bgr[y : y + roi_h, x : x + roi_w]


def now_timestamp() -> str:
    # Example: 20260127_142233_123
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def list_image_files(dir_path: str | Path) -> List[Path]:
    p = Path(dir_path)
    if not p.exists():
        return []
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def save_bgr_image(
    frame_bgr: np.ndarray,
    out_dir: str | Path,
    *,
    prefix: str = "frame",
    jpg_quality: int = 95,
    extra_suffix: str = "",
    timestamp: str | None = None,
) -> Path:
    """
    Save BGR image to out_dir with timestamp name.

    Returns:
        Path to saved file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Allow callers to provide a stable timestamp to group multiple files from the same event.
    ts = timestamp or now_timestamp()
    suffix = f"_{extra_suffix}" if extra_suffix else ""
    filename = f"{prefix}_{ts}{suffix}.jpg"
    path = out / filename

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok = cv2.imwrite(str(path), frame_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return path
