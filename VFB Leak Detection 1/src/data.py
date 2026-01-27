"""
Data and filesystem helpers.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional
import os

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


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
