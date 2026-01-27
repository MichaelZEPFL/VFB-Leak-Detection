"""
OpenCV camera wrapper with basic robustness:
- configurable backend (Windows-friendly)
- automatic reopen with backoff on repeated failures
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


_BACKEND_MAP = {
    "DEFAULT": 0,
    "DSHOW": getattr(cv2, "CAP_DSHOW", 0),
    "MSMF": getattr(cv2, "CAP_MSMF", 0),
}


@dataclass
class CameraConfig:
    index: int = 0
    backend: str = "DSHOW"
    width: int = 1280
    height: int = 720
    fps: int = 30


class Camera:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.consecutive_failures = 0
        self.last_open_attempt = 0.0

    def open(self) -> None:
        backend_flag = _BACKEND_MAP.get(self.cfg.backend.upper(), 0)
        # Avoid hammering open() if camera is down.
        now = time.time()
        if now - self.last_open_attempt < 1.0 and self.cap is not None and self.cap.isOpened():
            return

        self.last_open_attempt = now

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.cfg.index, backend_flag)
        if not self.cap.isOpened():
            # Keep cap object but report failure through read()
            return

        # Best-effort camera settings.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))

        self.consecutive_failures = 0

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns:
            (ok, frame_bgr)
        """
        if self.cap is None or not self.cap.isOpened():
            self.open()

        if self.cap is None or not self.cap.isOpened():
            self.consecutive_failures += 1
            return False, None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.consecutive_failures += 1
            # If a few reads fail, try reopening.
            if self.consecutive_failures >= 5:
                self.open()
            return False, None

        self.consecutive_failures = 0
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
