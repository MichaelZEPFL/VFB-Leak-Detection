"""
Configuration loading utilities.

Precedence:
  1) Environment variables (secrets)
  2) config/config.yaml
  3) Built-in defaults

This module keeps config as a nested dictionary (simple + flexible).
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "index": 0,
        # For Windows webcams, DSHOW is often the least glitchy.
        # Alternatives: "MSMF" or "DEFAULT"
        "backend": "DSHOW",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "roi": {
        # Crop the camera feed before saving/scoring to ignore irrelevant areas.
        # units: "relative" uses fractions of width/height; "pixels" uses absolute coords.
        "enabled": False,
        "units": "relative",
        "x": 0.0,
        "y": 0.0,
        "width": 1.0,
        "height": 1.0,
    },
    "image": {
        "width": 256,
        "height": 256,
    },
    "paths": {
        "data_normal_dir": "data/normal",
        "data_anomalies_dir": "data/anomalies",
        "model_dir": "model",
        "logs_dir": "logs",
        "state_mode_file": "state/mode.txt",
    },
    "capture": {
        "interval_seconds": 1.5,
        "jpg_quality": 95,
    },
    "training": {
        "batch_size": 16,
        "epochs": 30,
        "learning_rate": 1e-3,
        "validation_split": 0.2,
        "seed": 42,
        "augment": {
            "enabled": True,
            # Values are fractions (0.02 = ~2%).
            "rotation": 0.02,
            "translation": 0.02,
            # Contrast/brightness max delta fractions
            "contrast": 0.10,
            "brightness": 0.10,
        },
    },
    "threshold": {
        "percentile": 99.5,
    },
    "scoring": {
        # method: "mean" (global MSE) or "topk" (mean of top-k pixel errors)
        "method": "mean",
        # Percentage of highest-error pixels to average when method == "topk"
        "topk_percent": 1.0,
        # Minimum number of pixels to include in top-k (guards tiny ROIs)
        "topk_min_pixels": 100,
    },
    "heatmap_trigger": {
        # Second detection trigger based on localized high-error blobs in the
        # per-pixel reconstruction error heatmap.
        "enabled": True,
        # Pixel-error percentile (computed on validation normal pixels) used
        # to binarize the heatmap into candidate anomaly regions.
        "pixel_error_percentile": 99.9,
        # Blob-area percentile (computed on validation normal images from
        # largest connected component areas) used as anomaly area threshold.
        "blob_area_percentile": 99.5,
        # Connectivity for connected components (4 or 8).
        "connectivity": 8,
        # Optional floor to avoid tiny noisy blobs triggering alerts.
        "min_blob_area": 16,
    },
    "monitor": {
        "frame_interval_seconds": 1.0,
        "consecutive_anomalies": 3,
        "alert_cooldown_seconds": 300,
        # Log a heartbeat "frame_status" event at most once every N seconds.
        "heartbeat_seconds": 60,
        # If camera hasn't produced a valid frame for this many seconds, alert.
        "camera_down_seconds": 30,
        # Save reconstruction artifacts alongside anomalies (useful for debugging false positives).
        # These are only written when an alert is actually triggered.
        "save_reconstruction": True,
        "save_error_heatmap": True,
        # Save the resized model-input frame (at autoencoder resolution).
        "save_input_resized": True,
        # Save a side-by-side compare image: input | reconstruction | heatmap
        "save_compare_image": True,
        "save_context": {
            "enabled": False,
            "pre_frames": 1,
            "post_frames": 1,
        },
    },
    "notify": {
        "slack": {
            "enabled": True,
            # Filled from env var SLACK_WEBHOOK_URL
            "webhook_url": "",
        },
        "email": {
            "enabled": False,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "",
            "from_addr": "",
            "to_addrs": [],
            "use_tls": True,
            # Filled from env var SMTP_PASSWORD
            "smtp_password": "",
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    # Secrets
    slack = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if slack:
        cfg["notify"]["slack"]["webhook_url"] = slack

    smtp_pw = os.environ.get("SMTP_PASSWORD", "").strip()
    if smtp_pw:
        cfg["notify"]["email"]["smtp_password"] = smtp_pw


def _resolve_paths_in_config(cfg: Dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    for key in ["data_normal_dir", "data_anomalies_dir", "model_dir", "logs_dir", "state_mode_file"]:
        if key in paths and isinstance(paths[key], str):
            paths[key] = str(_resolve_path(paths[key]))


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load config from YAML and environment variables.

    Args:
        config_path: Optional explicit path. Default: PROJECT_ROOT/config/config.yaml

    Returns:
        Nested dict config with resolved absolute paths (as strings).
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        # Resolve relative paths against the project root so scripts work
        # even when launched from a different working directory (e.g., desktop shortcuts).
        if not config_path.is_absolute():
            config_path = (PROJECT_ROOT / config_path).resolve()

    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config YAML must be a mapping/dict. Got: {type(loaded)}")
        cfg = _deep_merge(cfg, loaded)

    _apply_env_overrides(cfg)
    _resolve_paths_in_config(cfg)

    return cfg


def ensure_directories(cfg: Dict[str, Any]) -> None:
    """Create required directories if they don't exist."""
    paths = cfg["paths"]
    Path(paths["data_normal_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["data_anomalies_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["logs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["state_mode_file"]).parent.mkdir(parents=True, exist_ok=True)


def read_mode_file(mode_file: str | Path) -> str:
    """
    Read state/mode.txt value in {CAPTURE, MONITOR, OFF}.
    Missing/invalid => OFF.
    """
    p = Path(mode_file)
    try:
        mode = p.read_text(encoding="utf-8").strip().upper()
    except FileNotFoundError:
        return "OFF"
    if mode not in {"CAPTURE", "MONITOR", "OFF"}:
        return "OFF"
    return mode
