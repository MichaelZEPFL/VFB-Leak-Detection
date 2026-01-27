"""
System health check for the leak monitor.

This script is intended for non-technical users to quickly verify that:
- config loads
- required directories are writable
- (optional) camera can be opened and produces frames
- (optional) model artifacts exist and can be loaded
- notification settings look sane (Slack/email)
- disk space is sufficient

Examples:

  python .\scripts\health_check.py
  python .\scripts\health_check.py --save_snapshot
  python .\scripts\health_check.py --notify   # sends a test notification (Slack/email)

Outputs (optional):
- reports/health_check/camera_snapshot_<timestamp>.jpg
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
import shutil
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.camera import Camera, CameraConfig
from src.config import ensure_directories, load_config
from src.data import now_timestamp, save_bgr_image
from src.logging_utils import JsonlLogger
from src.monitor import make_notifier


def _print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return float(usage.free) / (1024 ** 3)


def _check_camera(cfg: Dict[str, Any], *, save_snapshot: bool, reports_dir: Path, logger: JsonlLogger) -> Tuple[bool, str]:
    cam_cfg = cfg["camera"]
    camera = Camera(CameraConfig(
        index=int(cam_cfg["index"]),
        backend=str(cam_cfg.get("backend", "DSHOW")),
        width=int(cam_cfg.get("width", 1280)),
        height=int(cam_cfg.get("height", 720)),
        fps=int(cam_cfg.get("fps", 30)),
    ))

    ok = False
    frame = None
    try:
        # Warm-up reads
        for _ in range(10):
            ok, frame = camera.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)

        if not ok or frame is None:
            return False, "Failed to read a frame from the camera."

        msg = f"Camera OK. Frame shape: {frame.shape}"

        if save_snapshot:
            reports_dir.mkdir(parents=True, exist_ok=True)
            p = save_bgr_image(frame, reports_dir, prefix="camera_snapshot", jpg_quality=95)
            msg += f" | Snapshot saved: {p.name}"

        return True, msg
    except Exception as e:
        logger.log("health_check_error", component="health_check", area="camera", error=str(e))
        return False, f"Camera check raised an exception: {e}"
    finally:
        camera.release()


def _check_model(cfg: Dict[str, Any], logger: JsonlLogger) -> Tuple[bool, str]:
    paths = cfg["paths"]
    model_dir = Path(paths["model_dir"])
    model_path = model_dir / "autoencoder.keras"
    thr_path = model_dir / "threshold.json"

    missing = []
    if not model_path.exists():
        missing.append(str(model_path))
    if not thr_path.exists():
        missing.append(str(thr_path))
    if missing:
        return False, "Missing artifacts: " + ", ".join(missing)

    try:
        _ = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.log("health_check_error", component="health_check", area="model_load", error=str(e))
        return False, f"Failed to load model: {e}"

    # Validate threshold JSON
    try:
        obj = json.loads(thr_path.read_text(encoding="utf-8"))
        thr = float(obj["threshold"])
        pct = float(obj.get("percentile", 99.5))
        if not np.isfinite(thr) or not np.isfinite(pct):
            return False, "threshold.json contains non-finite values."
    except Exception as e:
        logger.log("health_check_error", component="health_check", area="threshold", error=str(e))
        return False, f"Failed to parse threshold.json: {e}"

    return True, f"Model artifacts OK. Model: {model_path.name} | Threshold p{pct}: {thr:.6f}"


def _check_notifications(cfg: Dict[str, Any], *, send_test: bool, logger: JsonlLogger) -> Tuple[bool, str]:
    notifier = make_notifier(cfg)
    slack_cfg = cfg["notify"]["slack"]
    email_cfg = cfg["notify"]["email"]

    slack_enabled = bool(slack_cfg.get("enabled", True))
    slack_has_url = bool(str(slack_cfg.get("webhook_url", "") or "").strip())
    email_enabled = bool(email_cfg.get("enabled", False))

    parts = []
    parts.append(f"Slack enabled={slack_enabled}, configured={slack_has_url}")
    parts.append(f"Email enabled={email_enabled}")

    if email_enabled:
        # Basic sanity checks (no network operations by default).
        missing = []
        for k in ["smtp_host", "smtp_port", "from_addr"]:
            if not str(email_cfg.get(k, "") or "").strip():
                missing.append(k)
        if not email_cfg.get("to_addrs", []):
            missing.append("to_addrs")
        if missing:
            parts.append("Email config missing: " + ", ".join(missing))

    if send_test:
        try:
            notifier.notify("Leak monitor health check", "Test notification from health_check.py")
            parts.append("Test notification: sent (best-effort)")
        except Exception as e:
            logger.log("health_check_error", component="health_check", area="notify", error=str(e))
            parts.append(f"Test notification failed: {e}")
            return False, " | ".join(parts)

    # Consider notification check a pass if at least one channel is enabled/configured.
    ok = (slack_enabled and slack_has_url) or email_enabled
    if not ok:
        parts.append("No notification channel is configured (Slack webhook missing and email disabled).")

    return ok, " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Health check for the leak monitor system.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip_camera", action="store_true", help="Skip camera check.")
    parser.add_argument("--skip_model", action="store_true", help="Skip model artifact check.")
    parser.add_argument("--save_snapshot", action="store_true", help="Save a camera snapshot to reports/health_check/.")
    parser.add_argument("--notify", action="store_true", help="Send a test Slack/email notification (best-effort).")

    parser.add_argument("--min_free_gb", type=float, default=2.0, help="Minimum free disk space (GB) required.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    paths = cfg["paths"]
    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="health_check")

    ok_all = True

    _print_section("1) Paths + disk space")
    root = PROJECT_ROOT
    free_gb = _disk_free_gb(root)
    print(f"Project root: {root.resolve()}")
    print(f"Free disk space (on drive containing project): {free_gb:.2f} GB")
    if free_gb < float(args.min_free_gb):
        ok_all = False
        print(f"[FAIL] Low disk space: {free_gb:.2f} GB < {args.min_free_gb:.2f} GB")
    else:
        print("[OK] Disk space check")

    _print_section("2) Directory write check")
    try:
        test_file = Path(paths["logs_dir"]) / f"health_check_write_test_{now_timestamp()}.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        print("[OK] Can write to logs directory")
    except Exception as e:
        ok_all = False
        logger.log("health_check_error", component="health_check", area="filesystem", error=str(e))
        print(f"[FAIL] Filesystem write test failed: {e}")

    reports_dir = (PROJECT_ROOT / "reports" / "health_check").resolve()

    _print_section("3) Camera check")
    if args.skip_camera:
        print("[SKIP] Camera check")
    else:
        ok_cam, msg_cam = _check_camera(cfg, save_snapshot=bool(args.save_snapshot), reports_dir=reports_dir, logger=logger)
        print(("[OK] " if ok_cam else "[FAIL] ") + msg_cam)
        ok_all = ok_all and ok_cam

    _print_section("4) Model artifacts check")
    if args.skip_model:
        print("[SKIP] Model artifact check")
    else:
        ok_model, msg_model = _check_model(cfg, logger)
        print(("[OK] " if ok_model else "[FAIL] ") + msg_model)
        ok_all = ok_all and ok_model

    _print_section("5) Notifications check")
    ok_notif, msg_notif = _check_notifications(cfg, send_test=bool(args.notify), logger=logger)
    print(("[OK] " if ok_notif else "[WARN] ") + msg_notif)
    # Notifications are important, but do not necessarily fail the whole check if absent:
    # leave ok_all unchanged unless user explicitly requested --notify and it failed.
    if args.notify:
        ok_all = ok_all and ok_notif

    logger.log("health_check_complete", component="health_check", ok=ok_all)

    _print_section("RESULT")
    if ok_all:
        print("HEALTH CHECK PASSED")
        raise SystemExit(0)
    else:
        print("HEALTH CHECK FAILED (see messages above)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
