"""
Monitoring loop for anomaly detection with debounce + cooldown.

Reads state/mode.txt:
  - only runs active monitoring when mode == MONITOR

Features:
- Debounce: require N consecutive anomalous frames
- Cooldown: suppress repeated alerts for a time window
- Optional context frames (pre/post) saved alongside anomaly
- Camera down detection + alert
- Save reconstruction + error heatmap for detected anomalies (helps debugging false positives)
- Optional second trigger from spatially localized heatmap blobs
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from .camera import Camera, CameraConfig
from .config import read_mode_file
from .data import apply_roi, now_timestamp, save_bgr_image
from .logging_utils import JsonlLogger
from .model import reconstruction_error_map, score_from_reconstruction
from .notify import EmailConfig, Notifier, SlackConfig


def _load_threshold(model_dir: Path) -> Tuple[float, float, Dict[str, Any]]:
    """Load threshold.json.

    Returns:
        (threshold, percentile, extras)
    """
    p = model_dir / "threshold.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {p}. "
            "Run training first to generate model/threshold.json."
        )

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Threshold file is not valid JSON: {p}") from e

    if "threshold" not in obj:
        raise KeyError(f"Missing required key 'threshold' in: {p}")

    try:
        threshold = float(obj["threshold"])
        percentile = float(obj.get("percentile", 99.5))
    except Exception as e:
        raise ValueError(f"Invalid threshold.json values in: {p}") from e

    if not np.isfinite(threshold) or not np.isfinite(percentile):
        raise ValueError(f"Non-finite threshold.json values in: {p}")

    extras = {
        "heatmap_trigger_enabled": bool(obj.get("heatmap_trigger_enabled", False)),
        "pixel_error_threshold": obj.get("pixel_error_threshold"),
        "blob_area_threshold": obj.get("blob_area_threshold"),
        "pixel_error_percentile": obj.get("pixel_error_percentile"),
        "blob_area_percentile": obj.get("blob_area_percentile"),
        "connectivity": obj.get("connectivity"),
        "min_blob_area": obj.get("min_blob_area"),
    }

    return threshold, percentile, extras


def _preprocess_frame_bgr(frame_bgr: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    """Convert BGR -> RGB, resize, normalize to [0,1], return float32 [1,H,W,3]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_w, image_h), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def _rgb01_to_bgr_uint8(rgb01: np.ndarray) -> np.ndarray:
    """Convert an RGB float image in [0,1] to BGR uint8 for OpenCV writing."""
    rgb_u8 = np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def _make_error_heatmap_bgr(x_rgb01: np.ndarray, xhat_rgb01: np.ndarray) -> np.ndarray:
    """Create a BGR heatmap from per-pixel squared error (averaged across channels)."""
    # [H,W] float32
    err = np.mean(np.square(x_rgb01 - xhat_rgb01), axis=2)
    # Normalize for visualization.
    p99 = float(np.percentile(err, 99.0))
    denom = (p99 if p99 > 1e-12 else float(err.max())) + 1e-12
    err_norm = np.clip(err / denom, 0.0, 1.0)
    # Convert to colormap.
    err_u8 = np.clip(err_norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(err_u8, cv2.COLORMAP_JET)


def _largest_blob_area(err_map: np.ndarray, *, pixel_error_threshold: float, connectivity: int) -> int:
    mask = (err_map > float(pixel_error_threshold)).astype(np.uint8)
    conn = 8 if int(connectivity) == 8 else 4
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=conn)
    if n_labels <= 1:
        return 0
    return int(np.max(stats[1:, cv2.CC_STAT_AREA]))


def make_notifier(cfg: Dict[str, Any]) -> Notifier:
    slack_cfg = cfg["notify"]["slack"]
    email_cfg = cfg["notify"]["email"]

    slack = SlackConfig(
        enabled=bool(slack_cfg.get("enabled", True)),
        webhook_url=str(slack_cfg.get("webhook_url", "") or "").strip(),
    )

    email = EmailConfig(
        enabled=bool(email_cfg.get("enabled", False)),
        smtp_host=str(email_cfg.get("smtp_host", "")),
        smtp_port=int(email_cfg.get("smtp_port", 587)),
        smtp_user=str(email_cfg.get("smtp_user", "")),
        smtp_password=str(email_cfg.get("smtp_password", "")),
        from_addr=str(email_cfg.get("from_addr", "")),
        to_addrs=list(email_cfg.get("to_addrs", [])),
        use_tls=bool(email_cfg.get("use_tls", True)),
    )

    return Notifier(slack=slack, email=email)


def _safe_notify(notifier: Notifier, logger: JsonlLogger, title: str, message: str) -> None:
    try:
        notifier.notify(title, message)
    except Exception as e:
        logger.log("notify_error", error=str(e), title=title)


def run_monitor(cfg: Dict[str, Any]) -> None:
    paths = cfg["paths"]
    img_cfg = cfg["image"]
    mon_cfg = cfg["monitor"]
    cam_cfg = cfg["camera"]
    roi_cfg = dict(cfg.get("roi", {}) or {})
    scoring_cfg = dict(cfg.get("scoring", {}) or {})
    heatmap_cfg = dict(cfg.get("heatmap_trigger", {}) or {})

    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    notifier = make_notifier(cfg)

    model_dir = Path(paths["model_dir"])
    model_path = model_dir / "autoencoder.keras"
    threshold_path = model_dir / "threshold.json"

    missing = []
    if not model_path.exists():
        missing.append({"artifact": "model", "path": str(model_path.resolve())})
    if not threshold_path.exists():
        missing.append({"artifact": "threshold", "path": str(threshold_path.resolve())})
    if missing:
        msg_lines = [
            "Leak monitor startup failed: required artifacts are missing.",
            "Run training first (scripts/train_autoencoder.py) to generate these files:",
        ]
        msg_lines.extend([f"- {m['artifact']}: {m['path']}" for m in missing])
        msg = "\n".join(msg_lines)

        logger.log(
            "startup_error",
            component="monitor",
            reason="missing_artifact",
            missing=missing,
        )
        _safe_notify(notifier, logger, "Leak monitor: startup failed", msg)
        raise SystemExit(msg)

    try:
        threshold, percentile, threshold_extras = _load_threshold(model_dir)
    except Exception as e:
        msg = (
            "Leak monitor startup failed: could not load threshold artifact.\n"
            f"- Path: {threshold_path.resolve()}\n"
            f"- Error: {e}"
        )
        logger.log(
            "startup_error",
            component="monitor",
            reason="artifact_error",
            artifact="threshold",
            path=str(threshold_path.resolve()),
            error=str(e),
        )
        _safe_notify(notifier, logger, "Leak monitor: startup failed", msg)
        raise SystemExit(msg)
    # Load TF model once.
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        msg = (
            "Leak monitor startup failed: could not load model artifact.\n"
            f"- Path: {model_path.resolve()}\n"
            f"- Error: {e}"
        )
        logger.log(
            "startup_error",
            component="monitor",
            reason="artifact_error",
            artifact="model_load",
            path=str(model_path.resolve()),
            error=str(e),
        )
        _safe_notify(notifier, logger, "Leak monitor: startup failed", msg)
        raise SystemExit(msg)

    heatmap_enabled_cfg = bool(heatmap_cfg.get("enabled", True))
    heatmap_enabled_model = bool(threshold_extras.get("heatmap_trigger_enabled", False))

    pixel_error_threshold = threshold_extras.get("pixel_error_threshold")
    blob_area_threshold = threshold_extras.get("blob_area_threshold")
    connectivity = int(heatmap_cfg.get("connectivity", threshold_extras.get("connectivity", 8)))
    min_blob_area = float(heatmap_cfg.get("min_blob_area", threshold_extras.get("min_blob_area", 16)))

    heatmap_ready = (
        heatmap_enabled_cfg
        and heatmap_enabled_model
        and pixel_error_threshold is not None
        and blob_area_threshold is not None
    )

    if heatmap_enabled_cfg and not heatmap_ready:
        logger.log(
            "startup_warning",
            component="monitor",
            reason="heatmap_trigger_disabled",
            message="Heatmap trigger requested but thresholds were not found in model/threshold.json. Retrain to enable.",
        )

    camera = Camera(
        CameraConfig(
            index=int(cam_cfg["index"]),
            backend=str(cam_cfg.get("backend", "DSHOW")),
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
            fps=int(cam_cfg.get("fps", 30)),
        )
    )

    mode_file = paths["state_mode_file"]
    anomalies_dir = Path(paths["data_anomalies_dir"])
    anomalies_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = float(mon_cfg.get("frame_interval_seconds", 1.0))
    consecutive_needed = int(mon_cfg.get("consecutive_anomalies", 3))
    cooldown_s = float(mon_cfg.get("alert_cooldown_seconds", 300))
    heartbeat_s = float(mon_cfg.get("heartbeat_seconds", 60))
    camera_down_s = float(mon_cfg.get("camera_down_seconds", 30))

    # Reconstruction/diagnostic image saving (defaults ON if omitted)
    #
    # Backward/forward compatible parsing:
    # - If monitor.save_reconstruction is a bool: treat it as on/off.
    # - If monitor.save_reconstruction is a dict: read {enabled, heatmap}.
    recon_cfg = mon_cfg.get("save_reconstruction", True)
    if isinstance(recon_cfg, dict):
        save_recon = bool(recon_cfg.get("enabled", True))
        save_heatmap = bool(recon_cfg.get("heatmap", True))
    else:
        save_recon = bool(recon_cfg)
        save_heatmap = bool(mon_cfg.get("save_error_heatmap", True))

    save_input_resized = bool(mon_cfg.get("save_input_resized", True))
    save_compare = bool(mon_cfg.get("save_compare_image", True))

    save_ctx = dict(mon_cfg.get("save_context", {}) or {})
    ctx_enabled = bool(save_ctx.get("enabled", False))
    ctx_pre = int(save_ctx.get("pre_frames", 1))
    ctx_post = int(save_ctx.get("post_frames", 1))
    ctx_buffer: Deque[np.ndarray] = deque(maxlen=max(0, ctx_pre))

    last_alert_ts = 0.0
    last_heartbeat_ts = 0.0
    last_ok_frame_ts = time.time()
    camera_down_alert_sent = False

    consecutive_anomalies = 0
    last_mode: Optional[str] = None

    logger.log(
        "startup",
        component="monitor",
        threshold=threshold,
        percentile=percentile,
        scoring_method=str(scoring_cfg.get("method", "mean")),
        topk_percent=float(scoring_cfg.get("topk_percent", 1.0)),
        topk_min_pixels=int(scoring_cfg.get("topk_min_pixels", 100)),
        heatmap_trigger_enabled=bool(heatmap_ready),
        pixel_error_threshold=float(pixel_error_threshold) if heatmap_ready else None,
        blob_area_threshold=float(blob_area_threshold) if heatmap_ready else None,
        connectivity=int(connectivity),
        min_blob_area=float(min_blob_area),
        roi=roi_cfg if roi_cfg else None,
        model_path=str(model_path.resolve()),
        threshold_path=str(threshold_path.resolve()),
    )

    try:
        while True:
            mode = read_mode_file(mode_file)

            if mode != last_mode:
                logger.log("mode_change", mode=mode)
                last_mode = mode

            if mode != "MONITOR":
                consecutive_anomalies = 0
                camera_down_alert_sent = False
                ctx_buffer.clear()
                time.sleep(1.0)
                continue

            ok, frame_bgr = camera.read()
            now = time.time()

            if not ok or frame_bgr is None:
                if now - last_ok_frame_ts > camera_down_s and not camera_down_alert_sent:
                    msg = f"Camera has not produced frames for {int(now - last_ok_frame_ts)}s."
                    logger.log("camera_error", message=msg)
                    _safe_notify(notifier, logger, "Leak monitor: camera feed down", msg)
                    camera_down_alert_sent = True
                time.sleep(1.0)
                continue

            last_ok_frame_ts = now
            camera_down_alert_sent = False

            frame_roi = apply_roi(frame_bgr, roi_cfg)

            if ctx_enabled and ctx_pre > 0:
                # Save a copy to avoid later mutation
                ctx_buffer.append(frame_roi.copy())

            x = _preprocess_frame_bgr(frame_roi, int(img_cfg["width"]), int(img_cfg["height"]))
            x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
            
            xhat_tf = model(x_tf, training=False)
            score_tf = score_from_reconstruction(
                x_tf,
                xhat_tf,
                method=str(scoring_cfg.get("method", "mean")),
                topk_percent=float(scoring_cfg.get("topk_percent", 1.0)),
                topk_min_pixels=int(scoring_cfg.get("topk_min_pixels", 100)),
            )
            score = float(score_tf.numpy()[0])
            score_triggered = score > threshold

            heatmap_triggered = False
            largest_blob_area = 0
            if heatmap_ready:
                err_map = reconstruction_error_map(x_tf, xhat_tf).numpy()[0]
                largest_blob_area = _largest_blob_area(
                    err_map,
                    pixel_error_threshold=float(pixel_error_threshold),
                    connectivity=connectivity,
                )
                heat_blob_thr = max(float(blob_area_threshold), float(min_blob_area))
                heatmap_triggered = float(largest_blob_area) >= heat_blob_thr

            is_anom = score_triggered or heatmap_triggered
            if is_anom:
                consecutive_anomalies += 1
            else:
                consecutive_anomalies = 0

            if now - last_heartbeat_ts >= heartbeat_s:
                cooldown_remaining_s = max(0.0, cooldown_s - (now - last_alert_ts))
                logger.log(
                    "frame_status",
                    mode=mode,
                    score=score,
                    threshold=threshold,
                    score_triggered=score_triggered,
                    heatmap_triggered=heatmap_triggered,
                    largest_blob_area=largest_blob_area if heatmap_ready else None,
                    blob_area_threshold=max(float(blob_area_threshold), float(min_blob_area)) if heatmap_ready else None,
                    is_anomaly=is_anom,
                    consecutive_anomalies=consecutive_anomalies,
                    consecutive_needed=consecutive_needed,
                    cooldown_seconds=cooldown_s,
                    cooldown_remaining_seconds=int(round(cooldown_remaining_s)),
                    frame_interval_seconds=frame_interval,
                )
                last_heartbeat_ts = now

            should_trigger = consecutive_anomalies >= consecutive_needed
            cooldown_ok = (now - last_alert_ts) >= cooldown_s

            if should_trigger and cooldown_ok:
                saved_paths = []
                event_id = now_timestamp()
                # Save pre-context frames first
                if ctx_enabled and ctx_pre > 0 and len(ctx_buffer) > 0:
                    for i, f in enumerate(list(ctx_buffer)[-ctx_pre:]):
                        p = save_bgr_image(
                            f,
                            anomalies_dir,
                            prefix="context_pre",
                            jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                            extra_suffix=f"i{i:02d}",
                            timestamp=event_id,
                        )
                        saved_paths.append(str(p.resolve()))

                extra_suffix = f"score{score:.6f}"
                saved_anom = save_bgr_image(
                    frame_roi,
                    anomalies_dir,
                    prefix="anomaly",
                    jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                    extra_suffix=extra_suffix,
                    timestamp=event_id,
                )
                saved_paths.append(str(saved_anom.resolve()))

                # Save reconstruction + error map at model resolution (helps debugging)
                if save_recon or save_heatmap or save_compare:
                    try:
                        xhat = xhat_tf.numpy()[0]
                        xin = x[0]

                        xin_bgr = _rgb01_to_bgr_uint8(xin)
                        if save_input_resized:
                            p_in = save_bgr_image(
                                xin_bgr,
                                anomalies_dir,
                                prefix="input_resized",
                                jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                                extra_suffix=extra_suffix,
                                timestamp=event_id,
                            )
                            saved_paths.append(str(p_in.resolve()))

                        recon_bgr: Optional[np.ndarray] = None
                        heat_bgr: Optional[np.ndarray] = None

                        if save_recon:
                            recon_bgr = _rgb01_to_bgr_uint8(xhat)
                            p_recon = save_bgr_image(
                                recon_bgr,
                                anomalies_dir,
                                prefix="reconstruction",
                                jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                                extra_suffix=extra_suffix,
                                timestamp=event_id,
                            )
                            saved_paths.append(str(p_recon.resolve()))

                        if save_heatmap or save_compare:
                            heat_bgr = _make_error_heatmap_bgr(xin, xhat)

                        if save_heatmap and heat_bgr is not None:
                            p_heat = save_bgr_image(
                                heat_bgr,
                                anomalies_dir,
                                prefix="error_heatmap",
                                jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                                extra_suffix=extra_suffix,
                                timestamp=event_id,
                            )
                            saved_paths.append(str(p_heat.resolve()))

                        if save_compare:
                            panels = [xin_bgr]
                            if recon_bgr is not None:
                                panels.append(recon_bgr)
                            if heat_bgr is not None:
                                panels.append(heat_bgr)
                            
                            # Guard against unexpected shape mismatches.
                            heights = {p.shape[0] for p in panels}
                            if len(heights) == 1:
                                compare_bgr = np.concatenate(panels, axis=1)
                                p_cmp = save_bgr_image(
                                    compare_bgr,
                                    anomalies_dir,
                                    prefix="compare",
                                    jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                                    extra_suffix=extra_suffix,
                                    timestamp=event_id,
                                )
                                saved_paths.append(str(p_cmp.resolve()))
                            else:
                                logger.log(
                                    "reconstruction_save_warning",
                                    message="Compare panels had mismatched heights; skipping compare image.",
                                )

                    except Exception as e:
                        logger.log("reconstruction_save_error", error=str(e))
                
                # Save post-context frames (best effort)
                if ctx_enabled and ctx_post > 0:
                    for i in range(ctx_post):
                        time.sleep(max(0.0, frame_interval))
                        ok2, frame2 = camera.read()
                        if not ok2 or frame2 is None:
                            break
                        frame2 = apply_roi(frame2, roi_cfg)
                        p = save_bgr_image(
                            frame2,
                            anomalies_dir,
                            prefix="context_post",
                            jpg_quality=int(cfg["capture"].get("jpg_quality", 95)),
                            extra_suffix=f"i{i:02d}",
                            timestamp=event_id,
                        )
                        saved_paths.append(str(p.resolve()))

                logger.log(
                    "anomaly_detected",
                    event_id=event_id,
                    score=score,
                    threshold=threshold,
                    score_triggered=score_triggered,
                    heatmap_triggered=heatmap_triggered,
                    largest_blob_area=largest_blob_area if heatmap_ready else None,
                    blob_area_threshold=max(float(blob_area_threshold), float(min_blob_area)) if heatmap_ready else None,
                    consecutive_anomalies=consecutive_anomalies,
                    saved_paths=saved_paths,
                )

                trigger_labels = []
                if score_triggered:
                    trigger_labels.append("score")
                if heatmap_triggered:
                    trigger_labels.append("heatmap")

                title = "Leak monitor: anomaly detected"
                message = (
                    f"Anomaly detected on camera.\n"
                    f"- Event ID: {event_id}\n"
                    f"- Trigger(s): {', '.join(trigger_labels) if trigger_labels else 'unknown'}\n"
                    f"- Score: {score:.6f}\n"
                    f"- Threshold (p{percentile}): {threshold:.6f}\n"
                    f"- Largest heatmap blob area: {largest_blob_area if heatmap_ready else 'n/a'}\n"
                    f"- Heatmap blob threshold: {max(float(blob_area_threshold), float(min_blob_area)) if heatmap_ready else 'n/a'}\n"
                    f"- Saved files: {', '.join(Path(p).name for p in saved_paths) if saved_paths else '(none)'}\n"
                    f"- Cooldown: {int(cooldown_s)}s\n"
                )

                try:
                    notifier.notify(title, message)
                except Exception as e:
                    logger.log("notify_error", error=str(e), title=title)

                last_alert_ts = now
                consecutive_anomalies = 0
                ctx_buffer.clear()

            time.sleep(frame_interval)

    except KeyboardInterrupt:
        logger.log("shutdown", component="monitor", reason="KeyboardInterrupt")
    finally:
        camera.release()
