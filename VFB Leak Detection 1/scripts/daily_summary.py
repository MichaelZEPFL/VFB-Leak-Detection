"""
Generate a daily summary report from logs/events.jsonl.

This is useful for:
- quick morning review: did anything anomalous happen overnight?
- tracking false positives / drift over time

Outputs (default):
- reports/daily_summary/summary_<YYYY-MM-DD>.md
- reports/daily_summary/summary_<YYYY-MM-DD>.json
- (optional) reports/daily_summary/scores_<YYYY-MM-DD>.png

Examples:
  python .\scripts\daily_summary.py
  python .\scripts\daily_summary.py --date 2026-01-27
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
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.config import ensure_directories, load_config
from src.data import now_timestamp
from src.logging_utils import JsonlLogger


def _parse_iso_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _load_events(log_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    events: List[Dict[str, Any]] = []
    bad_lines = 0
    if not log_path.exists():
        return events, bad_lines

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
                else:
                    bad_lines += 1
            except json.JSONDecodeError:
                bad_lines += 1
    return events, bad_lines


def _filter_by_date(events: List[Dict[str, Any]], target: date) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in events:
        ts = e.get("ts")
        if not isinstance(ts, str):
            continue
        dt = _parse_iso_ts(ts)
        if dt is None:
            continue
        if dt.date() == target:
            out.append(e)
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _make_markdown(report: Dict[str, Any]) -> str:
    d = report["date"]
    lines: List[str] = []
    lines.append(f"# Daily Summary — {d}\n")
    lines.append(f"Generated: {report['generated_at']}\n")

    lines.append("## Overview\n")
    lines.append("| Metric | Value |\n|---|---:|\n")
    for k, v in report["overview"].items():
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n")

    lines.append("## Event counts\n")
    lines.append("| Event type | Count |\n|---|---:|\n")
    for k, v in sorted(report["counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n")

    if report.get("top_anomalies"):
        lines.append("## Top anomalies\n")
        lines.append("| Time | Score | Threshold | Event ID | Saved files |\n|---|---:|---:|---|---|\n")
        for a in report["top_anomalies"]:
            saved = ", ".join(a.get("saved_files", []))
            lines.append(f"| {a['time']} | {a['score']:.6f} | {a['threshold']:.6f} | {a.get('event_id','')} | {saved} |\n")
        lines.append("\n")

    if report.get("notes"):
        lines.append("## Notes\n")
        for n in report["notes"]:
            lines.append(f"- {n}\n")
        lines.append("\n")

    return "".join(lines)


def _plot_scores(score_points: List[Tuple[datetime, float]], threshold: Optional[float], out_path: Path) -> None:
    if not score_points:
        return
    score_points = sorted(score_points, key=lambda x: x[0])
    times = [p[0] for p in score_points]
    scores = [p[1] for p in score_points]

    plt.figure(figsize=(12, 4))
    plt.plot(times, scores)
    if threshold is not None:
        plt.axhline(threshold, linestyle="--", linewidth=2, label=f"threshold ~ {threshold:.6f}")
        plt.legend()
    plt.title("Reconstruction score over time (from frame_status heartbeats)")
    plt.xlabel("Time")
    plt.ylabel("Reconstruction error (MSE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a daily summary report from logs/events.jsonl")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--date", type=str, default="", help="Date to summarize (YYYY-MM-DD). Default: today.")
    parser.add_argument("--out_dir", type=str, default="reports/daily_summary", help="Output directory for reports.")
    parser.add_argument("--no_plot", action="store_true", help="Do not generate a score time-series plot.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top anomalies to include in the report.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)
    paths = cfg["paths"]

    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="daily_summary")

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except Exception:
            raise SystemExit("--date must be YYYY-MM-DD (example: 2026-01-27)")
    else:
        target_date = datetime.now().date()

    log_path = Path(paths["logs_dir"]) / "events.jsonl"
    events, bad_lines = _load_events(log_path)
    day_events = _filter_by_date(events, target_date)

    counts: Dict[str, int] = {}
    for e in day_events:
        t = str(e.get("type", "unknown"))
        counts[t] = counts.get(t, 0) + 1

    anomalies: List[Dict[str, Any]] = []
    score_points: List[Tuple[datetime, float]] = []
    thresholds: List[float] = []

    for e in day_events:
        et = str(e.get("type", ""))
        ts = e.get("ts")
        dt = _parse_iso_ts(ts) if isinstance(ts, str) else None

        if et == "anomaly_detected":
            score = _safe_float(e.get("score"))
            thr = _safe_float(e.get("threshold"))
            if score is None or thr is None or dt is None:
                continue
            saved_paths = e.get("saved_paths", [])
            saved_files = []
            if isinstance(saved_paths, list):
                for p in saved_paths:
                    try:
                        saved_files.append(Path(str(p)).name)
                    except Exception:
                        pass
            anomalies.append({
                "time": dt.strftime("%H:%M:%S"),
                "score": float(score),
                "threshold": float(thr),
                "event_id": str(e.get("event_id", "")),
                "saved_files": saved_files,
            })

        if et == "frame_status" and dt is not None:
            score = _safe_float(e.get("score"))
            thr = _safe_float(e.get("threshold"))
            if score is not None:
                score_points.append((dt, float(score)))
            if thr is not None:
                thresholds.append(float(thr))

    # Sort anomalies by score descending
    anomalies_sorted = sorted(anomalies, key=lambda a: a["score"], reverse=True)
    top_anoms = anomalies_sorted[: int(args.top_k)]

    approx_thr = float(np.median(thresholds)) if thresholds else (top_anoms[0]["threshold"] if top_anoms else None)

    notes: List[str] = []
    if bad_lines > 0:
        notes.append(f"Skipped {bad_lines} malformed log lines.")
    if not day_events:
        notes.append("No events found for this date.")
    if counts.get("camera_error", 0) > 0:
        notes.append("Camera errors were logged today. If monitoring was active, check USB connection and camera driver.")
    if counts.get("startup_error", 0) > 0:
        notes.append("Startup errors occurred. Check that model artifacts exist and config paths are correct.")

    report = {
        "date": str(target_date),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overview": {
            "Total events": len(day_events),
            "Anomalies detected": counts.get("anomaly_detected", 0),
            "Camera errors": counts.get("camera_error", 0),
            "Notify errors": counts.get("notify_error", 0),
            "Heartbeat frame_status events": counts.get("frame_status", 0),
        },
        "counts": counts,
        "top_anomalies": top_anoms,
        "notes": notes,
        "log_path": str(log_path.resolve()),
    }

    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"summary_{target_date}.json"
    md_path = out_dir / f"summary_{target_date}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_make_markdown(report), encoding="utf-8")

    plot_path = None
    if not args.no_plot and score_points:
        plot_path = out_dir / f"scores_{target_date}.png"
        _plot_scores(score_points, approx_thr, plot_path)

    logger.log(
        "daily_summary_complete",
        component="daily_summary",
        date=str(target_date),
        out_dir=str(out_dir.resolve()),
        json=str(json_path.resolve()),
        md=str(md_path.resolve()),
        plot=(str(plot_path.resolve()) if plot_path else None),
    )

    print("Daily summary generated.")
    print(f"Date: {target_date}")
    print(f"Markdown: {md_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    if plot_path:
        print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
