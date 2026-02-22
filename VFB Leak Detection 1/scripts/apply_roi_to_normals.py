"""
Apply a specific ROI crop to images in data/normal after capture and before training.

Use case:
- ROI was OFF during capture, but monitoring/training should use an ROI.
- You want ROI settings different from capture-time ROI.

Behavior:
- Reads config (for paths + current ROI values for comparison/reporting).
- Applies ROI SETTINGS DEFINED IN THIS SCRIPT by default.
- Optionally overwrite originals with backups, or write to a separate output dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Any, Dict, List, Tuple

import cv2

from src.config import ensure_directories, load_config
from src.data import apply_roi, list_image_files
from src.logging_utils import JsonlLogger


# ------------------------------------------------------------
# ROI SETTINGS USED BY THIS SCRIPT (authoritative crop values)
# ------------------------------------------------------------
SCRIPT_ROI: Dict[str, Any] = {
    "enabled": True,
    "units": "relative",  # "relative" or "pixels"
    "x": 0.0,
    "y": 0.0,
    "width": 0.8,
    "height": 1.0,
}


def _fmt_roi(roi: Dict[str, Any]) -> str:
    return (
        f"enabled={bool(roi.get('enabled', False))}, "
        f"units={roi.get('units', 'relative')}, "
        f"x={roi.get('x')}, y={roi.get('y')}, width={roi.get('width')}, height={roi.get('height')}"
    )


def _same_roi(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = ["enabled", "units", "x", "y", "width", "height"]
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if isinstance(av, (int, float)) or isinstance(bv, (int, float)):
            try:
                if abs(float(av) - float(bv)) > 1e-9:
                    return False
            except Exception:
                return False
        else:
            if str(av) != str(bv):
                return False
    return True


def _write_image(path: Path, img_bgr) -> None:
    ok = cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed writing image: {path}")


def _backup_and_overwrite(src_path: Path, cropped_bgr, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / src_path.name
    if not backup_path.exists():
        src_path.replace(backup_path)
    _write_image(src_path, cropped_bgr)


def _process_images(
    image_paths: List[Path],
    *,
    roi_cfg: Dict[str, Any],
    overwrite: bool,
    out_dir: Path,
    backup_dir: Path,
    dry_run: bool,
) -> Tuple[int, int, int]:
    processed = 0
    skipped = 0
    failed = 0

    if not overwrite:
        out_dir.mkdir(parents=True, exist_ok=True)

    for p in image_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            failed += 1
            continue

        cropped = apply_roi(bgr, roi_cfg)
        if cropped.shape[0] == bgr.shape[0] and cropped.shape[1] == bgr.shape[1]:
            skipped += 1

        if dry_run:
            processed += 1
            continue

        try:
            if overwrite:
                _backup_and_overwrite(p, cropped, backup_dir)
            else:
                out_path = out_dir / p.name
                _write_image(out_path, cropped)
            processed += 1
        except Exception:
            failed += 1

    return processed, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply script-defined ROI crop to normal images post-capture.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input_dir", type=str, default="", help="Input image directory (default: paths.data_normal_dir)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory when not overwriting (default: data/normal_roi_adjusted)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input images (creates backup first).")
    parser.add_argument(
        "--backup_dir",
        type=str,
        default="",
        help="Backup directory used with --overwrite (default: data/normal_backup_before_roi)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Preview processing without writing files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_directories(cfg)

    paths = cfg["paths"]
    logger = JsonlLogger(Path(paths["logs_dir"]) / "events.jsonl")
    logger.log("startup", component="apply_roi_to_normals")

    cfg_roi = dict(cfg.get("roi", {}) or {})
    script_roi = dict(SCRIPT_ROI)

    print("Config ROI:", _fmt_roi(cfg_roi))
    print("Script ROI:", _fmt_roi(script_roi))

    if _same_roi(cfg_roi, script_roi):
        print("ROI comparison: config and script ROI are identical.")
    else:
        print("ROI comparison: config and script ROI differ. Applying SCRIPT ROI values.")

    input_dir = Path(args.input_dir) if args.input_dir else Path(paths["data_normal_dir"])
    if not input_dir.is_absolute():
        input_dir = (PROJECT_ROOT / input_dir).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "data" / "normal_roi_adjusted")
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    backup_dir = Path(args.backup_dir) if args.backup_dir else (PROJECT_ROOT / "data" / "normal_backup_before_roi")
    if not backup_dir.is_absolute():
        backup_dir = (PROJECT_ROOT / backup_dir).resolve()

    image_paths = list_image_files(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in input_dir: {input_dir}")

    processed, skipped, failed = _process_images(
        image_paths,
        roi_cfg=script_roi,
        overwrite=bool(args.overwrite),
        out_dir=out_dir,
        backup_dir=backup_dir,
        dry_run=bool(args.dry_run),
    )

    logger.log(
        "roi_apply_complete",
        component="apply_roi_to_normals",
        input_dir=str(input_dir),
        output_dir=(None if args.overwrite else str(out_dir)),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        script_roi=script_roi,
        config_roi=cfg_roi,
        n_input=len(image_paths),
        n_processed=processed,
        n_same_shape=skipped,
        n_failed=failed,
    )

    print("ROI apply complete.")
    print(f"Input images: {len(image_paths)}")
    print(f"Processed: {processed}")
    print(f"Same-shape after crop (likely full-frame/no-op): {skipped}")
    print(f"Failed: {failed}")
    if args.overwrite:
        print(f"Overwrite mode: ON (backup dir: {backup_dir})")
    else:
        print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
