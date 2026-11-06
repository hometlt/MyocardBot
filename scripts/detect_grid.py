#!/usr/bin/env python

"""Detect the millimetre grid on ECG images, at their original resolution.

Reports, per file: whether a grid is there at all, its pitch in pixels and the
implied pixels-per-millimetre, its rotation, and how much of the sheet it was
visible on. Writes a JSON report and, optionally, an overlay showing the fitted
lattice and the per-tile result.

    python scripts/detect_grid.py --input examples --overlay

Nothing here downsamples the picture: the grid on a phone photo is only a few
pixels wide, and the pipeline's model-facing resize costs about half the linear
resolution, which is most of the signal. Large images are split into overlapping
tiles instead, so memory stays bounded and the pitch is measured locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from miocard.pipeline.gridscan import (  # noqa: E402
    GridScan,
    GridScanConfig,
    scan_grid,
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"})


def _collect(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _read(path: Path) -> np.ndarray:
    # imdecode via numpy so non-ASCII paths work on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def _report(path: Path, scan: GridScan) -> dict[str, object]:
    return {
        "file": path.name,
        "image": {"width": scan.image_size[1], "height": scan.image_size[0]},
        "present": scan.present,
        "kind": scan.kind,
        "confidence": scan.confidence,
        "probability": round(scan.probability, 3),
        "period_x_px": round(scan.period_x_px, 3),
        "period_y_px": round(scan.period_y_px, 3),
        "px_per_mm": round(scan.px_per_mm, 3),
        "large_cell_px": round(scan.px_per_mm * 5.0, 2) if scan.px_per_mm else 0.0,
        "angle_deg": round(scan.angle_deg, 3),
        "phase_x_px": round(scan.phase_x_px, 3),
        "phase_y_px": round(scan.phase_y_px, 3),
        "blank_tiles": scan.blank_tiles,
        "coverage": round(scan.coverage, 3),
        "consistency": round(scan.consistency, 3),
        "tiles": len(scan.tiles),
        "note": scan.note,
        "tile_detail": [
            {
                "left": tile.left,
                "top": tile.top,
                "width": tile.width,
                "height": tile.height,
                "kind": tile.kind,
                "period_x_px": round(tile.period_x_px, 3),
                "period_y_px": round(tile.period_y_px, 3),
                "angle_deg": round(tile.angle_deg, 3),
                "strength": round(tile.strength, 4),
            }
            for tile in scan.tiles
        ],
    }


def _overlay(image: np.ndarray, scan: GridScan, max_width: int = 1600) -> np.ndarray:
    """Draw the fitted lattice and the per-tile verdict over the sheet."""
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for tile in scan.tiles:
        colour = {
            "lattice": (60, 200, 60),
            "lines_x": (0, 170, 255),
            "lines_y": (255, 170, 0),
        }.get(tile.kind, (60, 60, 220))
        cv2.rectangle(
            canvas,
            (tile.left, tile.top),
            (tile.left + tile.width - 1, tile.top + tile.height - 1),
            colour,
            2,
        )

    if scan.present and scan.px_per_mm > 1:
        # Only the 5 mm block lines are drawn. At a millimetre pitch the overlay
        # would cover the sheet completely and you could not see whether it
        # landed on the print, which is the only thing it is for.
        overlay = canvas.copy()
        step_x = (scan.period_x_px or scan.px_per_mm) * 5.0
        step_y = (scan.period_y_px or scan.px_per_mm) * 5.0
        x = scan.phase_x_px
        while x < width and step_x > 0:
            cv2.line(overlay, (int(round(x)), 0), (int(round(x)), height), (200, 80, 255), 1)
            x += step_x
        y = scan.phase_y_px
        while y < height and step_y > 0:
            cv2.line(overlay, (0, int(round(y))), (width, int(round(y))), (200, 80, 255), 1)
            y += step_y
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    banner = np.full((64, width, 3), 25, dtype=np.uint8)
    cv2.putText(
        banner,
        scan.summary(),
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, width / 2400),
        (140, 255, 140) if scan.present else (120, 120, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "overlay = fitted 5 mm blocks | tiles: green lattice, orange rulings, red none, blue blank",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, width / 3200),
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    canvas = np.vstack((banner, canvas))

    if canvas.shape[1] > max_width:
        scale = max_width / canvas.shape[1]
        canvas = cv2.resize(
            canvas,
            (max_width, int(canvas.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return canvas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT_DIR / "examples"))
    parser.add_argument("--output-dir", default=None, help="Where to write report and overlays.")
    parser.add_argument("--tile-px", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-period-px", type=float, default=3.0)
    parser.add_argument("--max-period-px", type=float, default=100.0)
    parser.add_argument("--max-tiles", type=int, default=400)
    parser.add_argument(
        "--rotation-search-deg",
        type=float,
        default=8.0,
        help="How far off square the mesh may be; 0 disables the per-tile rotation search.",
    )
    parser.add_argument("--overlay", action="store_true", help="Write an annotated image per file.")
    parser.add_argument("--json", default=None, help="Path for the JSON report.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = GridScanConfig(
        tile_px=args.tile_px,
        overlap=args.overlap,
        min_period_px=args.min_period_px,
        max_period_px=args.max_period_px,
        max_tiles=args.max_tiles,
        rotation_search_deg=args.rotation_search_deg,
    )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else ROOT_DIR / "tmp-grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for path in _collect(Path(args.input).expanduser()):
        image = _read(path)
        scan = scan_grid(image, config)
        reports.append(_report(path, scan))
        print(f"{path.name}: {scan.summary()}")
        if args.overlay:
            target = output_dir / f"{path.stem}_grid.png"
            cv2.imwrite(str(target), _overlay(image, scan), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"    overlay -> {target}")

    if not reports:
        print("No images matched.")
        return 2

    destination = Path(args.json) if args.json else output_dir / "grid_report.json"
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nReport -> {destination}")
    found = sum(1 for item in reports if item["present"])
    print(f"Grid found on {found}/{len(reports)} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
