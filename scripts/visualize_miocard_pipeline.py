#!/usr/bin/env python

"""Create visual diagnostics for the MyocardBot2 reconstruction pipeline.

Run it over a folder of ECG photographs to get, per image: an overview with the
stage confidences, the reconstruction overlay (the quickest way to see whether
the solver tracked the printed traces), an optional per-stage board and the
per-class probability maps.

    python scripts/visualize_miocard_pipeline.py --show-stages --show-probability-maps
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from miocard.imaging import ProcessedImage, process_ecg_image  # noqa: E402
from miocard.pipeline import to_json  # noqa: E402
from miocard.stage_render import (  # noqa: E402
    add_label,
    draw_text_block,
    render_all_stages,
    render_probability_maps_board,
    render_reconstruction_overlay,
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"})


def _decode(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Unable to decode source image.")
    return image


def _fit_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / height
    return cv2.resize(
        image, (max(1, int(width * scale)), target_height), interpolation=cv2.INTER_AREA
    )


def _collect_input_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _stage_lines(result: ProcessedImage, source_name: str, end_stage: int) -> list[str]:
    pipeline = result.pipeline
    lines = [
        f"Source: {source_name}",
        f"Output: {result.width}x{result.height} | visual tokens: {result.estimated_visual_tokens}",
        f"Blur: {result.blur_score:.1f}  Brightness: {result.brightness:.1f}  "
        f"Contrast: {result.contrast:.1f}",
    ]
    if pipeline is None:
        return [*lines, "pipeline unavailable"]

    grid = result.grid
    beats = pipeline.beats
    lines += [
        f"Grid: {grid.small_cell_px:.2f} px/mm ({grid.confidence})" if grid else "Grid: none",
        f"Layout: {pipeline.layout.get('row_count', 0)} rows, "
        f"pitch {float(pipeline.layout.get('row_pitch_px', 0.0)):.1f} px",
        f"Rhythm: {beats.heart_rate_bpm:.0f} bpm over {beats.count} beats "
        f"(regularity {beats.regularity:.2f})"
        if beats
        else "Rhythm: not measured",
        f"Quality: {pipeline.quality_grade} "
        f"(p={float(pipeline.quality.get('probability', 0.0)):.2f})",
        "",
        f"Pipeline stages 1..{end_stage}:",
    ]
    for index, stage in enumerate(pipeline.stages[:end_stage], start=1):
        evidence = ", ".join(stage.evidence)
        line = f"  {index:02d}. {stage.stage}: {stage.confidence} (p={stage.probability:.2f})"
        if evidence:
            line += f" | {evidence}"
        lines.append(line)
    return lines


def _write(path: Path, image: np.ndarray) -> Path:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")
    return path


def _visualize_image(
    input_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
    index: int,
) -> list[Path]:
    raw = input_path.read_bytes()
    result = process_ecg_image(raw, args.model, args.max_bytes)

    source = add_label(_decode(raw), f"Original ({input_path.name})")
    processed_raw = _decode(result.data)
    processed = add_label(processed_raw, "Processed by MyocardBot2")

    target_height = max(source.shape[0], processed.shape[0])
    source = _fit_to_height(source, target_height)
    processed = _fit_to_height(processed, target_height)
    gap = np.full((target_height, 12, 3), 220, dtype=np.uint8)
    body = np.hstack((source, gap, processed))

    lines = _stage_lines(result, input_path.name, args.end_stage)
    overview = np.vstack((body, draw_text_block(lines, body.shape[1], 24 + 16 * len(lines))))

    stem = f"{index:03d}_{input_path.stem}"
    saved = [_write(out_dir / f"{stem}_overview.png", overview)]
    saved.append(
        _write(
            out_dir / f"{stem}_reconstruction.png",
            render_reconstruction_overlay(processed_raw, result),
        )
    )

    if args.show_stages:
        board = render_all_stages(
            source,
            processed,
            result,
            card_width=max(body.shape[1], 1200),
            stage_start=max(1, args.start_stage),
            stage_end=max(1, args.end_stage),
        )
        if board.size == 0:
            raise RuntimeError("render_all_stages returned an empty board.")
        saved.append(
            _write(out_dir / f"{stem}_stages_{args.start_stage}_{args.end_stage}.png", board)
        )

    if args.show_probability_maps:
        for name, panel in render_probability_maps_board(processed_raw, result):
            saved.append(_write(out_dir / f"{stem}_p_{name}.png", panel))

    if args.export_json and result.pipeline is not None:
        payload = out_dir / f"{stem}_export.json"
        payload.write_text(to_json(result.pipeline, indent=2), encoding="utf-8")
        saved.append(payload)

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MyocardBot2 visualization images.")
    parser.add_argument(
        "--input",
        default=str(ROOT_DIR / "examples"),
        help="Path to one image or a directory of images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Defaults to tmp-ecg-visualizations/<timestamp>.",
    )
    parser.add_argument(
        "--model", default="gpt-5.6-sol", help="Model id used for image profile selection."
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=20 * 1024 * 1024,
        help="Maximum accepted input size in bytes.",
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Optional max number of files to process."
    )
    parser.add_argument(
        "--show-stages",
        action="store_true",
        help="Generate the per-stage board in addition to the overview.",
    )
    parser.add_argument(
        "--show-probability-maps",
        action="store_true",
        help="Render one panel per cost-map class.",
    )
    parser.add_argument(
        "--export-json", action="store_true", help="Write the stage 15 export payload."
    )
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else ROOT_DIR / "tmp-ecg-visualizations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_input_files(Path(args.input).expanduser())
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    saved: list[Path] = []
    for index, image_path in enumerate(files, start=1):
        for path in _visualize_image(image_path, args, output_dir, index):
            print(f"saved: {path}")
            saved.append(path)

    if not saved:
        print("No input images matched. No files were written.")
        return 2
    print(f"Total files: {len(saved)}")
    print(f"Saved in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
