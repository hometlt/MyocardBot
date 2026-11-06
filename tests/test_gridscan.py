"""Tests for the standalone full-resolution grid scanner.

The cases mirror what actually turns up on an ECG upload: a printed lattice of
thin rulings, a lattice of dots, a sheet ruled in one direction only, a mesh
rotated against the page, and plain paper with no grid at all.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from miocard.pipeline.gridscan import GridScanConfig, iter_tiles, scan_grid

EXAMPLE_IMAGES = tuple((Path(__file__).resolve().parents[1] / "examples").glob("*.jpg"))
TRUE_PITCH = 20


def _trace(image: np.ndarray) -> None:
    """A QRS-like waveform, so the grid is never the only ink on the sheet."""
    height, width = image.shape[:2]
    points = [
        (x, int(height / 2 + 40 * np.sin(x / 25) + (70 if x % 160 < 4 else 0)))
        for x in range(20, width - 20)
    ]
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, 20, 2)


def make_sheet(
    kind: str,
    width: int = 1200,
    height: int = 900,
    pitch: int = TRUE_PITCH,
    angle: float = 0.0,
) -> np.ndarray:
    image = np.full((height, width), 250, dtype=np.uint8)
    if kind == "dots":
        for x in range(0, width, pitch):
            for y in range(0, height, pitch):
                cv2.circle(image, (x, y), 1, 215, -1)
    if kind in {"lines", "lines_x"}:
        for x in range(0, width, pitch):
            cv2.line(image, (x, 0), (x, height), 225, 1)
    if kind == "lines":
        for y in range(0, height, pitch):
            cv2.line(image, (0, y), (width, y), 225, 1)
    _trace(image)
    if angle:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=250)
    return image


@pytest.mark.parametrize("kind", ["lines", "dots"])
def test_lattice_pitch_is_measured_for_lines_and_dots(kind: str) -> None:
    """A dotted grid and a ruled grid must both give the same pitch."""
    scan = scan_grid(make_sheet(kind))

    assert scan.present
    assert scan.kind == "lattice"
    assert scan.px_per_mm == pytest.approx(TRUE_PITCH, abs=0.5)
    assert scan.period_x_px == pytest.approx(TRUE_PITCH, abs=0.5)
    assert scan.period_y_px == pytest.approx(TRUE_PITCH, abs=0.5)
    assert scan.confidence == "high"


def test_sheet_ruled_in_one_direction_is_not_called_a_lattice() -> None:
    scan = scan_grid(make_sheet("lines_x"))

    assert scan.present
    assert scan.kind == "lines_x"
    assert scan.period_x_px == pytest.approx(TRUE_PITCH, abs=0.5)
    assert scan.period_y_px == 0.0
    # Without a second axis there is no square millimetre cell to report.
    assert scan.px_per_mm == 0.0


def test_rotated_grid_is_measured_and_its_angle_reported() -> None:
    scan = scan_grid(make_sheet("lines", angle=5.0))

    assert scan.present
    assert scan.kind == "lattice"
    assert scan.px_per_mm == pytest.approx(TRUE_PITCH, abs=0.6)
    assert abs(scan.angle_deg) == pytest.approx(5.0, abs=1.5)


def test_plain_paper_reports_no_grid() -> None:
    """A trace on unruled paper must not produce an invented pitch."""
    scan = scan_grid(make_sheet("none"))

    assert not scan.present
    assert scan.px_per_mm == 0.0
    assert scan.kind == "absent"
    assert scan.note


def test_rotation_search_can_be_disabled() -> None:
    config = GridScanConfig(rotation_search_deg=0.0)
    scan = scan_grid(make_sheet("lines"), config)

    assert scan.present
    assert scan.px_per_mm == pytest.approx(TRUE_PITCH, abs=0.5)
    assert scan.angle_deg == 0.0


def test_large_image_is_split_into_overlapping_tiles() -> None:
    config = GridScanConfig(tile_px=256, overlap=0.25)
    rectangles = list(iter_tiles(2000, 3000, config))

    assert len(rectangles) > 1
    # Every tile fits inside the image.
    for left, top, width, height in rectangles:
        assert left >= 0 and top >= 0
        assert left + width <= 3000
        assert top + height <= 2000
    # The tiles reach both far edges.
    assert max(left + width for left, _t, width, _h in rectangles) == 3000
    assert max(top + height for _l, top, _w, height in rectangles) == 2000


def test_tiles_report_where_the_grid_was_visible() -> None:
    """Half a sheet ruled, half blank: coverage must reflect that."""
    image = make_sheet("lines", width=1600, height=800)
    image[:, 800:] = 250  # wipe the right half, grid and all
    scan = scan_grid(image, GridScanConfig(tile_px=384))

    assert scan.present
    assert scan.px_per_mm == pytest.approx(TRUE_PITCH, abs=0.5)
    assert any(tile.kind == "lattice" for tile in scan.tiles)
    # The wiped half is reported as blank, not as "ink but no grid", and it does
    # not count against the coverage of the half that was printed.
    assert scan.blank_tiles > 0
    assert any(tile.kind == "blank" for tile in scan.tiles)
    assert scan.coverage == pytest.approx(1.0)


@pytest.mark.skipif(not EXAMPLE_IMAGES, reason="no example photographs available")
def test_example_photographs_all_yield_a_plausible_millimetre_grid() -> None:
    """Real sheets print 1 mm cells; at these resolutions that is 4-6 px."""
    for path in EXAMPLE_IMAGES:
        image = cv2.imread(str(path))
        assert image is not None
        scan = scan_grid(image)

        assert scan.present, f"{path.name}: no grid found"
        assert scan.kind == "lattice", f"{path.name}: {scan.kind}"
        assert 3.5 <= scan.px_per_mm <= 7.0, f"{path.name}: {scan.px_per_mm:.2f} px/mm"
        assert scan.coverage >= 0.5
        assert scan.confidence in {"high", "medium"}


@pytest.mark.skipif(not EXAMPLE_IMAGES, reason="no example photographs available")
def test_scanner_agrees_with_the_pipeline_grid_detector() -> None:
    """Two independent methods on the same sheet should land on the same pitch."""
    from miocard.pipeline.grid import measure_grid

    for path in EXAMPLE_IMAGES:
        image = cv2.imread(str(path))
        scan = scan_grid(image)
        reference = measure_grid(image)
        assert reference is not None, f"{path.name}: pipeline found no grid"
        assert scan.px_per_mm == pytest.approx(reference.small_cell_px, rel=0.10), (
            f"{path.name}: scanner {scan.px_per_mm:.2f} vs pipeline {reference.small_cell_px:.2f}"
        )
