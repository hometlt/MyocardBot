"""Flatten a photographed ECG sheet against its own millimetre grid.

A photograph of paper is never flat. The page curls, the camera is off-axis, and
the result is that the printed grid - which is regular by construction - arrives
as a mesh whose spacing drifts across the frame. Everything downstream pays for
that: the row pitch that fits the top of the sheet does not fit the bottom, and
intervals measured in millimetres are wrong by however much the local scale
differs from the global one.

The grid is also the fix. Because the true lattice is regular, any departure
from regularity in the image *is* the distortion, measured directly, with no
need for calibration targets or lens models.

How it works:

1. The sheet is tiled. In each tile the grid *phase* is measured - where the
   rulings sit inside one period - which pins that patch of paper to the lattice
   to sub-pixel precision.
2. Phase is only known modulo one period, so a tile a hundred periods along
   looks identical to its neighbour. The phases are unwrapped by propagating
   across neighbouring tiles, choosing at each step the whole number of periods
   that keeps the displacement continuous. This is the same problem, and the
   same solution, as unwrapping an interferogram.
3. That yields a correspondence per tile: where the paper is, against where it
   would be on a regular lattice. A low-order polynomial warp is fitted through
   those correspondences and the image is remapped.

The warp is deliberately low-order. A page curl is a smooth, gentle deformation;
letting the fit chase every tile would model the noise in the phase estimates
instead, and would happily invent distortion on a sheet that was already flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from miocard.pipeline.gridscan import GridScan, GridScanConfig, grid_ink, scan_grid


@dataclass(frozen=True)
class RectifyConfig:
    tile_px: int = 256
    overlap: float = 0.5
    # Degree of the fitted warp. Two is enough for curl and perspective; higher
    # starts fitting the phase noise.
    degree: int = 2
    # How faint a grid may be and still pin a patch of paper. The bar is set
    # relative to the sheet's own tiles, because how sharply the mesh prints
    # varies hugely between photographs; the absolute floor only rejects tiles
    # whose phase is indistinguishable from noise.
    keep_quantile: float = 0.30
    min_phase_contrast: float = 0.005
    # Residual, in pixels, above which the fit is not trusted and the image is
    # returned untouched rather than warped by a bad model.
    max_residual_px: float = 6.0
    min_tiles: int = 12


DEFAULT_RECTIFY_CONFIG = RectifyConfig()


@dataclass(frozen=True)
class LatticeField:
    """Where the paper is, against where a regular lattice says it should be."""

    centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    targets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    contrast: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    period_x: float = 0.0
    period_y: float = 0.0

    @property
    def count(self) -> int:
        return int(self.centres.shape[0])

    @property
    def displacement(self) -> np.ndarray:
        """How far each patch of paper has moved, in pixels."""
        if self.count == 0:
            return np.zeros((0, 2), np.float32)
        return self.targets - self.centres


@dataclass(frozen=True)
class Rectified:
    image: np.ndarray
    applied: bool
    px_per_mm: float = 0.0
    residual_px: float = 0.0
    max_shift_px: float = 0.0
    tiles_used: int = 0
    note: str = ""

    def summary(self) -> str:
        if not self.applied:
            return f"Not rectified ({self.note})."
        return (
            f"Rectified against {self.tiles_used} grid patches: "
            f"largest correction {self.max_shift_px:.1f} px, "
            f"fit residual {self.residual_px:.2f} px, "
            f"grid now {self.px_per_mm:.2f} px/mm"
        )


# ---------------------------------------------------------------------------
# Phase measurement
# ---------------------------------------------------------------------------


def _phase_along(ink: np.ndarray, axis: int, period: float) -> tuple[float, float]:
    """Sub-pixel phase of the grid along one axis, and how sharply it is defined.

    The profile is folded over one period and the result treated as a circular
    quantity: its first Fourier coefficient gives the phase without the
    quantisation that taking the darkest bin would impose, and its magnitude
    says how much the fold actually concentrated - which is exactly the
    confidence in that phase.
    """
    if period <= 1:
        return 0.0, 0.0
    profile = ink.mean(axis=0 if axis == 1 else 1).astype(np.float64)
    if profile.size < period * 3:
        return 0.0, 0.0
    profile = profile - profile.mean()
    positions = np.arange(profile.size, dtype=np.float64)
    angle = 2.0 * np.pi * positions / period
    # Grid lines are dark, so ink is high there: the phase of the fundamental
    # points at the rulings.
    coefficient = np.sum(profile * np.exp(-1j * angle))
    magnitude = float(np.abs(coefficient)) / max(profile.size, 1)
    spread = float(np.std(profile)) or 1.0
    phase = float(np.angle(coefficient)) % (2.0 * np.pi)
    return phase / (2.0 * np.pi) * period, magnitude / spread


def _unwrap(values: np.ndarray, period: float, valid: np.ndarray) -> np.ndarray:
    """Unwrap phases over a tile grid so displacement stays continuous.

    A phase is only known modulo the period, so a patch displaced by one whole
    period is indistinguishable from one not displaced at all. Walking the tile
    grid and choosing, at each step, the multiple of the period that keeps the
    change from the already-unwrapped neighbours smallest resolves it - provided
    the true displacement never jumps by more than half a period between
    adjacent tiles, which for a sheet of paper it does not.
    """
    rows, columns = values.shape
    unwrapped = np.array(values, dtype=np.float64, copy=True)
    done = np.zeros((rows, columns), dtype=bool)

    seeds = np.argwhere(valid)
    if seeds.size == 0:
        return unwrapped
    start_row, start_column = seeds[0]
    done[start_row, start_column] = True

    # Breadth-first over the tile lattice, so every tile is unwrapped against a
    # neighbour that has already been settled.
    queue = [(int(start_row), int(start_column))]
    while queue:
        row, column = queue.pop(0)
        reference = unwrapped[row, column]
        for delta_row, delta_column in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            next_row, next_column = row + delta_row, column + delta_column
            if not (0 <= next_row < rows and 0 <= next_column < columns):
                continue
            if done[next_row, next_column] or not valid[next_row, next_column]:
                continue
            offset = unwrapped[next_row, next_column] - reference
            unwrapped[next_row, next_column] -= period * round(offset / period)
            done[next_row, next_column] = True
            queue.append((next_row, next_column))
    return unwrapped


def measure_lattice_field(
    image: np.ndarray,
    scan: GridScan,
    config: RectifyConfig = DEFAULT_RECTIFY_CONFIG,
) -> LatticeField:
    """Measure, per patch of paper, where the grid actually sits."""
    if not scan.present or scan.period_x_px <= 1 or scan.period_y_px <= 1:
        return LatticeField()

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    size = max(64, config.tile_px)
    step = max(32, int(size * (1.0 - min(max(config.overlap, 0.0), 0.75))))
    tops = list(range(0, max(1, height - size + 1), step))
    lefts = list(range(0, max(1, width - size + 1), step))
    if not tops or not lefts:
        return LatticeField()

    rows, columns = len(tops), len(lefts)
    phase_x = np.zeros((rows, columns))
    phase_y = np.zeros((rows, columns))
    contrast = np.zeros((rows, columns))
    measured = np.zeros((rows, columns), dtype=bool)

    for row, top in enumerate(tops):
        for column, left in enumerate(lefts):
            patch = gray[top : top + size, left : left + size]
            if patch.shape[0] < size // 2 or patch.shape[1] < size // 2:
                continue
            ink = grid_ink(patch)
            px, sharp_x = _phase_along(ink, 1, scan.period_x_px)
            py, sharp_y = _phase_along(ink, 0, scan.period_y_px)
            # Phase is measured inside the tile; express it in image coordinates.
            phase_x[row, column] = (left + px) % scan.period_x_px
            phase_y[row, column] = (top + py) % scan.period_y_px
            contrast[row, column] = min(sharp_x, sharp_y)
            measured[row, column] = True

    if not measured.any():
        return LatticeField(period_x=scan.period_x_px, period_y=scan.period_y_px)
    observed = contrast[measured]
    threshold = max(
        config.min_phase_contrast, float(np.quantile(observed, config.keep_quantile))
    )
    valid = measured & (contrast >= threshold)

    if int(valid.sum()) < config.min_tiles:
        return LatticeField(period_x=scan.period_x_px, period_y=scan.period_y_px)

    unwrapped_x = _unwrap(phase_x, scan.period_x_px, valid)
    unwrapped_y = _unwrap(phase_y, scan.period_y_px, valid)
    # The reference is the middle of the sheet, so a correction pushes the edges
    # around rather than sliding the whole image sideways.
    reference_x = float(np.median(unwrapped_x[valid]))
    reference_y = float(np.median(unwrapped_y[valid]))

    centres: list[tuple[float, float]] = []
    targets: list[tuple[float, float]] = []
    weights: list[float] = []
    for row, top in enumerate(tops):
        for column, left in enumerate(lefts):
            if not valid[row, column]:
                continue
            centre_x = left + size / 2.0
            centre_y = top + size / 2.0
            # Where this patch of paper has drifted to, relative to a regular
            # lattice pinned at the middle of the sheet.
            shift_x = unwrapped_x[row, column] - reference_x
            shift_y = unwrapped_y[row, column] - reference_y
            centres.append((centre_x, centre_y))
            targets.append((centre_x - shift_x, centre_y - shift_y))
            weights.append(contrast[row, column])

    return LatticeField(
        centres=np.asarray(centres, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        contrast=np.asarray(weights, dtype=np.float32),
        period_x=float(scan.period_x_px),
        period_y=float(scan.period_y_px),
    )


# ---------------------------------------------------------------------------
# Warp fitting
# ---------------------------------------------------------------------------


def _design(points: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial terms in x and y up to *degree*."""
    x = points[:, 0]
    y = points[:, 1]
    terms = [np.ones_like(x)]
    for total in range(1, degree + 1):
        for power_x in range(total + 1):
            terms.append((x**power_x) * (y ** (total - power_x)))
    return np.stack(terms, axis=1)


def _fit_warp(
    field: LatticeField,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares polynomial from rectified coordinates back to source pixels.

    Fitting target -> source is what ``cv2.remap`` needs: for every pixel of the
    output it must say where to read from in the input.
    """
    design = _design(field.targets, degree)
    weights = np.sqrt(np.clip(field.contrast, 1e-3, None))[:, None]
    solution_x, *_ = np.linalg.lstsq(design * weights, field.centres[:, 0] * weights[:, 0], rcond=None)
    solution_y, *_ = np.linalg.lstsq(design * weights, field.centres[:, 1] * weights[:, 0], rcond=None)
    predicted = np.stack([design @ solution_x, design @ solution_y], axis=1)
    residual = float(np.sqrt(np.mean(np.sum((predicted - field.centres) ** 2, axis=1))))
    return solution_x, solution_y, residual


def rectify_to_grid(
    image: np.ndarray,
    scan: GridScan | None = None,
    config: RectifyConfig = DEFAULT_RECTIFY_CONFIG,
    scan_config: GridScanConfig | None = None,
) -> Rectified:
    """Warp the sheet so its printed grid becomes regular."""
    if image is None or image.size == 0:
        return Rectified(image=image, applied=False, note="empty image")
    if scan is None:
        scan = scan_grid(image, scan_config or GridScanConfig())
    if not scan.present:
        return Rectified(image=image, applied=False, note="no grid to rectify against")

    field = measure_lattice_field(image, scan, config)
    if field.count < config.min_tiles:
        return Rectified(
            image=image,
            applied=False,
            px_per_mm=scan.px_per_mm,
            tiles_used=field.count,
            note="too few patches carried a usable grid phase",
        )

    solution_x, solution_y, residual = _fit_warp(field, config.degree)
    if residual > config.max_residual_px:
        return Rectified(
            image=image,
            applied=False,
            px_per_mm=scan.px_per_mm,
            residual_px=residual,
            tiles_used=field.count,
            note=f"warp did not fit the measurements (residual {residual:.1f} px)",
        )

    height, width = image.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    output = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    design = _design(output, config.degree)
    map_x = (design @ solution_x).reshape(height, width).astype(np.float32)
    map_y = (design @ solution_y).reshape(height, width).astype(np.float32)

    shift = np.hypot(map_x - grid_x, map_y - grid_y)
    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Rectified(
        image=rectified,
        applied=True,
        px_per_mm=scan.px_per_mm,
        residual_px=residual,
        max_shift_px=float(np.percentile(shift, 99.5)),
        tiles_used=field.count,
    )
