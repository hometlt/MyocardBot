"""Standalone millimetre-grid scanner, run at the source resolution.

The grid may be printed as thin continuous rulings, as a lattice of dots, as
rulings in one direction only, or it may be absent altogether - a plain trace on
white paper is a legitimate input. Those look quite different to a line
detector, so this scanner does not look for lines at all. It measures
*periodicity*, which dots and rulings share: the pitch is whatever repeats.

The measurement is an autocorrelation along each axis, averaged over every line
of a tile. That averaging is the whole trick. A printed mesh only a few grey
levels deep is invisible in any single line, but it repeats identically on all
of them, so hundreds of per-line autocorrelations add it coherently while the
trace and the sensor noise average away. A dotted grid works for the same
reason: only the lines that pass through a row of dots carry the period, and
those add while the rest contribute noise.

A two-dimensional transform of the same tile was tried first and is not good
enough. Measured on a real sheet, the true grid frequency stood at z = 1.5
against a noise floor of z = 3.5 - the peak was there, but buried, because one
2-D transform cannot average across lines the way this does. The spectrum is
still used, but only for what it *is* reliable at: locating the dominant
direction, so a mesh rotated relative to the page can be straightened per tile
before it is measured.

"No grid" is a real answer here rather than a detector quietly returning noise:
a pitch is only reported when a peak clears the prominence floor *and* its
harmonics are present, which an accidental peak's are not.

The image is processed in overlapping tiles. That keeps memory bounded on a
40-megapixel photograph, and it makes the pitch a *local* measurement, so a
curled or perspective-distorted sheet reports its true variation across the page
rather than one averaged pitch that fits nowhere.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

# Millimetre pitches worth searching for, in pixels. Below about three pixels a
# grid is indistinguishable from sensor noise; above a hundred it would imply a
# crop of a couple of centimetres rather than a sheet.
MIN_PERIOD_PX = 3.0
MAX_PERIOD_PX = 100.0
# A square grid has one pitch. Axes further apart than this are not a millimetre
# grid, whatever else they may be.
MAX_AXIS_RATIO = 1.25


@dataclass(frozen=True)
class GridScanConfig:
    tile_px: int = 512
    overlap: float = 0.25
    min_period_px: float = MIN_PERIOD_PX
    max_period_px: float = MAX_PERIOD_PX
    # Tiles whose ink is this sparse carry nothing to measure - blank margin,
    # or the empty half of a sheet.
    min_ink_fraction: float = 0.002
    max_tiles: int = 400
    # Autocorrelation prominence a peak must reach to be a candidate pitch.
    min_prominence: float = 0.05
    # How far off square the mesh may be and still be straightened per tile.
    rotation_search_deg: float = 8.0


DEFAULT_SCAN_CONFIG = GridScanConfig()


@dataclass(frozen=True)
class TileGrid:
    """What one tile of the sheet says about the grid."""

    left: int
    top: int
    width: int
    height: int
    period_x_px: float = 0.0
    period_y_px: float = 0.0
    angle_deg: float = 0.0
    strength: float = 0.0
    kind: str = "absent"

    @property
    def present(self) -> bool:
        return self.kind not in {"absent", "blank"}

    @property
    def measurable(self) -> bool:
        """Whether the tile carried enough ink for the question to mean anything."""
        return self.kind != "blank"

    @property
    def centre(self) -> tuple[float, float]:
        return self.left + self.width / 2.0, self.top + self.height / 2.0


@dataclass(frozen=True)
class GridScan:
    """The scanner's verdict for a whole image."""

    image_size: tuple[int, int]
    tiles: tuple[TileGrid, ...] = field(default_factory=tuple)
    present: bool = False
    probability: float = 0.0
    period_x_px: float = 0.0
    period_y_px: float = 0.0
    angle_deg: float = 0.0
    kind: str = "absent"
    coverage: float = 0.0
    consistency: float = 0.0
    blank_tiles: int = 0
    # Offset of the first ruling from the image edge, in pixels. Without this
    # the pitch alone cannot say *where* the lines are.
    phase_x_px: float = 0.0
    phase_y_px: float = 0.0
    note: str = ""

    @property
    def px_per_mm(self) -> float:
        """Pixels per millimetre, if a square grid was found."""
        if not self.present or self.period_x_px <= 0 or self.period_y_px <= 0:
            return 0.0
        return float((self.period_x_px + self.period_y_px) / 2.0)

    @property
    def confidence(self) -> str:
        if self.probability >= 0.70:
            return "high"
        if self.probability >= 0.45:
            return "medium"
        return "low"

    def summary(self) -> str:
        if not self.present:
            return f"No grid detected ({self.note or 'no periodic structure'})."
        return (
            f"{self.kind} grid at {self.px_per_mm:.2f} px/mm "
            f"({self.period_x_px:.2f} x {self.period_y_px:.2f} px, "
            f"{self.angle_deg:+.2f} deg), seen on {self.coverage * 100:.0f}% of tiles, "
            f"consistency {self.consistency:.2f}, confidence {self.confidence}"
        )


# ---------------------------------------------------------------------------
# Ink extraction
# ---------------------------------------------------------------------------


def grid_ink(tile: np.ndarray) -> np.ndarray:
    """Paper-corrected ink with the pen strokes flattened out.

    The grid is far fainter than the trace, so the trace would dominate the
    spectrum and its beat interval could be mistaken for a pitch. Clipping ink
    above a high percentile turns every pen stroke into the same constant, which
    contributes to the zero frequency and nowhere else.
    """
    gray = tile if tile.ndim == 2 else cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    ink = cv2.subtract(background, gray).astype(np.float32)
    cap = max(float(np.percentile(ink, 90)), 2.0)
    return np.minimum(ink, cap)


# ---------------------------------------------------------------------------
# Spectral peak finding
# ---------------------------------------------------------------------------


def _power_spectrum(ink: np.ndarray) -> np.ndarray:
    """Centred log power spectrum of a windowed tile."""
    rows, columns = ink.shape
    centred = ink - float(ink.mean())
    # A Hann window stops the tile edges from smearing a cross through the
    # spectrum, which would otherwise look exactly like an axis-aligned grid.
    window = np.outer(np.hanning(rows), np.hanning(columns)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(centred * window))
    return np.log1p(np.abs(spectrum).astype(np.float32))


def _radius_mask(shape: tuple[int, int], low: float, high: float) -> np.ndarray:
    """Frequencies whose period falls inside the accepted range."""
    rows, columns = shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(columns))[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    with np.errstate(divide="ignore"):
        period = np.where(radius > 0, 1.0 / np.maximum(radius, 1e-12), np.inf)
    return (period >= low) & (period <= high)


def _peak_prominence_map(spectrum: np.ndarray) -> np.ndarray:
    """How far each frequency stands above its own neighbourhood, in sigmas."""
    background = cv2.medianBlur(spectrum, 5)
    residual = spectrum - background
    spread = float(np.std(residual))
    if spread <= 1e-6:
        return np.zeros_like(residual)
    return residual / spread


def _find_peaks(
    prominence: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    count: int = 6,
) -> list[tuple[float, float, float]]:
    """Strongest distinct frequencies, as (fy, fx, z) in cycles per pixel."""
    rows, columns = prominence.shape
    fy_axis = np.fft.fftshift(np.fft.fftfreq(rows))
    fx_axis = np.fft.fftshift(np.fft.fftfreq(columns))
    working = np.where(mask, prominence, -np.inf)

    # Suppress a neighbourhood around each pick so one broad peak is not
    # reported several times over.
    radius = max(2, min(rows, columns) // 64)
    peaks: list[tuple[float, float, float]] = []
    for _ in range(count * 2):
        index = int(np.argmax(working))
        row, column = divmod(index, columns)
        z = float(working[row, column])
        if not np.isfinite(z) or z < threshold:
            break
        fy, fx = float(fy_axis[row]), float(fx_axis[column])
        # The spectrum is symmetric; keep one half so a peak and its mirror are
        # not counted as two directions.
        if fx > 0 or (abs(fx) < 1e-12 and fy >= 0):
            peaks.append((fy, fx, z))
        working[
            max(0, row - radius) : row + radius + 1,
            max(0, column - radius) : column + radius + 1,
        ] = -np.inf
        if len(peaks) >= count:
            break
    return peaks


def _direction(fy: float, fx: float) -> tuple[float, float]:
    """Period in pixels and direction in degrees for one spectral peak."""
    radius = math.hypot(fy, fx)
    if radius <= 0:
        return 0.0, 0.0
    return 1.0 / radius, math.degrees(math.atan2(fy, fx))


def _axis_autocorrelation(ink: np.ndarray, axis: int, max_lag: int) -> np.ndarray:
    """Autocorrelation along one axis, averaged over every line of the tile.

    This, and not a two-dimensional transform of the tile, is what makes a grid
    only a few grey levels deep measurable. The mesh repeats identically on
    every line, so averaging hundreds of per-line autocorrelations adds it
    coherently while the trace and the noise average away. A single 2-D spectrum
    of the same tile buries the grid peak: measured on a real sheet it stood at
    z=1.5 against a noise floor of z=3.5, while this recovers it cleanly.

    It works on a dotted grid for the same reason. Only the lines that pass
    through a row of dots carry the period, but those add coherently and the
    rest contribute nothing but noise.
    """
    lines = ink if axis == 1 else ink.T
    length = lines.shape[1]
    if length < max_lag * 2:
        return np.zeros(0, dtype=np.float32)
    window = max(3, int(max_lag) | 1)
    trend = cv2.blur(lines, (window, 1), borderType=cv2.BORDER_REPLICATE)
    centred = lines - trend
    padded = 1 << int(math.ceil(math.log2(length * 2)))
    spectrum = np.fft.rfft(centred, n=padded, axis=1)
    power = (spectrum * np.conjugate(spectrum)).real.mean(axis=0)
    correlation = np.fft.irfft(power, n=padded)
    if correlation[0] <= 0:
        return np.zeros(0, dtype=np.float32)
    return (correlation[: max_lag + 1] / correlation[0]).astype(np.float32)


def _peak_prominence(correlation: np.ndarray, lag: int, half: int) -> float:
    if lag <= 0 or lag + 1 >= correlation.size:
        return 0.0
    value = correlation[lag]
    if value < correlation[lag - 1] or value < correlation[lag + 1]:
        return 0.0
    left = correlation[max(0, lag - half) : lag]
    right = correlation[lag + 1 : lag + 1 + half]
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(value - (float(left.min()) + float(right.min())) / 2.0)


def _harmonic_score(correlation: np.ndarray, lag: int, half: int, orders: int = 3) -> float:
    """Weakest of the first few harmonics of a candidate period.

    A printed grid repeats at its pitch and at every multiple of it, so the
    weakest harmonic discriminates: an accidental peak has nothing at three
    times its lag, however tall it happens to be.
    """
    values: list[float] = []
    for order in range(1, orders + 1):
        target = order * lag
        offsets = [o for o in (target - 1, target, target + 1) if 0 < o < correlation.size - 1]
        if not offsets:
            break
        values.append(max(_peak_prominence(correlation, o, half) for o in offsets))
    return min(values) if len(values) >= 2 else 0.0


def _fundamental(correlation: np.ndarray, config: GridScanConfig) -> tuple[float, float]:
    """The pitch: the fundamental of the strongest harmonic series."""
    if correlation.size < 8:
        return 0.0, 0.0
    low = max(2, int(math.floor(config.min_period_px)))
    high = min(int(math.ceil(config.max_period_px)), correlation.size - 2)
    if high <= low:
        return 0.0, 0.0
    half = max(2, low // 2 + 1)
    floor = config.min_prominence
    candidates = [
        lag for lag in range(low, high + 1) if _peak_prominence(correlation, lag, half) >= floor
    ]
    if not candidates:
        return 0.0, 0.0

    best = max(candidates, key=lambda lag: _harmonic_score(correlation, lag, half))
    if _harmonic_score(correlation, best, half) <= 0.0:
        return 0.0, 0.0
    # The strongest series may be a multiple of the pitch - the 5 mm block, or a
    # beat against the JPEG block grid. Whenever half of it is itself a genuine
    # peak, that half is the real pitch.
    while best // 2 >= low and _peak_prominence(correlation, best // 2, half) >= floor:
        best //= 2

    left, right = correlation[best - 1], correlation[best + 1]
    denominator = left - 2 * correlation[best] + right
    offset = 0.5 * (left - right) / denominator if denominator else 0.0
    return float(best) + float(np.clip(offset, -0.5, 0.5)), _peak_prominence(
        correlation, best, half
    )


def _tile_rotation(ink: np.ndarray, config: GridScanConfig) -> float:
    """Rotation of the mesh relative to the page, from the tile's spectrum.

    The two-dimensional spectrum is unreliable for *measuring* the pitch here,
    but it locates the dominant direction well enough to straighten the tile
    before the autocorrelation, which is what that measurement needs.
    """
    if config.rotation_search_deg <= 0:
        return 0.0
    spectrum = _power_spectrum(ink)
    mask = _radius_mask(spectrum.shape, config.min_period_px, config.max_period_px)
    if not mask.any():
        return 0.0
    peaks = _find_peaks(_peak_prominence_map(spectrum), mask, 1.5, count=4)
    if not peaks:
        return 0.0
    fy, fx, _z = peaks[0]
    _period, angle = _direction(fy, fx)
    rotation = ((angle + 45.0) % 90.0) - 45.0
    if abs(rotation) > config.rotation_search_deg:
        return 0.0
    return float(rotation)


def analyse_tile(tile: np.ndarray, config: GridScanConfig) -> TileGrid:
    """Measure the grid in one tile, or return ``None`` if there is nothing to measure."""
    ink = grid_ink(tile)
    if float(np.count_nonzero(ink > 1.0)) / max(ink.size, 1) < config.min_ink_fraction:
        # Blank paper. Reported as its own kind rather than dropped, so a margin
        # is never confused with a region that has ink but no grid.
        return TileGrid(0, 0, tile.shape[1], tile.shape[0], kind="blank")

    rotation = _tile_rotation(ink, config)
    if abs(rotation) > 0.2:
        height, width = ink.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1.0)
        ink = cv2.warpAffine(
            ink, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    max_lag = int(min(config.max_period_px * 2 + 2, min(ink.shape) // 2))
    period_x, strength_x = _fundamental(_axis_autocorrelation(ink, 1, max_lag), config)
    period_y, strength_y = _fundamental(_axis_autocorrelation(ink, 0, max_lag), config)

    empty = TileGrid(0, 0, tile.shape[1], tile.shape[0], angle_deg=rotation)
    if period_x <= 0 and period_y <= 0:
        return empty

    if period_x > 0 and period_y > 0:
        ratio = max(period_x, period_y) / max(min(period_x, period_y), 1e-6)
        if ratio <= MAX_AXIS_RATIO:
            return TileGrid(
                0,
                0,
                tile.shape[1],
                tile.shape[0],
                period_x_px=period_x,
                period_y_px=period_y,
                angle_deg=rotation,
                strength=float(min(strength_x, strength_y)),
                kind="lattice",
            )
        # Two periodicities that are not a square grid - most often the lead-row
        # spacing masquerading as a ruling. Keep only the stronger direction.
        if strength_x >= strength_y:
            period_y, strength_y = 0.0, 0.0
        else:
            period_x, strength_x = 0.0, 0.0

    if period_x > 0:
        return TileGrid(
            0,
            0,
            tile.shape[1],
            tile.shape[0],
            period_x_px=period_x,
            angle_deg=rotation,
            strength=float(strength_x),
            kind="lines_x",
        )
    return TileGrid(
        0,
        0,
        tile.shape[1],
        tile.shape[0],
        period_y_px=period_y,
        angle_deg=rotation,
        strength=float(strength_y),
        kind="lines_y",
    )


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


def iter_tiles(
    height: int,
    width: int,
    config: GridScanConfig,
) -> Iterator[tuple[int, int, int, int]]:
    """Overlapping tile rectangles covering the image."""
    size = max(64, int(config.tile_px))
    if height <= size and width <= size:
        yield 0, 0, width, height
        return
    step = max(32, int(size * (1.0 - min(max(config.overlap, 0.0), 0.75))))
    tops = list(range(0, max(1, height - size + 1), step))
    lefts = list(range(0, max(1, width - size + 1), step))
    if tops[-1] + size < height:
        tops.append(max(0, height - size))
    if lefts[-1] + size < width:
        lefts.append(max(0, width - size))
    for top in tops:
        for left in lefts:
            yield left, top, min(size, width - left), min(size, height - top)


def scan_grid(image: np.ndarray, config: GridScanConfig = DEFAULT_SCAN_CONFIG) -> GridScan:
    """Detect the millimetre grid across a full-resolution image."""
    if image is None or image.size == 0:
        return GridScan(image_size=(0, 0), note="empty image")
    height, width = image.shape[:2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles = list(iter_tiles(height, width, config))
    if len(rectangles) > config.max_tiles:
        # Keep the budget by sampling evenly rather than by shrinking the tiles,
        # which would cost frequency resolution exactly where it is needed.
        stride = int(math.ceil(len(rectangles) / config.max_tiles))
        rectangles = rectangles[::stride]

    tiles: list[TileGrid] = []
    for left, top, tile_width, tile_height in rectangles:
        patch = gray[top : top + tile_height, left : left + tile_width]
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            continue
        measured = analyse_tile(patch, config)
        tiles.append(
            TileGrid(
                left=left,
                top=top,
                width=tile_width,
                height=tile_height,
                period_x_px=measured.period_x_px,
                period_y_px=measured.period_y_px,
                angle_deg=measured.angle_deg,
                strength=measured.strength,
                kind=measured.kind,
            )
        )

    scan = _aggregate(tiles, (height, width))
    if not scan.present:
        return scan
    # Phase is measured once on the whole sheet: it is a property of the print,
    # not of a tile, and the tiles already agreed on the pitch.
    ink = grid_ink(gray)
    return replace(
        scan,
        phase_x_px=estimate_phase(ink, 1, scan.period_x_px),
        phase_y_px=estimate_phase(ink, 0, scan.period_y_px),
    )


def estimate_phase(ink: np.ndarray, axis: int, period: float) -> float:
    """Offset of the first ruling along one axis, in pixels.

    The pitch says how often the lines repeat; the phase says where they start.
    Folding the ink profile over one period and taking the darkest bin gives it
    directly, and it is what lets an overlay be checked against the print.
    """
    if period <= 1:
        return 0.0
    profile = ink.mean(axis=0 if axis == 1 else 1).astype(np.float64)
    if profile.size < period * 2:
        return 0.0
    bins = max(2, int(round(period)))
    positions = np.arange(profile.size)
    index = np.minimum((positions % period / period * bins).astype(np.int32), bins - 1)
    totals = np.bincount(index, weights=profile, minlength=bins)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    folded = totals / np.maximum(counts, 1.0)
    return float(np.argmax(folded)) / bins * period


def _aggregate(tiles: list[TileGrid], image_size: tuple[int, int]) -> GridScan:
    if not tiles:
        return GridScan(image_size=image_size, note="image too small to tile")

    blank = [tile for tile in tiles if not tile.measurable]
    inked = [tile for tile in tiles if tile.measurable]
    if not inked:
        return GridScan(
            image_size=image_size,
            tiles=tuple(tiles),
            blank_tiles=len(blank),
            note="no tile carried enough ink to measure",
        )

    lattices = [tile for tile in inked if tile.kind == "lattice"]
    ruled = [tile for tile in inked if tile.kind in {"lines_x", "lines_y"}]
    measured = lattices or ruled
    # Coverage is over the tiles that had ink in them. Counting blank margin
    # against the grid would punish a sheet for having margins.
    coverage = len(measured) / len(inked)
    if not measured:
        return GridScan(
            image_size=image_size,
            tiles=tuple(tiles),
            coverage=0.0,
            blank_tiles=len(blank),
            note="ink present but nothing repeating at a plausible pitch",
        )

    def median(values: list[float]) -> float:
        return float(np.median(values)) if values else 0.0

    period_x = median([tile.period_x_px for tile in measured if tile.period_x_px > 0])
    period_y = median([tile.period_y_px for tile in measured if tile.period_y_px > 0])
    angle = median([tile.angle_deg for tile in measured])
    strength = median([tile.strength for tile in measured])

    # Consistency: how tightly the tiles agree on the pitch. A real printed grid
    # varies only with perspective; scattered values mean the peaks were noise.
    spread: list[float] = []
    for values, centre in ((
        [tile.period_x_px for tile in measured if tile.period_x_px > 0],
        period_x,
    ), (
        [tile.period_y_px for tile in measured if tile.period_y_px > 0],
        period_y,
    )):
        if len(values) >= 2 and centre > 0:
            spread.append(float(np.median(np.abs(np.asarray(values) - centre))) / centre)
    consistency = float(np.clip(1.0 - 4.0 * (np.mean(spread) if spread else 1.0), 0.0, 1.0))

    kind = "lattice" if lattices else (ruled[0].kind if ruled else "absent")
    probability = float(
        np.clip(
            0.15 * np.clip(strength / 0.25, 0.0, 1.0)
            + 0.35 * coverage
            + 0.35 * consistency
            + (0.15 if kind == "lattice" else 0.0),
            0.0,
            1.0,
        )
    )
    return GridScan(
        image_size=image_size,
        tiles=tuple(tiles),
        present=True,
        probability=probability,
        period_x_px=period_x,
        period_y_px=period_y,
        angle_deg=angle,
        kind=kind,
        coverage=coverage,
        consistency=consistency,
        blank_tiles=len(blank),
    )
