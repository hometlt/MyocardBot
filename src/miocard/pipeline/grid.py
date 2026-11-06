"""Stage 3 - Grid detector.

The millimetre grid is what turns pixels into seconds and millivolts, so it is
measured rather than assumed. A printed grid repeats at the small-square pitch
and at every multiple of it; the shortest autocorrelation peak that also has a
second-harmonic partner is therefore the millimetre pitch and not the 5 mm block
or an accidental rhythm in the waveform.

Both axes are measured independently. They must agree, because a square grid has
one pitch: a disagreement means residual perspective or a spurious peak, and
reporting nothing is safer than handing the model a wrong time scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from miocard.pipeline.config import DEFAULT_CONFIG, PipelineConfig
from miocard.pipeline.state import PipelineStageResult, clamp01

LARGE_CELL_SMALL_SQUARES = 5

# Kept as module constants because the imaging facade and its tests refer to
# them; the authoritative values live on PipelineConfig.
MIN_GRID_PERIOD_PX = DEFAULT_CONFIG.min_grid_period_px
MAX_GRID_PERIOD_PX = DEFAULT_CONFIG.max_grid_period_px
GRID_PEAK_THRESHOLD = DEFAULT_CONFIG.grid_peak_threshold


@dataclass(frozen=True)
class GridMetrics:
    """Millimetre grid pitch measured locally, in pixels of the analysed image."""

    small_cell_px: float
    strength: float
    axis_ratio: float
    period_x_px: float = 0.0
    period_y_px: float = 0.0
    phase_x_px: int = 0
    phase_y_px: int = 0
    strength_x: float = 0.0
    strength_y: float = 0.0

    def __post_init__(self) -> None:
        # Older callers construct the metrics from the averaged pitch alone.
        if self.period_x_px <= 0.0:
            object.__setattr__(self, "period_x_px", self.small_cell_px)
        if self.period_y_px <= 0.0:
            object.__setattr__(self, "period_y_px", self.small_cell_px)
        if self.strength_x <= 0.0:
            object.__setattr__(self, "strength_x", self.strength)
        if self.strength_y <= 0.0:
            object.__setattr__(self, "strength_y", self.strength)

    @property
    def large_cell_px(self) -> float:
        return self.small_cell_px * LARGE_CELL_SMALL_SQUARES

    @property
    def px_per_mm(self) -> float:
        return self.small_cell_px

    @property
    def confidence(self) -> str:
        if self.strength >= 0.22 and 0.97 <= self.axis_ratio <= 1.03:
            return "high"
        if self.strength >= GRID_PEAK_THRESHOLD and 0.93 <= self.axis_ratio <= 1.07:
            return "medium"
        return "low"

    @property
    def probability(self) -> float:
        agreement = clamp01(1.0 - abs(self.axis_ratio - 1.0) / 0.10)
        return clamp01(0.15 + 0.55 * clamp01(self.strength / 0.30) + 0.30 * agreement)


def grid_ink(image: np.ndarray) -> np.ndarray:
    """Paper-corrected ink with the pen strokes removed, leaving the printed mesh.

    The grid is an order of magnitude fainter than the trace, so averaging the
    raw image swamps it. Subtracting the reconstructed paper and then discarding
    anything as dark as pen ink leaves the mesh as the dominant structure.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    ink = cv2.subtract(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel), gray).astype(np.float32)
    cap = max(float(np.percentile(ink, 90)), 3.0)
    return np.where(ink > cap, 0.0, ink).astype(np.float32)


def _axis_autocorrelation(ink: np.ndarray, axis: int, config: PipelineConfig) -> np.ndarray:
    """Autocorrelation along one axis, averaged over every line of the image.

    Averaging a thousand per-line autocorrelations rather than autocorrelating
    the mean profile is what makes a mesh only a few grey levels deep
    measurable: the grid repeats identically on every line, so it adds
    coherently while noise averages away. It is also phase-insensitive, so a
    fraction of a degree of residual rotation does not smear the peak.
    """
    lines = ink if axis == 1 else ink.T
    length = lines.shape[1]
    max_lag = config.max_grid_period_px * 2 + 2
    if length < max_lag * 2:
        return np.zeros(0)
    window = config.max_grid_period_px * 2 + 1
    trend = cv2.blur(lines, (window, 1), borderType=cv2.BORDER_REPLICATE)
    centered = lines - trend
    padded = int(2 ** math.ceil(math.log2(length * 2)))
    spectrum = np.fft.rfft(centered, n=padded, axis=1)
    power = (spectrum * np.conjugate(spectrum)).real.mean(axis=0)
    correlation = np.fft.irfft(power, n=padded)
    if correlation[0] <= 0:
        return np.zeros(0)
    return correlation[: max_lag + 1] / correlation[0]


def _peak_prominence(correlation: np.ndarray, lag: int, config: PipelineConfig) -> float:
    """Height of a peak above the decaying autocorrelation envelope around it.

    The envelope falls steeply near zero lag, so an absolute threshold would miss
    a perfectly regular grid whose peaks ride low on that slope.
    """
    if lag <= 0 or lag + 1 >= correlation.size:
        return 0.0
    value = correlation[lag]
    if value < correlation[lag - 1] or value < correlation[lag + 1]:
        return 0.0
    half = config.min_grid_period_px // 2 + 1
    left = correlation[max(0, lag - half) : lag]
    right = correlation[lag + 1 : lag + 1 + half]
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(value - (float(left.min()) + float(right.min())) / 2)


HARMONIC_COUNT = 3


def _harmonic_score(correlation: np.ndarray, lag: int, config: PipelineConfig) -> float:
    """Weakest of a candidate's first few harmonics.

    A printed grid repeats at its pitch and at *every* multiple of it, so the
    weakest harmonic is the discriminating statistic: an accidental peak has
    nothing at three times its lag, however tall it happens to be.
    """
    prominences: list[float] = []
    for order in range(1, HARMONIC_COUNT + 1):
        target = order * lag
        offsets = [offset for offset in (target - 1, target, target + 1) if 0 < offset < correlation.size - 1]
        if not offsets:
            break
        prominences.append(max(_peak_prominence(correlation, offset, config) for offset in offsets))
    if len(prominences) < 2:
        return 0.0
    return float(min(prominences))


def _smallest_periodic_peak(
    correlation: np.ndarray,
    config: PipelineConfig,
) -> tuple[float, float]:
    """The millimetre pitch: the fundamental of the strongest harmonic series."""
    last_lag = min(config.max_grid_period_px, correlation.size - 2)
    candidates = [
        lag
        for lag in range(config.min_grid_period_px, last_lag + 1)
        if _peak_prominence(correlation, lag, config) >= config.grid_peak_threshold
    ]
    if not candidates:
        return 0.0, 0.0

    best = max(candidates, key=lambda lag: _harmonic_score(correlation, lag, config))
    if _harmonic_score(correlation, best, config) <= 0.0:
        return 0.0, 0.0

    # The strongest series may be a multiple of the pitch - the 5 mm block, or an
    # aliasing beat with the JPEG block grid. Whenever half of it is itself a
    # genuine peak, that half is the real millimetre pitch.
    while best // 2 >= config.min_grid_period_px:
        half = best // 2
        if _peak_prominence(correlation, half, config) < config.grid_peak_threshold:
            break
        best = half

    # Parabolic interpolation recovers sub-pixel pitch, which compounds into
    # whole squares over the width of a row.
    left, right = correlation[best - 1], correlation[best + 1]
    denominator = left - 2 * correlation[best] + right
    offset = 0.5 * (left - right) / denominator if denominator else 0.0
    prominence = _peak_prominence(correlation, best, config)
    return float(best) + float(np.clip(offset, -0.5, 0.5)), prominence


def estimate_phase(profile: np.ndarray, period: float) -> int:
    """Offset of the first grid line, so the grid model lands on the printed lines."""
    rounded = max(1, int(round(period)))
    if profile.size == 0 or rounded <= 1:
        return 0
    detrended = profile - cv2.blur(
        profile.reshape(1, -1).astype(np.float32),
        (rounded * 2 + 1, 1),
        borderType=cv2.BORDER_REPLICATE,
    ).ravel()
    scores = [float(detrended[offset::rounded].mean()) for offset in range(rounded)]
    return int(np.argmax(scores))


def measure_grid(
    image: np.ndarray,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> GridMetrics | None:
    """Measure the millimetre grid pitch so intervals do not rest on eyeballed squares."""
    ink = grid_ink(image)
    # Lags along x reveal the vertical rulings, and vice versa.
    period_x, strength_x = _smallest_periodic_peak(
        _axis_autocorrelation(ink, 1, config), config
    )
    period_y, strength_y = _smallest_periodic_peak(
        _axis_autocorrelation(ink, 0, config), config
    )
    if period_x <= 0 or period_y <= 0:
        return None
    column_profile = ink.mean(axis=0)
    row_profile = ink.mean(axis=1)
    return GridMetrics(
        small_cell_px=(period_x + period_y) / 2,
        strength=min(strength_x, strength_y),
        axis_ratio=period_x / period_y,
        period_x_px=period_x,
        period_y_px=period_y,
        phase_x_px=estimate_phase(column_profile, period_x),
        phase_y_px=estimate_phase(row_profile, period_y),
        strength_x=strength_x,
        strength_y=strength_y,
    )


def grid_probability_map(image: np.ndarray, grid: GridMetrics | None) -> np.ndarray:
    """Per-pixel proximity to the modelled grid lines, in [0, 1].

    This is a prior on *where* the grid is, not evidence that ink is present; the
    cost map multiplies it with the chromatic grid response so a pixel only
    counts as grid when the model and the picture agree.
    """
    height, width = image.shape[:2]
    if grid is None or grid.small_cell_px < 1:
        return np.zeros((height, width), dtype=np.float32)

    period_x = max(1.0, float(grid.period_x_px))
    period_y = max(1.0, float(grid.period_y_px))
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    # Distance to the nearest modelled line, wrapped into a half period.
    x_dist = np.abs(((xs - grid.phase_x_px + period_x / 2) % period_x) - period_x / 2)
    y_dist = np.abs(((ys - grid.phase_y_px + period_y / 2) % period_y) - period_y / 2)
    # A printed line is roughly one pixel wide plus blur; fall off over a pixel
    # and a half so the model is tolerant of sub-pixel phase error.
    sigma = max(0.8, min(period_x, period_y) / 8.0)
    x_near = np.exp(-0.5 * (x_dist / sigma) ** 2)
    y_near = np.exp(-0.5 * (y_dist / sigma) ** 2)
    proximity = np.maximum(x_near[None, :], y_near[:, None])
    return np.clip(proximity, 0.0, 1.0).astype(np.float32)


def grid_stage(grid: GridMetrics | None) -> PipelineStageResult:
    if grid is None:
        return PipelineStageResult(
            "Grid detector",
            0.10,
            "No repeating millimetre pitch found; the time scale stays unknown.",
            evidence=("grid_confidence=missing",),
        )
    return PipelineStageResult(
        "Grid detector",
        grid.probability,
        f"Millimetre pitch {grid.small_cell_px:.2f} px "
        f"(5 mm block {grid.large_cell_px:.1f} px).",
        evidence=(
            f"grid_confidence={grid.confidence}",
            f"period_x={grid.period_x_px:.2f}",
            f"period_y={grid.period_y_px:.2f}",
            f"axis_ratio={grid.axis_ratio:.3f}",
            f"strength={grid.strength:.3f}",
        ),
    )
