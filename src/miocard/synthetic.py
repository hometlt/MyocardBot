"""Synthetic ECG sheets photographed badly, with the answer known exactly.

Every measurement in this project has been fought over five photographs, of
which one prints a heart rate and four print a pair of amplitudes. That is not
enough to tell a fix from a coincidence: twice during development a change
improved the score and was wrong, because the metric and the code shared an
assumption. What is missing is not a better metric but a bigger sample with a
known answer.

So this module prints an ECG sheet the way the recorder does - grid, twelve
rows, names in the left margin, calibration step, header and footer - draws a
signal it generated itself, and then abuses the result the way a phone camera
does: perspective, page curl, uneven light, glare, blur and JPEG. The signal in
millivolts, the row positions and the printed geometry all come back with it,
in the coordinates of the *final* picture rather than the clean one, which is
the only form in which they are worth anything.

The distortion is deliberately restricted to transformations that can be
inverted in closed form - a homography and a bow that depends on x alone - so
the ground truth is mapped exactly rather than resampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "DEFAULT_RECIPE",
    "SheetRecipe",
    "SyntheticLead",
    "SyntheticSheet",
    "make_sheet",
]

LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Peak R amplitude in mV per lead, roughly as a normal record prints: aVR
# inverted, poor R in V1 growing across the chest leads.
LEAD_R_MV = {
    "I": 0.7, "II": 1.1, "III": 0.5, "aVR": -0.8, "aVL": 0.35, "aVF": 0.75,
    "V1": 0.25, "V2": 0.6, "V3": 1.0, "V4": 1.6, "V5": 1.5, "V6": 1.1,
}
LEAD_S_MV = {
    "I": -0.1, "II": -0.15, "III": -0.1, "aVR": -0.1, "aVL": -0.08, "aVF": -0.12,
    "V1": -0.9, "V2": -1.2, "V3": -0.8, "V4": -0.35, "V5": -0.2, "V6": -0.1,
}
LEAD_T_MV = {
    "I": 0.18, "II": 0.25, "III": 0.08, "aVR": -0.15, "aVL": 0.08, "aVF": 0.15,
    "V1": 0.05, "V2": 0.3, "V3": 0.35, "V4": 0.32, "V5": 0.28, "V6": 0.2,
}
# The rows this recorder puts a calibration step on, counting from one.
CALIBRATION_ROWS = (3, 9)
# The clean sheet is drawn this many times larger and averaged down. Without it
# the millimetre ruling picks up a beat against the pixel grid: at a 4.8 px
# pitch the lines land on fractions 0, .8, .6, .4, .2 and repeat every fifth
# line, so the rendering itself manufactures a five-millimetre signal and the
# grid detector reports a scale five times too coarse.
SUPERSAMPLE = 3



@dataclass(frozen=True)
class SheetRecipe:
    """Everything that varies between one generated sheet and the next."""

    width: int = 1278
    height: int = 950
    px_per_mm: float = 4.8
    paper_speed_mm_s: float = 25.0
    gain_mm_mv: float = 10.0
    heart_rate_bpm: float = 72.0
    row_pitch_px: float = 74.0
    top_margin_px: float = 60.0
    # How the photograph was taken.
    rotation_deg: float = 0.8
    perspective: float = 0.012
    bow_px: float = 0.0
    glare: float = 0.0
    blur_px: float = 0.6
    illumination: float = 0.12
    noise_mv: float = 0.02
    wander_mv: float = 0.05
    # Beat-to-beat variation of the R-R interval, as a fraction. A metronome is
    # not a heart, and a perfectly regular record is not merely unrealistic - it
    # puts a strong clean periodicity on the sheet at the beat spacing, which
    # the grid scanner then locks onto in preference to the millimetre ruling.
    rr_variability: float = 0.06
    grain: float = 3.0
    jpeg_quality: int = 88
    ink: int = 35
    seed: int = 0


@dataclass(frozen=True)
class SyntheticLead:
    label: str
    # Row of this lead's isoelectric line in the final picture, per column.
    baseline_y: np.ndarray
    # The signal that was drawn, one sample per column of the clean sheet.
    samples_mv: np.ndarray
    r_peak_mv: float
    s_trough_mv: float


@dataclass(frozen=True)
class SyntheticSheet:
    image: np.ndarray
    recipe: SheetRecipe
    leads: tuple[SyntheticLead, ...]
    # Centre of each printed lead name in the final picture.
    label_rows: dict[str, tuple[float, float]] = field(default_factory=dict)
    r_peak_columns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    @property
    def heart_rate_bpm(self) -> float:
        return self.recipe.heart_rate_bpm

    def encode(self) -> bytes:
        ok, buffer = cv2.imencode(
            ".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, self.recipe.jpeg_quality]
        )
        if not ok:  # pragma: no cover - cv2 only fails on malformed input
            raise RuntimeError("Could not encode the synthetic sheet.")
        return buffer.tobytes()


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------


def _beat(t: np.ndarray, r_mv: float, s_mv: float, t_mv: float) -> np.ndarray:
    """One PQRST complex sampled at times ``t`` seconds from the R peak."""

    def bell(centre: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - centre) / width) ** 2)

    return (
        0.12 * bell(-0.16, 0.022)  # P
        + (-0.08 * abs(r_mv)) * bell(-0.022, 0.007)  # Q
        + r_mv * bell(0.0, 0.010)
        + s_mv * bell(0.026, 0.012)
        + t_mv * bell(0.22, 0.045)
    ).astype(np.float32)


def _signal(
    label: str,
    columns: int,
    seconds_per_column: float,
    recipe: SheetRecipe,
    rng: np.random.Generator,
    beat_times: np.ndarray,
) -> np.ndarray:
    time = np.arange(columns, dtype=np.float32) * seconds_per_column
    values = np.zeros(columns, dtype=np.float32)
    for beat_time in beat_times:
        near = np.abs(time - beat_time) < 0.45
        if near.any():
            values[near] += _beat(
                time[near] - beat_time, LEAD_R_MV[label], LEAD_S_MV[label], LEAD_T_MV[label]
            )
    if recipe.wander_mv > 0:
        phase = rng.uniform(0, 2 * np.pi)
        cycles = rng.uniform(0.6, 1.8)
        values += recipe.wander_mv * np.sin(2 * np.pi * cycles * time / max(time[-1], 1e-6) + phase)
    if recipe.noise_mv > 0:
        values += rng.normal(0.0, recipe.noise_mv, columns).astype(np.float32)
    return values


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _ruling(size: int, period: float, softness: float) -> np.ndarray:
    """Periodic line pattern, 1 on a line and 0 between, without rounding.

    Drawing each line at a rounded pixel puts the millimetre ruling at 0, 5, 10,
    14, 19 ... for a 4.8 px pitch, which is a different periodic signal from the
    one intended - and the grid detector, which works by autocorrelation, then
    locks onto the five-millimetre ruling instead and reports a scale five times
    too coarse. Compositing a continuous pattern keeps the period exact.
    """
    coordinate = np.arange(size, dtype=np.float32)
    distance = np.abs(((coordinate / period + 0.5) % 1.0) - 0.5) * period
    return np.clip(1.0 - distance / softness, 0.0, 1.0)


def _draw_grid(sheet: np.ndarray, px_per_mm: float, scale: int = 1) -> None:
    """Composite the millimetre ruling. Widths are given in final-image pixels."""
    height, width = sheet.shape[:2]
    field = np.zeros((height, width), dtype=np.float32)
    # Weights set by measurement, not by eye: the amplitude of the ruling in the
    # mean column profile of a blank patch is 0.37-0.72 grey levels on the real
    # photographs, against 2.6 for a version drawn to look right on screen. A
    # grid four times too strong is a different detection problem from the one
    # that has to be solved, and the whole point here is that it is the same one.
    for period, weight, softness in (
        (px_per_mm, 6.5, 0.80 * scale),
        (px_per_mm * 5.0, 9.0, 1.05 * scale),
    ):
        columns = _ruling(width, period, softness)[None, :]
        rows = _ruling(height, period, softness)[:, None]
        field = np.maximum(field, weight * np.maximum(columns, rows))
    tinted = sheet.astype(np.float32)
    # Printed in a warm red, so the blue channel loses most.
    for channel, share in enumerate((1.0, 0.72, 0.45)):
        tinted[:, :, channel] -= field * share
    sheet[:] = np.clip(tinted, 0, 255).astype(np.uint8)


def _draw_trace(
    sheet: np.ndarray,
    values_mv: np.ndarray,
    baseline: float,
    px_per_mv: float,
    left: int,
    ink: int,
    scale: int,
) -> None:
    xs = np.arange(values_mv.size, dtype=np.float32) * scale + left
    ys = baseline - values_mv * px_per_mv
    points = np.stack((xs, np.clip(ys, 0, sheet.shape[0] - 1)), axis=1).astype(np.int32)
    cv2.polylines(sheet, [points], False, (ink, ink, ink), 2 * scale, cv2.LINE_AA)


def _draw_furniture(
    sheet: np.ndarray,
    recipe: SheetRecipe,
    baselines: dict[str, float],
    left: int,
    scale: int,
) -> dict[str, tuple[float, float]]:
    """Lead names, calibration steps, header and footer; returns the name centres."""
    ink = (recipe.ink, recipe.ink, recipe.ink)
    label_rows: dict[str, tuple[float, float]] = {}
    for position, label in enumerate(LEAD_ORDER, start=1):
        baseline = baselines[label]
        # The recorder prints the name above the trace it names.
        y = baseline - 0.28 * recipe.row_pitch_px
        cv2.putText(sheet, label, (18 * scale, int(round(y * scale))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * scale, ink, scale, cv2.LINE_AA)
        label_rows[label] = (34.0, y - 5.0)
        if position in CALIBRATION_ROWS:
            step = recipe.gain_mm_mv * recipe.px_per_mm * scale
            x0, x1 = (left - 34) * scale, (left - 12) * scale
            base, top = int(baseline * scale), int(round(baseline * scale - step))
            cv2.polylines(
                sheet,
                [np.array([[x0, base], [x0, top], [x1, top], [x1, base]])],
                False,
                ink,
                2 * scale,
                cv2.LINE_AA,
            )
    cv2.putText(sheet, "ID :   Age :0 years   Sex :", (300 * scale, 22 * scale),
                cv2.FONT_HERSHEY_DUPLEX, 0.45 * scale, ink, scale, cv2.LINE_AA)
    cv2.putText(sheet, f"Rate: {recipe.heart_rate_bpm:.0f} bpm", (300 * scale, 40 * scale),
                cv2.FONT_HERSHEY_DUPLEX, 0.45 * scale, ink, scale, cv2.LINE_AA)
    footer = (
        f"Base:0Hz LPF:150Hz AC:50Hz     {recipe.gain_mm_mv:.1f}mm/mV   "
        f"{recipe.paper_speed_mm_s:.1f}mm/sec     EKG2000 Bionet Co., Ltd."
    )
    cv2.putText(sheet, footer, (18 * scale, sheet.shape[0] - 14 * scale),
                cv2.FONT_HERSHEY_DUPLEX, 0.45 * scale, ink, scale, cv2.LINE_AA)
    return label_rows


# ---------------------------------------------------------------------------
# Photographing it badly
# ---------------------------------------------------------------------------


def _homography(recipe: SheetRecipe) -> np.ndarray:
    width, height = recipe.width, recipe.height
    angle = np.deg2rad(recipe.rotation_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    centre = np.array([[1.0, 0.0, -width / 2], [0.0, 1.0, -height / 2], [0.0, 0.0, 1.0]])
    tilt = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [recipe.perspective / width, 0.0, 1.0]]
    )
    back = np.array([[1.0, 0.0, width / 2], [0.0, 1.0, height / 2], [0.0, 0.0, 1.0]])
    return back @ tilt @ rotation @ centre


def _bow(recipe: SheetRecipe, x: np.ndarray) -> np.ndarray:
    """Vertical displacement of a curled page, a function of x alone."""
    if recipe.bow_px == 0.0:
        return np.zeros_like(x, dtype=np.float64)
    return recipe.bow_px * np.sin(np.pi * np.clip(x, 0, recipe.width) / recipe.width)


def _forward(recipe: SheetRecipe, matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Clean-sheet points to final-picture points. The exact inverse of the warp."""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=1) @ matrix.T
    mapped = homogeneous[:, :2] / homogeneous[:, 2:3]
    mapped[:, 1] += _bow(recipe, mapped[:, 0])
    return mapped


def _photograph(sheet: np.ndarray, recipe: SheetRecipe, matrix: np.ndarray) -> np.ndarray:
    height, width = sheet.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height,
                                                                              dtype=np.float32))
    # remap wants destination -> source, and the warp is invertible in closed
    # form: undo the bow at this column, then the homography.
    source_y = grid_y - _bow(recipe, grid_x).astype(np.float32)
    stacked = np.stack((grid_x, source_y, np.ones_like(grid_x)), axis=-1)
    inverse = np.linalg.inv(matrix).astype(np.float32)
    warped = stacked @ inverse.T
    map_x = (warped[..., 0] / warped[..., 2]).astype(np.float32)
    map_y = (warped[..., 1] / warped[..., 2]).astype(np.float32)
    out = cv2.remap(sheet, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    rng = np.random.default_rng(recipe.seed + 991)
    if recipe.illumination > 0:
        gradient = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        vertical = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        field = 1.0 + recipe.illumination * (0.6 * gradient + 0.4 * vertical)
        out = np.clip(out.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)
    if recipe.glare > 0:
        cx = int(rng.uniform(0.25, 0.75) * width)
        cy = int(rng.uniform(0.25, 0.75) * height)
        radius = int(min(width, height) * 0.22)
        spot = np.zeros((height, width), dtype=np.float32)
        cv2.circle(spot, (cx, cy), radius, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), radius * 0.5)
        out = np.clip(
            out.astype(np.float32) + recipe.glare * 255.0 * spot[..., None], 0, 255
        ).astype(np.uint8)
    if recipe.blur_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), recipe.blur_px)
    if recipe.grain > 0:
        grain = rng.normal(0.0, recipe.grain, out.shape[:2]).astype(np.float32)
        out = np.clip(out.astype(np.float32) + grain[..., None], 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def make_sheet(recipe: SheetRecipe | None = None) -> SyntheticSheet:
    """Print, draw and photograph one sheet, and return it with its answer."""
    recipe = recipe or DEFAULT_RECIPE
    rng = np.random.default_rng(recipe.seed)
    scale = SUPERSAMPLE
    sheet = np.full((recipe.height * scale, recipe.width * scale, 3), 252, dtype=np.uint8)
    _draw_grid(sheet, recipe.px_per_mm * scale, scale)

    left = int(round(recipe.px_per_mm * 12))
    columns = recipe.width - left - 8
    seconds_per_column = 1.0 / (recipe.px_per_mm * recipe.paper_speed_mm_s)
    interval = 60.0 / recipe.heart_rate_bpm
    duration = columns * seconds_per_column
    count = max(2, int(duration / interval))
    jitter = rng.normal(0.0, recipe.rr_variability, count) if recipe.rr_variability > 0 else (
        np.zeros(count)
    )
    beat_times = 0.35 * interval + np.cumsum(interval * (1.0 + jitter)) - interval
    beat_times = beat_times[beat_times < duration]

    baselines = {
        label: recipe.top_margin_px + position * recipe.row_pitch_px
        for position, label in enumerate(LEAD_ORDER)
    }
    px_per_mv = recipe.px_per_mm * recipe.gain_mm_mv

    signals: dict[str, np.ndarray] = {}
    for label in LEAD_ORDER:
        values = _signal(label, columns, seconds_per_column, recipe, rng, beat_times)
        signals[label] = values
        _draw_trace(
            sheet, values, baselines[label] * scale, px_per_mv * scale, left * scale,
            recipe.ink, scale,
        )
    label_rows = _draw_furniture(sheet, recipe, baselines, left, scale)
    sheet = cv2.resize(sheet, (recipe.width, recipe.height), interpolation=cv2.INTER_AREA)

    matrix = _homography(recipe)
    image = _photograph(sheet, recipe, matrix)

    xs = np.arange(left, left + columns, dtype=np.float64)
    leads = []
    for label in LEAD_ORDER:
        mapped = _forward(recipe, matrix, np.stack((xs, np.full_like(xs, baselines[label])),
                                                   axis=1))
        baseline_y = np.interp(np.arange(recipe.width), mapped[:, 0], mapped[:, 1])
        leads.append(
            SyntheticLead(
                label=label,
                baseline_y=baseline_y.astype(np.float32),
                samples_mv=signals[label],
                r_peak_mv=float(LEAD_R_MV[label]),
                s_trough_mv=float(LEAD_S_MV[label]),
            )
        )

    mapped_labels = {
        label: tuple(_forward(recipe, matrix, np.array([point]))[0])
        for label, point in label_rows.items()
    }
    peak_columns = _forward(
        recipe,
        matrix,
        np.stack(
            (left + beat_times / seconds_per_column, np.full(beat_times.size, recipe.height / 2)),
            axis=1,
        ),
    )[:, 0]
    return SyntheticSheet(
        image=image,
        recipe=recipe,
        leads=tuple(leads),
        label_rows={label: (float(x), float(y)) for label, (x, y) in mapped_labels.items()},
        r_peak_columns=np.round(peak_columns).astype(np.int32),
    )


DEFAULT_RECIPE = SheetRecipe()
