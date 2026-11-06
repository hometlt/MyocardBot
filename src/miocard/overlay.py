"""The marked-up photograph that goes back to the person who sent it.

This is a different job from :mod:`miocard.stage_render`, which draws what each
stage published so a developer can see where the pipeline went wrong. Here the
reader is whoever photographed the sheet, and the question being answered is
narrower: *did the machine read my ECG correctly, and which parts did it not
actually read?*

So the drawing shows only what can be checked against the paper underneath it -
which strip was taken for which lead, where the pen was reconstructed, which
stretches were filled in rather than measured, and which deflections were
counted as beats. Nothing here is an interpretation; a claim about the heart
belongs in the report text, not drawn onto the evidence.

Every colour means one thing:

    green    the pen position as measured from the print
    orange   interpolated - there was no readable ink here
    blue     a complex the beat detector counted
    grey     the span outside which nothing was measured
"""

from __future__ import annotations

import cv2
import numpy as np

from miocard.imaging import ImageProcessingError, ProcessedImage
from miocard.pipeline.series import effective_trace
from miocard.pipeline.state import ECGPipelineState, LeadPath

__all__ = ["annotate_digitisation", "digitisation_summary", "render_digitisation"]

MEASURED = (60, 190, 60)
INTERPOLATED = (0, 150, 255)
UNREADABLE = (170, 70, 170)
BEAT = (230, 120, 30)
LABEL_INK = (25, 25, 25)
BOUND = (150, 150, 150)

# How far the photograph is washed out before the overlay goes on. Enough that a
# one-pixel green line reads over a printed trace, little enough that the reader
# can still see the print and disagree with us.
FADE = 0.42
BANNER_HEIGHT = 34
LEGEND_HEIGHT = 30


def _fade(image: np.ndarray) -> np.ndarray:
    white = np.full_like(image, 255)
    return cv2.addWeighted(image, 1.0 - FADE, white, FADE, 0)


def _scale(image: np.ndarray) -> float:
    """Line and text scale, so a 900 px sheet and a 2500 px one look alike."""
    return max(1.0, min(image.shape[1], image.shape[0]) / 900.0)


def _reconstructed(state: ECGPipelineState, path: LeadPath) -> np.ndarray:
    px_per_mm = float(state.grid.small_cell_px) if state.grid is not None else 0.0
    smooth_px = max(3, int(round(px_per_mm))) if px_per_mm > 1 else 3
    return effective_trace(path, smooth_px)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) spans where ``flags`` is true."""
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=True))


def _polyline(
    panel: np.ndarray,
    rows: np.ndarray,
    x_start: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if rows.size < 2:
        return
    xs = np.arange(rows.size, dtype=np.int32) + int(x_start)
    ys = np.clip(np.round(rows), 0, panel.shape[0] - 1).astype(np.int32)
    cv2.polylines(panel, [np.stack((xs, ys), axis=1)], False, color, thickness, cv2.LINE_AA)


def _chip(
    panel: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = LABEL_INK,
) -> None:
    """A lead name on an opaque tag, because the sheet under it is not blank."""
    font_scale = 0.46 * scale
    thickness = max(1, int(round(scale)))
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    pad = max(2, int(round(3 * scale)))
    x = int(np.clip(anchor[0], 0, panel.shape[1] - width - 2 * pad - 1))
    y = int(np.clip(anchor[1] - height // 2 - pad, 0, panel.shape[0] - height - 2 * pad - 1))
    cv2.rectangle(
        panel,
        (x, y),
        (x + width + 2 * pad, y + height + 2 * pad),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(panel, (x, y), (x + width + 2 * pad, y + height + 2 * pad), color, 1)
    cv2.putText(
        panel,
        text,
        (x + pad, y + height + pad - 1),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _band(panel: np.ndarray, height: int, text: str, color: tuple[int, int, int]) -> np.ndarray:
    strip = np.full((height, panel.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        strip,
        text,
        (10, height - max(9, height // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return strip


def _legend(width: int, scale: float) -> np.ndarray:
    strip = np.full((LEGEND_HEIGHT, width, 3), 245, dtype=np.uint8)
    entries = (
        (MEASURED, "measured"),
        (INTERPOLATED, "interpolated"),
        (UNREADABLE, "unreadable/inferred"),
        (BEAT, "counted beat"),
        (BOUND, "no trace outside"),
    )
    x = 12
    y = LEGEND_HEIGHT // 2
    for color, text in entries:
        cv2.line(strip, (x, y), (x + 22, y), color, max(2, int(round(2 * scale))), cv2.LINE_AA)
        x += 28
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(
            strip, text, (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA
        )
        x += text_width + 26
        if x > width - 90:
            break
    return strip


def digitisation_summary(state: ECGPipelineState) -> str:
    """One line stating what was read, in the terms the drawing uses."""
    leads = len(state.paths)
    parts = [f"{leads} leads digitised"]
    grid = state.grid
    if grid is not None and grid.confidence != "low":
        parts.append(f"{grid.small_cell_px:.1f} px/mm")
    beats = state.beats
    if beats is not None and beats.count:
        parts.append(f"{beats.heart_rate_bpm:.0f} bpm on {beats.source_lead}")
    interpolated = _interpolated_fraction(state)
    if interpolated > 0.005:
        parts.append(f"{interpolated * 100:.0f}% interpolated")
    unreadable = _unreadable_fraction(state)
    if unreadable > 0.005:
        parts.append(f"{unreadable * 100:.0f}% unreadable")
    parts.append(f"quality: {state.quality_grade}")
    return "  |  ".join(parts)


def _interpolated_fraction(state: ECGPipelineState) -> float:
    covered = 0
    filled = 0
    for path in state.paths:
        span = max(0, path.x_end - path.x_start)
        covered += span
        if path.recovered is not None and span:
            filled += int(np.count_nonzero(path.recovered[path.x_start : path.x_end]))
    return filled / covered if covered else 0.0


def _unreadable_fraction(state: ECGPipelineState) -> float:
    covered = 0
    unreadable = 0
    for path in state.paths:
        span = max(0, path.x_end - path.x_start)
        covered += span
        if path.unreadable is not None and span:
            unresolved = path.unreadable & ~(
                path.recovered
                if path.recovered is not None
                else np.zeros(path.width, dtype=bool)
            )
            unreadable += int(np.count_nonzero(unresolved[path.x_start : path.x_end]))
    return unreadable / covered if covered else 0.0


def render_digitisation(image: np.ndarray, state: ECGPipelineState) -> np.ndarray:
    """Draw the reconstruction over the sheet it was read from.

    ``image`` must be the picture the pipeline ran on, since every column and
    row published on the blackboard indexes into exactly those pixels.
    """
    panel = _fade(image.copy())
    scale = _scale(panel)
    thickness = max(1, int(round(1.6 * scale)))

    if not state.paths:
        return np.vstack(
            (
                _band(panel, BANNER_HEIGHT, "No lead rows could be read from this sheet.", (80, 80, 255)),
                panel,
            )
        )

    height, width = panel.shape[:2]
    for path in state.paths:
        rows = _reconstructed(state, path)
        start, end = int(path.x_start), int(path.x_end)
        if end <= start:
            continue

        # Outside the traced span nothing was measured, so say so rather than
        # letting a reader assume the lead was flat there.
        for edge in (start, end - 1):
            cv2.line(
                panel,
                (edge, max(0, path.band.top + 2)),
                (edge, min(height - 1, path.band.bottom - 2)),
                BOUND,
                1,
                cv2.LINE_AA,
            )

        recovered = path.recovered
        if recovered is None:
            recovered = np.zeros(path.y.size, dtype=bool)
        unreadable = path.unreadable
        if unreadable is None:
            unreadable = recovered.copy()
        measured_window = ~unreadable[start:end].astype(bool)
        recovered_window = recovered[start:end].astype(bool)
        unresolved_window = unreadable[start:end].astype(bool) & ~recovered_window
        span = rows[start:end]
        for run_start, run_end in _runs(measured_window):
            _polyline(panel, span[run_start:run_end], start + run_start, MEASURED, thickness)
        for run_start, run_end in _runs(recovered_window):
            # Draw interpolation over the raw solver path: the envelope trick
            # that recovers a QRS amplitude has no meaning where there was no
            # ink to take an envelope of.
            _polyline(
                panel,
                path.y[start + run_start : start + run_end],
                start + run_start,
                INTERPOLATED,
                thickness + 1,
            )
        for run_start, run_end in _runs(unresolved_window):
            # This is the solver's best guess, retained because the export must
            # have one value per column, but it is explicitly not a measurement.
            _polyline(
                panel,
                path.y[start + run_start : start + run_end],
                start + run_start,
                UNREADABLE,
                thickness + 1,
            )

        for peak in state.beats_per_lead.get(path.index, ()):
            column = int(peak)
            if not start <= column < end:
                continue
            if unreadable[column]:
                continue
            row = int(np.clip(rows[column], 0, height - 1))
            cv2.circle(panel, (column, row), max(3, int(round(3.2 * scale))), BEAT, thickness)

        # One left-hand column of tags, at the height the line actually starts:
        # the traced span begins at a different column on every lead, and tags
        # scattered across the margin read as noise rather than as an index.
        # Kept inside the lead's own strip, because a line that happens to begin
        # on a rising edge would otherwise put its tag above its neighbour's and
        # read as two leads swapped.
        label = path.label or f"row {path.index + 1}"
        margin = max(2, int(round(6 * scale)))
        anchor = float(np.clip(rows[start], path.band.top + margin, path.band.bottom - margin))
        _chip(panel, label, (int(round(4 * scale)), int(round(anchor))), scale)

    banner = _band(panel, BANNER_HEIGHT, digitisation_summary(state), (170, 245, 170))
    return np.vstack((banner, panel, _legend(width, scale)))


def annotate_digitisation(image: ProcessedImage) -> bytes | None:
    """Encode the marked-up sheet as PNG, or ``None`` if nothing was read.

    Returning ``None`` rather than a bare photograph is deliberate: an overlay
    with no overlay on it would suggest the sheet was checked and found clean.
    """
    state = image.pipeline
    if state is None or not state.paths:
        return None
    decoded = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageProcessingError("Unable to decode image for the digitisation overlay.")
    success, encoded = cv2.imencode(
        ".png",
        render_digitisation(decoded, state),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not success:
        raise ImageProcessingError("Unable to encode the digitisation overlay.")
    return encoded.tobytes()
