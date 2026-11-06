"""Debug visualisations for the MyocardBot2 pipeline.

MyocardBot2.md lists debug visualisations as a design principle, so every
renderer here draws what the corresponding stage actually published on the
blackboard - the fitted grid model, the detected lead bands, the class
posteriors, the solved paths, the recovered gaps, the beats, the templates and
the digitised signals. Nothing is re-derived from the picture for display.

Stage index (1-based) matches the pipeline order in MyocardBot2.md:

    1.  Image Loader              9.  Path Solver
    2.  Geometry Analyzer         10. Gap Recovery
    3.  Grid Detector             11. Beat Detector
    4.  Layout Detector           12. Template Builder
    5.  OCR & Semantic Analysis   13. Signal Optimizer
    6.  ECG Trace Probability     14. Quality Analyzer
    7.  Noise / Artifact          15. Export
    8.  Cost Map Builder

``render_stage(stage_index, source, processed, result)`` returns a colour
``np.ndarray`` (BGR) suitable for ``cv2.imwrite``.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable

import cv2
import numpy as np

from miocard.imaging import ProcessedImage
from miocard.pipeline.series import effective_trace
from miocard.pipeline.state import CLASS_NAMES, ECGPipelineState

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

PALETTE = (
    (60, 220, 60),
    (255, 170, 40),
    (60, 120, 255),
    (255, 90, 220),
    (40, 220, 220),
    (200, 120, 255),
)


def fit_to_size(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize ``image`` to fit inside *target_width* x *target_height*."""
    h, w = image.shape[:2]
    scale = min(target_width / max(w, 1), target_height / max(h, 1))
    scale = min(scale, 1.0)  # never upscale
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    y = max(0, (target_height - image.shape[0]) // 2)
    x = max(0, (target_width - image.shape[1]) // 2)
    canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """Add a dark label band above the image."""
    band = np.full((28, image.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(
        band, text, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA
    )
    return np.vstack((band, image))


def pad_to_square(
    image: np.ndarray,
    background_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    h, w = image.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size, 3), background_color, dtype=np.uint8)
    y = (size - h) // 2
    x = (size - w) // 2
    canvas[y : y + h, x : x + w] = image
    return canvas


def wrap_lines(lines: Iterable[str], width_chars: int = 120) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width_chars) or [""])
    return wrapped


def draw_text_block(
    lines: Iterable[str],
    width: int,
    height: int,
    start_y: int = 8,
) -> np.ndarray:
    """Render a multi-line text block on a light-grey canvas."""
    wrapped = wrap_lines(lines, width_chars=max(40, width // 8))
    if not wrapped:
        wrapped = ["No data."]
    block = np.full((height, width, 3), 245, dtype=np.uint8)
    line_height = 16
    y = start_y
    for line in wrapped[: max(1, (height - start_y) // line_height)]:
        cv2.putText(
            block, line, (10, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA
        )
        y += line_height
    return block


def _banner(panel: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(panel, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _pipeline(result: ProcessedImage) -> ECGPipelineState | None:
    return result.pipeline


def _polyline(
    panel: np.ndarray,
    y: np.ndarray,
    color,
    thickness: int = 1,
    x_offset: int = 0,
) -> None:
    if y.size < 2:
        return
    xs = np.arange(y.size, dtype=np.int32) + x_offset
    points = np.stack((xs, np.clip(np.round(y), 0, panel.shape[0] - 1).astype(np.int32)), axis=1)
    cv2.polylines(panel, [points], False, color, thickness, cv2.LINE_AA)


def _traced(path) -> tuple[np.ndarray, int]:
    """The part of the raw solver path that stands on a printed trace."""
    return path.y[path.x_start : path.x_end], path.x_start


def _reconstructed(state: ECGPipelineState, path) -> tuple[np.ndarray, int]:
    """The pen position the signal stages actually use, over its traced span.

    This is not the same as the solver path. Where the pen swept vertically
    through a QRS the whole column is ink, and the solver - which places one row
    per column - correctly stays on the baseline while the amplitude lives at
    the end of that ink run. Drawing the raw path here would show a flat line
    through every QRS and misrepresent a reconstruction that is in fact right.
    """
    px_per_mm = float(state.grid.small_cell_px) if state.grid is not None else 0.0
    smooth_px = max(3, int(round(px_per_mm))) if px_per_mm > 1 else 3
    rows = effective_trace(path, smooth_px)
    return rows[path.x_start : path.x_end], path.x_start


# ---------------------------------------------------------------------------
# Probability-map helpers
# ---------------------------------------------------------------------------


def _probability_map_red(
    image: np.ndarray,
    probability: np.ndarray | None,
    title: str,
) -> np.ndarray:
    """Overlay the top 10 % of a probability map in red."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    if probability is None:
        probability = np.zeros((h, w), dtype=np.float32)
    else:
        probability = np.asarray(probability, dtype=np.float32)
        if probability.shape != (h, w):
            probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)

    weight = np.clip(probability, 0.0, 1.0)
    flat = weight.reshape(-1)
    threshold = float(np.quantile(flat, 0.90)) if flat.size else 1.0

    if threshold >= 1.0:
        alpha = np.zeros_like(weight, dtype=np.float32)
    else:
        alpha = np.clip((weight - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        alpha = alpha.astype(np.float32) ** 0.75

    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    red = np.zeros_like(base)
    red[:, :, 2] = 255.0
    blended = (base * (1.0 - alpha[..., None]) + red * alpha[..., None]).astype(np.uint8)
    cv2.putText(
        blended,
        f"{title}  p(mean)={float(np.mean(weight)):.3f}, highlight>{threshold:.2f}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return blended


def _probability_map_heat(
    image: np.ndarray,
    probability: np.ndarray | None,
    title: str,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Heatmap blended over the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if probability is None:
        probability = np.zeros((h, w), dtype=np.float32)
    else:
        probability = np.asarray(probability, dtype=np.float32)
        if probability.shape != (h, w):
            probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)

    normalized = np.clip(probability, 0.0, 1.0)
    norm = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, colormap)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
    cv2.putText(
        blended,
        f"{title} p(mean)={float(np.mean(normalized)):.3f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return blended


# ---------------------------------------------------------------------------
# Per-stage renderers  (stage_index is 1-based)
# ---------------------------------------------------------------------------


def render_image_loader(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 1: original next to the picture the model is shown."""
    h = max(source.shape[0], processed.shape[0])
    s = np.full((h, source.shape[1], 3), 245, dtype=np.uint8)
    s[: source.shape[0]] = source
    p = np.full((h, processed.shape[1], 3), 245, dtype=np.uint8)
    p[: processed.shape[0]] = processed
    gap = np.full((h, 12, 3), 200, dtype=np.uint8)
    panel = np.hstack((s, gap, p))
    return add_label(panel, "1. Image Loader - original (left) -> processed (right)")


def render_geometry(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 2: the measured skew angle against a level reference."""
    panel = processed.copy()
    state = _pipeline(result)
    geometry = state.geometry if state else {}
    angle = float(geometry.get("deskew_angle_deg", 0.0) or 0.0)
    sharpness = float(geometry.get("angle_sharpness", 0.0) or 0.0)
    h, w = panel.shape[:2]
    cx, cy = w // 2, h // 2
    length = int(min(w, h) * 0.48)
    cv2.line(panel, (cx - length // 2, cy), (cx + length // 2, cy), (170, 170, 170), 1, cv2.LINE_AA)
    rad = math.radians(angle)
    dx = int(math.cos(rad) * length / 2)
    dy = int(math.sin(rad) * length / 2)
    cv2.line(panel, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 180, 0), 2, cv2.LINE_AA)
    _banner(
        panel,
        f"deskew {angle:+.2f} deg  sharpness={sharpness:.2f}  "
        f"applied={str(geometry.get('rotation_applied', False)).lower()}  "
        f"cropped={str(geometry.get('crop_applied', False)).lower()}",
        (120, 255, 120),
    )
    return add_label(panel, "2. Geometry Analyzer")


def render_grid(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 3: the fitted grid model, at its measured pitch and phase."""
    panel = processed.copy()
    grid = result.grid
    if grid is None:
        _banner(panel, "No confident grid detected", (80, 80, 255))
        return add_label(panel, "3. Grid Detector")

    h, w = panel.shape[:2]
    overlay = panel.copy()
    x = float(grid.phase_x_px)
    while x < w:
        cv2.line(overlay, (int(round(x)), 0), (int(round(x)), h - 1), (255, 130, 0), 1)
        x += grid.period_x_px
    y = float(grid.phase_y_px)
    while y < h:
        cv2.line(overlay, (0, int(round(y))), (w - 1, int(round(y))), (255, 130, 0), 1)
        y += grid.period_y_px
    cv2.addWeighted(overlay, 0.45, panel, 0.55, 0, panel)
    _banner(
        panel,
        f"pitch x={grid.period_x_px:.2f} y={grid.period_y_px:.2f} px  "
        f"ratio={grid.axis_ratio:.3f}  strength={grid.strength:.3f}  ({grid.confidence})",
        (255, 190, 120),
    )
    return add_label(panel, "3. Grid Detector - fitted millimetre model")


def render_layout(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 4: detected lead bands and their baselines."""
    panel = processed.copy()
    state = _pipeline(result)
    h, w = panel.shape[:2]
    if state is None or not state.bands:
        _banner(panel, "No lead rows detected", (80, 80, 255))
        return add_label(panel, "4. Layout Detector")

    overlay = panel.copy()
    for band in state.bands:
        color = PALETTE[band.index % len(PALETTE)]
        cv2.rectangle(overlay, (0, band.top), (w - 1, band.bottom), color, -1)
        cv2.line(
            panel,
            (0, int(round(band.baseline))),
            (w - 1, int(round(band.baseline))),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            band.label,
            (6, max(12, band.top + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.14, panel, 0.86, 0, panel)
    layout = state.layout
    _banner(
        panel,
        f"{len(state.bands)} rows  pitch={float(layout.get('row_pitch_px', 0)):.1f} px  "
        f"comb={float(layout.get('comb_score', 0)):.2f}  "
        f"regularity={float(layout.get('baseline_regularity', 0)):.2f}  "
        f"columns={int(layout.get('column_blocks', 1))}",
        (255, 210, 120),
    )
    return add_label(panel, "4. Layout Detector - lead bands and baselines")


def render_ocr(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 5: semantic objects kept as penalties, never painted over."""
    panel = processed.copy()
    state = _pipeline(result)
    if state is None:
        return add_label(panel, "5. OCR & Semantic Analysis")

    colors = {
        "lead_label": (60, 220, 60),
        "calibration": (0, 165, 255),
        "header": (255, 160, 60),
        "footer": (255, 160, 60),
        "annotation": (200, 120, 255),
    }
    for obj in state.text_objects:
        color = colors.get(obj.kind, (180, 180, 180))
        cv2.rectangle(panel, (obj.left, obj.top), (obj.right, obj.bottom), color, 1, cv2.LINE_AA)
    _banner(
        panel,
        f"{len(state.text_objects)} objects; gain={float(state.text.get('gain_mm_mv', 0)):.1f} mm/mV "
        f"({'measured' if state.text.get('gain_measured') else 'assumed'}); "
        f"engine={'pytesseract' if state.text.get('ocr_used') else 'geometry only'}",
        (150, 220, 255),
    )
    return add_label(panel, "5. OCR & Semantic Analysis - nothing erased")


def render_trace_probability(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 6: P(trace) as published by the estimator."""
    state = _pipeline(result)
    maps = state.probability_maps if state else {}
    panel = _probability_map_heat(
        processed, maps.get("trace"), "P(trace)", cv2.COLORMAP_TURBO
    )
    return add_label(panel, "6. ECG Trace Probability Estimator")


def render_noise_artifact(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 7: P(noise) and P(artifact) side by side."""
    state = _pipeline(result)
    maps = state.probability_maps if state else {}
    left = _probability_map_heat(processed, maps.get("noise"), "P(noise)", cv2.COLORMAP_HOT)
    right = _probability_map_heat(
        processed, maps.get("artifact"), "P(artifact)", cv2.COLORMAP_INFERNO
    )
    gap = np.full((left.shape[0], 8, 3), 60, dtype=np.uint8)
    return add_label(np.hstack((left, gap, right)), "7. Noise / Artifact Detector")


def render_cost_map(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 8: the traversal cost the solver actually walks."""
    state = _pipeline(result)
    unary = state.unary_cost if state else None
    if unary is None:
        return add_label(processed.copy(), "8. Cost Map Builder - unavailable")
    low, high = float(np.percentile(unary, 1)), float(np.percentile(unary, 99))
    normalized = np.clip((unary - low) / max(high - low, 1e-6), 0.0, 1.0)
    panel = cv2.applyColorMap((255 * normalized).astype(np.uint8), cv2.COLORMAP_PLASMA)
    summary = state.cost_map
    _banner(
        panel,
        "traversal cost = -log P(trace) + text/grid/noise/artifact penalties   "
        f"mean={float(summary.get('unary_mean', 0.0)):.2f}",
        (255, 235, 200),
    )
    return add_label(panel, "8. Cost Map Builder (dark = cheap)")


def render_path_solver(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 9: the minimum-cost path found for every lead."""
    panel = processed.copy()
    state = _pipeline(result)
    if state is None or not state.paths:
        _banner(panel, "No paths solved", (80, 80, 255))
        return add_label(panel, "9. Path Solver")
    for path in state.paths:
        color = PALETTE[path.index % len(PALETTE)]
        raw, offset = _traced(path)
        _polyline(panel, raw, (170, 170, 170), 1, offset)
        rows, offset = _reconstructed(state, path)
        _polyline(panel, rows, color, 1, offset)
        # Where the line was judged to begin, and how sure that was.
        for column, probability in (
            (path.x_start, path.start_probability),
            (path.x_end - 1, path.end_probability),
        ):
            shade = int(60 + 195 * float(np.clip(probability, 0.0, 1.0)))
            cv2.line(
                panel,
                (int(column), max(0, path.band.top)),
                (int(column), min(panel.shape[0] - 1, path.band.bottom)),
                (shade, shade, 40),
                1,
                cv2.LINE_AA,
            )
    solver = state.stage("Path solver")
    _banner(
        panel,
        f"{len(state.paths)} paths  " + (", ".join(solver.evidence) if solver else ""),
        (140, 255, 140),
    )
    return add_label(
        panel, "9. Path Solver - solver path (grey) and the pen it reconstructs (colour)"
    )


def render_gap_recovery(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 10: measured trace in green, interpolated stretches in orange."""
    panel = processed.copy()
    state = _pipeline(result)
    if state is None or not state.paths:
        return add_label(panel, "10. Gap Recovery - no paths")
    for path in state.paths:
        traced, offset = _reconstructed(state, path)
        _polyline(panel, traced, (60, 200, 60), 1, offset)
        recovered = path.recovered
        if recovered is not None and recovered.any():
            padded = np.concatenate(([False], recovered, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            for start, end in zip(edges[0::2], edges[1::2], strict=True):
                _polyline(panel, path.y[start:end], (0, 165, 255), 2, int(start))
        unreadable = path.unreadable
        if unreadable is not None:
            unresolved = unreadable & ~(
                recovered if recovered is not None else np.zeros(path.width, dtype=bool)
            )
            padded = np.concatenate(([False], unresolved, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            for start, end in zip(edges[0::2], edges[1::2], strict=True):
                _polyline(panel, path.y[start:end], (170, 70, 170), 2, int(start))
    gap = state.stage("Gap recovery")
    _banner(panel, gap.summary if gap else "", (150, 200, 255))
    return add_label(
        panel,
        "10. Gap Recovery - orange = interpolated, purple = unreadable/inferred",
    )


def render_beat_detector(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 11: detected R peaks marked on every lead."""
    panel = processed.copy()
    state = _pipeline(result)
    if state is None or state.beats is None:
        _banner(panel, "No beats detected", (80, 80, 255))
        return add_label(panel, "11. Beat Detector")
    beats = state.beats
    by_label = {path.label: path for path in state.paths}
    reference = by_label.get(beats.source_lead)
    for path in state.paths:
        faint, faint_offset = _reconstructed(state, path)
        _polyline(panel, faint, (170, 170, 170), 1, faint_offset)
    if reference is not None:
        traced, offset = _reconstructed(state, reference)
        _polyline(panel, traced, (60, 200, 60), 2, offset)
        for peak in beats.r_peaks:
            x = int(np.clip(peak, 0, reference.y.size - 1))
            index = int(np.clip(x - offset, 0, max(traced.size - 1, 0)))
            y = int(np.clip(traced[index] if traced.size else 0, 0, panel.shape[0] - 1))
            cv2.circle(panel, (x, y), 5, (0, 80, 255), 2, cv2.LINE_AA)
    _banner(
        panel,
        f"{beats.count} beats on {beats.source_lead}: {beats.heart_rate_bpm:.0f} bpm, "
        f"RR regularity {beats.regularity:.2f}",
        (150, 200, 255),
    )
    return add_label(panel, "11. Beat Detector")


def render_template_builder(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 12: the median beat of each lead, drawn as a strip of cards."""
    state = _pipeline(result)
    width = processed.shape[1]
    if state is None or not state.templates:
        panel = np.full((160, width, 3), 245, dtype=np.uint8)
        _banner(panel, "No templates built", (80, 80, 255))
        return add_label(panel, "12. Template Builder")

    columns = min(6, len(state.templates))
    rows = int(math.ceil(len(state.templates) / columns))
    card_w = max(120, width // columns)
    card_h = 120
    board = np.full((rows * card_h + 30, card_w * columns, 3), 245, dtype=np.uint8)
    for position, template in enumerate(state.templates):
        row, column = divmod(position, columns)
        x0, y0 = column * card_w, row * card_h + 30
        card = board[y0 : y0 + card_h, x0 : x0 + card_w]
        cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (210, 210, 210), 1)
        values = template.samples_mv.astype(np.float32)
        span = float(np.max(values) - np.min(values)) or 1.0
        ys = card_h - 22 - (values - float(np.min(values))) / span * (card_h - 40)
        xs = np.linspace(6, card_w - 6, values.size)
        points = np.stack((xs, ys), axis=1).astype(np.int32)
        cv2.polylines(card, [points], False, PALETTE[template.index % len(PALETTE)], 1, cv2.LINE_AA)
        cv2.putText(
            card,
            f"{template.label}  r={float(np.median(template.correlations)):.2f}",
            (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    stage = state.stage("Template builder")
    _banner(board, stage.summary if stage else "", (150, 200, 255))
    return add_label(board, "12. Template Builder - median beat per lead")


def render_signal_optimizer(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 13: the digitised signals, redrawn from the exported values."""
    state = _pipeline(result)
    width = processed.shape[1]
    if state is None or not state.lead_signals:
        panel = np.full((200, width, 3), 245, dtype=np.uint8)
        _banner(panel, "No signals reconstructed", (80, 80, 255))
        return add_label(panel, "13. Signal Optimizer")

    lane = 70
    board = np.full((lane * len(state.lead_signals) + 30, width, 3), 250, dtype=np.uint8)
    units = str(state.signals.get("units", "px"))
    for position, signal in enumerate(state.lead_signals):
        y0 = 30 + position * lane
        centre = y0 + lane // 2
        cv2.line(board, (0, centre), (width - 1, centre), (225, 225, 225), 1)
        values = signal.traced.astype(np.float32)
        if values.size < 2:
            values = signal.values_mv.astype(np.float32)
        span = float(np.percentile(np.abs(values), 99)) or 1.0
        ys = centre - np.clip(values / span, -1.4, 1.4) * (lane * 0.42)
        xs = np.arange(values.size, dtype=np.float32)
        step = max(1, values.size // width)
        points = np.stack((xs[::step], ys[::step]), axis=1).astype(np.int32)
        cv2.polylines(
            board, [points], False, PALETTE[signal.index % len(PALETTE)], 1, cv2.LINE_AA
        )
        cv2.putText(
            board,
            f"{signal.label}  {signal.amplitude_mv:.2f} {units}  conf={float(np.mean(signal.confidence)):.2f}",
            (6, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    stage = state.stage("Signal optimizer")
    _banner(board, stage.summary if stage else "", (150, 220, 255))
    return add_label(board, "13. Signal Optimizer - digitised leads")


def render_quality_analyzer(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 14: sub-scores as bars, plus the stated limitations."""
    state = _pipeline(result)
    width = max(560, processed.shape[1])
    board = np.full((260, width, 3), 245, dtype=np.uint8)
    if state is None:
        _banner(board, "No quality data", (80, 80, 255))
        return add_label(board, "14. Quality Analyzer")

    sub_scores: dict[str, float] = state.quality.get("sub_scores", {})  # type: ignore[assignment]
    y = 52
    for name, value in sorted(sub_scores.items()):
        cv2.putText(
            board, name, (12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA
        )
        x0, x1 = 140, width - 20
        cv2.rectangle(board, (x0, y), (x1, y + 14), (225, 225, 225), -1)
        filled = int(x0 + (x1 - x0) * float(np.clip(value, 0, 1)))
        color = (60, 200, 60) if value >= 0.7 else (0, 165, 255) if value >= 0.45 else (60, 60, 230)
        cv2.rectangle(board, (x0, y), (filled, y + 14), color, -1)
        cv2.putText(
            board,
            f"{value:.2f}",
            (x1 - 44, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 22

    limits = list(state.quality.get("limitations", ()) or ())
    lines = ["Limitations:", *[f"  - {item}" for item in limits]] if limits else ["No limitations flagged."]
    block = draw_text_block(lines, width, max(20, 260 - y - 4))
    board[y : y + block.shape[0]] = block
    _banner(
        board,
        f"grade={state.quality_grade}  overall={float(state.quality.get('probability', 0)):.2f}",
        (150, 220, 255),
    )
    return add_label(board, "14. Quality Analyzer")


def render_export(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Stage 15: the packaged payload, summarised."""
    state = _pipeline(result)
    width = max(640, processed.shape[1])
    if state is None:
        return add_label(np.full((160, width, 3), 245, dtype=np.uint8), "15. Export")

    payload = state.signals.get("export", {})
    lines = [
        f"schema: {payload.get('schema', 'n/a')}",
        f"grid: {payload.get('grid')}",
        f"layout: {payload.get('layout')}",
        f"paper: {payload.get('paper')}",
        f"rhythm: {payload.get('rhythm')}",
        f"quality: grade={state.quality_grade} p={float(state.quality.get('probability', 0)):.2f}",
        f"signals: {len(state.lead_signals)} leads in {state.signals.get('units', 'px')}",
        "",
        "Stage confidences:",
    ]
    lines += [f"  {stage.stage}: {stage.confidence} (p={stage.probability:.2f})" for stage in state.stages]
    board = draw_text_block(lines, width, 24 + 16 * (len(lines) + 2), start_y=34)
    _banner(board, "Export payload (JSON-serialisable)", (200, 255, 200))
    return add_label(board, "15. Export")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_STAGE_RENDERERS = {
    1: ("Image loader", render_image_loader),
    2: ("Geometry analyzer", render_geometry),
    3: ("Grid detector", render_grid),
    4: ("Layout detector", render_layout),
    5: ("OCR & semantic analysis", render_ocr),
    6: ("Trace probability estimator", render_trace_probability),
    7: ("Noise / artifact detector", render_noise_artifact),
    8: ("Cost map builder", render_cost_map),
    9: ("Path solver", render_path_solver),
    10: ("Gap recovery", render_gap_recovery),
    11: ("Beat detector", render_beat_detector),
    12: ("Template builder", render_template_builder),
    13: ("Signal optimizer", render_signal_optimizer),
    14: ("Quality analyzer", render_quality_analyzer),
    15: ("Export", render_export),
}


def render_stage(
    stage_index: int,
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Render a single pipeline stage as a labelled image panel.

    Parameters
    ----------
    stage_index:
        1-based stage number (1 ... 15).
    source:
        Original (pre-processed) BGR image.
    processed:
        Processed BGR image (the one in ``result.data``).
    result:
        ``ProcessedImage`` returned by ``process_ecg_image``.

    Returns
    -------
    BGR image suitable for ``cv2.imwrite``.
    """
    meta = _STAGE_RENDERERS.get(stage_index)
    if meta is None:
        h, w = processed.shape[:2]
        panel = np.full((h, w, 3), 32, dtype=np.uint8)
        cv2.putText(
            panel,
            f"Unknown stage {stage_index}",
            (12, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return panel
    return meta[1](source, processed, result)


def render_all_stages(
    source: np.ndarray,
    processed: np.ndarray,
    result: ProcessedImage,
    *,
    card_width: int = 800,
    stage_start: int = 1,
    stage_end: int = 15,
) -> np.ndarray:
    """Build a vertical board with all requested stage cards."""
    state = _pipeline(result)
    stages = state.stages if state else ()
    end = max(1, min(stage_end, 15))
    start = max(1, min(stage_start, end))

    cards: list[np.ndarray] = []
    for idx in range(start, end + 1):
        stage = stages[idx - 1] if len(stages) >= idx else None
        stage_name = stage.stage if stage else "pipeline unavailable"
        conf = stage.confidence if stage else "unknown"
        prob = stage.probability if stage else 0.0
        summary = stage.summary if stage else "No stage data."
        evidence = ", ".join(stage.evidence) if stage and stage.evidence else ""

        card = np.full((240, card_width, 3), 245, dtype=np.uint8)
        cv2.line(card, (0, 0), (card_width - 1, 0), (210, 210, 210), 2)
        cv2.putText(
            card,
            f"{idx:02d}. {stage_name} ({conf}, p={prob:.2f})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        lines = [summary] + ([f"evidence: {evidence}"] if evidence else [])
        for offset, line in enumerate(wrap_lines(lines, width_chars=72)):
            cv2.putText(
                card,
                line,
                (10, 54 + offset * 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        thumb = fit_to_size(render_stage(idx, source, processed, result), 350, 210)
        y = max(0, card.shape[0] - thumb.shape[0] - 6)
        x = card_width - thumb.shape[1] - 10
        card[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cards.append(card)

    return np.vstack(cards) if cards else np.zeros((0, card_width, 3), dtype=np.uint8)


def render_probability_maps_board(
    processed: np.ndarray,
    result: ProcessedImage,
) -> list[tuple[str, np.ndarray]]:
    """Render every class probability map as a red-overlay panel."""
    state = _pipeline(result)
    maps = state.probability_maps if state else {}
    panels: list[tuple[str, np.ndarray]] = []
    for name in CLASS_NAMES:
        panel = _probability_map_red(processed, maps.get(name), f"P({name})")
        panels.append((name, pad_to_square(panel)))
    return panels


def render_reconstruction_overlay(
    processed: np.ndarray,
    result: ProcessedImage,
) -> np.ndarray:
    """Every reconstructed path drawn over the sheet it came from.

    This is the single most useful check on the whole pipeline: if the coloured
    lines do not sit on the printed traces, nothing downstream is worth reading.
    """
    panel = processed.copy()
    state = _pipeline(result)
    if state is None or not state.paths:
        _banner(panel, "No reconstruction", (80, 80, 255))
        return panel
    for path in state.paths:
        color = PALETTE[path.index % len(PALETTE)]
        traced, offset = _reconstructed(state, path)
        _polyline(panel, traced, color, 1, offset)
    _banner(
        panel,
        f"{len(state.paths)} leads | grade={state.quality_grade} | "
        + (
            f"{state.beats.heart_rate_bpm:.0f} bpm"
            if state.beats is not None
            else "rate not measured"
        ),
        (140, 255, 140),
    )
    return panel
