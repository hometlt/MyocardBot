"""Stage 5 - OCR & semantic analysis.

MyocardBot2.md is explicit: text is never erased. This stage finds glyph-like
ink, labels it by where it sits on the sheet, and hands the result to the cost
map as a penalty. A lead label printed across a trace stays visible in the
image; it just becomes expensive for the path solver to walk through.

Detection runs in two steps, because a single blob is a weak hypothesis and a
*line* of blobs is a strong one. Individual glyph candidates are cheap to find
and full of false positives - the steep flank of a QRS looks much like a stroke
of type. Grouping them into rows of similar-height, evenly-spaced neighbours is
what separates the two: printed text always comes in lines, waveform flanks
never do. That grouping is also what makes the header and footer blocks
detectable as wholes.

The glyph pass is deliberately independent of the lead layout, so the runner can
call it *before* stage 4 and let the layout detector exclude text rows. Naming
which lead a label belongs to happens afterwards, once the bands exist.

Real character recognition is optional. When ``pytesseract`` happens to be
installed the detected lines are transcribed and the printed speed/gain legend
is read; without it the geometry still carries the semantics that matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.ink import InkLayers
from miocard.pipeline.layout import STANDARD_LEAD_ORDERS
from miocard.pipeline.state import (
    DEFAULT_GAIN_MM_MV,
    DEFAULT_PAPER_SPEED_MM_S,
    LeadBand,
    PipelineStageResult,
    TextObject,
    clamp01,
)

# Printed glyphs on an ECG sheet run from roughly 1.2 mm to 7 mm tall.
MIN_GLYPH_MM = 1.2
MAX_GLYPH_MM = 7.0
LEFT_MARGIN_FRACTION = 0.12
# A run of this many aligned glyphs is a printed line rather than a coincidence.
LINE_MIN_GLYPHS = 3
# Share of the sheet width a text line must cover before it is allowed to tell
# the layout detector that no lead baseline lives on those rows.
WIDE_TEXT_FRACTION = 0.25
SPEED_PATTERN = re.compile(r"(\d{1,3}(?:[.,]\d)?)\s*mm\s*/\s*s", re.IGNORECASE)
GAIN_PATTERN = re.compile(r"(\d{1,3}(?:[.,]\d)?)\s*mm\s*/\s*mv", re.IGNORECASE)


@dataclass(frozen=True)
class Glyph:
    left: int
    top: int
    right: int
    bottom: int
    area: int
    score: float

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def centre_y(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(frozen=True)
class TextLine:
    glyphs: tuple[Glyph, ...]
    left: int
    top: int
    right: int
    bottom: int
    probability: float

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


@dataclass(frozen=True)
class TextDetection:
    """Everything found before the lead layout is known."""

    lines: tuple[TextLine, ...]
    text_map: np.ndarray
    # The glyph mask itself, so the targeted lead-label search can work from
    # the same evidence without recomputing it.
    glyph_mask: np.ndarray
    # Share of each image row covered by a page-wide block of type. The layout
    # detector combines this with its own isoelectric evidence rather than
    # taking it as a veto, because a header can sit across the first lead.
    row_text_coverage: np.ndarray
    glyph_count: int = 0


def _layout_label_lines(
    detection: TextDetection,
    image_width: int,
    px_per_mm: float,
) -> tuple[TextLine, ...]:
    """Lead-name-sized text lines that may anchor the row layout.

    This deliberately does not assign names.  Before the lead bands exist the
    only safe statement is that a short, glyph-sized line in the left margin is
    a possible row anchor.  Header sentences are too wide, page borders start
    too close to the edge, and calibration strokes are too tall.  The layout
    stage accepts these points only when an ordered subset fits a smooth
    twelve-row model, so false positives remain harmless evidence rather than a
    veto.
    """
    if image_width < 16 or px_per_mm <= 1.0:
        return ()

    candidates: list[TextLine] = []
    for line in detection.lines:
        width = line.right - line.left
        height = line.bottom - line.top
        if line.right > 0.25 * image_width:
            continue
        if line.left < 0.035 * image_width:
            continue
        if not px_per_mm <= height <= 5.5 * px_per_mm:
            continue
        if width > 9.0 * px_per_mm or line.probability < 0.18:
            continue
        candidates.append(line)

    return tuple(sorted(candidates, key=lambda line: (line.top, line.left)))


def layout_label_anchors(
    detection: TextDetection,
    image_width: int,
    px_per_mm: float,
) -> np.ndarray:
    """Centres of possible lead labels for the row-layout stage."""
    candidates = _layout_label_lines(detection, image_width, px_per_mm)
    if not candidates:
        return np.zeros(0, dtype=np.float32)

    return np.asarray(
        [(line.top + line.bottom) / 2.0 for line in candidates],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class SemanticReport:
    objects: tuple[TextObject, ...]
    text_map: np.ndarray
    paper_speed_mm_s: float
    gain_mm_mv: float
    gain_measured: bool
    ocr_used: bool
    probability: float
    labels_found: int = 0
    line_count: int = 0
    # Right edge of the left-margin furniture (name, calibration step) per row.
    label_columns: dict[int, int] = field(default_factory=dict)
    labels_expected: int = 0
    calibration_marks: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Glyph candidates
# ---------------------------------------------------------------------------


def _glyph_mask(layers: InkLayers, px_per_mm: float) -> np.ndarray:
    """Strong ink with the long horizontal pen strokes removed.

    The trace is one continuous horizontal structure, so opening the mask with a
    wide horizontal kernel isolates it; what survives the subtraction is glyphs,
    calibration steps and the steep flanks of the QRS.
    """
    mask = (layers.strong > 0.5).astype(np.uint8)
    span = max(9, int(round(4.0 * px_per_mm)))
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1)),
    )
    remainder = cv2.subtract(mask, horizontal)
    # Reconnect glyph strokes that the subtraction nicked.
    return cv2.morphologyEx(
        remainder,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def _glyph_score(glyph_width: int, glyph_height: int, area: int, px_per_mm: float) -> float:
    """How much a blob looks like printed type, before any grouping."""
    height_mm = glyph_height / max(px_per_mm, 1e-6)
    if not MIN_GLYPH_MM <= height_mm <= MAX_GLYPH_MM:
        return 0.0
    # A character is about as tall as it is wide. A fragment of flat trace,
    # chopped into pieces where it crosses the grid, is far wider than it is
    # tall - and a row of those fragments would otherwise group into a
    # convincing "text line" straight along a lead baseline.
    aspect = glyph_width / max(glyph_height, 1)
    if not 0.12 <= aspect <= 2.0:
        return 0.0
    fill = area / max(glyph_width * glyph_height, 1)
    if fill < 0.10:
        return 0.0
    fill_score = clamp01((fill - 0.10) / 0.45)
    size_score = clamp01(1.0 - abs(height_mm - 2.6) / 4.0)
    return clamp01(0.25 + 0.45 * fill_score + 0.30 * size_score)


def find_glyphs(layers: InkLayers, px_per_mm: float, config: PipelineConfig) -> list[Glyph]:
    mask = _glyph_mask(layers, px_per_mm)
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    max_area = max(config.min_text_area_px, int(config.max_text_area_fraction * height * width))

    glyphs: list[Glyph] = []
    for index in range(1, count):
        left, top, box_width, box_height, area = (int(value) for value in stats[index])
        if area < config.min_text_area_px or area > max_area:
            continue
        score = _glyph_score(box_width, box_height, area, px_per_mm)
        if score <= 0.0:
            continue
        glyphs.append(Glyph(left, top, left + box_width, top + box_height, area, score))
    return glyphs


# ---------------------------------------------------------------------------
# Grouping into printed lines
# ---------------------------------------------------------------------------


def group_text_lines(glyphs: list[Glyph]) -> list[TextLine]:
    """Chain glyphs that sit on a common baseline into printed lines.

    Two glyphs belong to the same line when their heights are comparable, their
    centres are vertically close relative to that height, and the horizontal gap
    between them is no wider than a couple of characters. That is exactly the
    structure a run of QRS flanks lacks: those repeat at the beat interval,
    which is far wider than any letter spacing.
    """
    if not glyphs:
        return []
    ordered = sorted(glyphs, key=lambda glyph: (glyph.left, glyph.top))
    used = [False] * len(ordered)
    lines: list[TextLine] = []

    for index, glyph in enumerate(ordered):
        if used[index]:
            continue
        chain = [glyph]
        used[index] = True
        current = glyph
        for other_index in range(index + 1, len(ordered)):
            if used[other_index]:
                continue
            other = ordered[other_index]
            reference = max(current.height, other.height)
            if min(current.height, other.height) < 0.45 * reference:
                continue
            if abs(other.centre_y - current.centre_y) > 0.55 * reference:
                continue
            gap = other.left - current.right
            if gap < -reference or gap > 2.2 * reference:
                continue
            chain.append(other)
            used[other_index] = True
            current = other

        lines.append(_line_from(chain))
    return _merge_into_rows(lines)


def _line_from(glyphs: list[Glyph]) -> TextLine:
    mean_score = float(np.mean([item.score for item in glyphs]))
    # A long line of aligned type is near-certain; a lone blob is a guess.
    support = clamp01((len(glyphs) - 1) / (LINE_MIN_GLYPHS - 1))
    return TextLine(
        tuple(glyphs),
        min(item.left for item in glyphs),
        min(item.top for item in glyphs),
        max(item.right for item in glyphs),
        max(item.bottom for item in glyphs),
        clamp01(mean_score * (0.35 + 0.65 * support)),
    )


def _merge_into_rows(words: list[TextLine]) -> list[TextLine]:
    """Join words that share a printed row into one line.

    The first pass chains glyphs across letter spacing, which breaks at the far
    wider gap between words. A header reads as a dozen separate fragments until
    they are merged back, and it is the merged extent - not a single word - that
    tells the layout detector a whole image row is given over to type.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda line: ((line.top + line.bottom) / 2, line.left))
    merged: list[list[TextLine]] = []
    for word in ordered:
        centre = (word.top + word.bottom) / 2
        for group in merged:
            reference = max(max(item.height for item in group), word.height)
            group_centre = float(np.mean([(item.top + item.bottom) / 2 for item in group]))
            if abs(centre - group_centre) > 0.55 * reference:
                continue
            gap = min(
                abs(word.left - item.right) if word.left >= item.right else 0
                for item in group
            )
            # Word spacing is wide but bounded; a gap of several characters
            # still belongs to the same printed line.
            if gap > 6.0 * reference:
                continue
            group.append(word)
            break
        else:
            merged.append([word])

    return [
        _line_from([glyph for item in group for glyph in item.glyphs]) for group in merged
    ]


def detect_text(
    layers: InkLayers,
    px_per_mm: float,
    config: PipelineConfig,
) -> TextDetection:
    """Find printed text without needing the lead layout."""
    height, width = layers.shape
    text_map = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros(height, dtype=np.float32)
    if px_per_mm <= 1:
        # Without a millimetre scale the glyph size test is meaningless.
        empty = np.zeros((height, width), dtype=np.uint8)
        return TextDetection((), text_map, empty, coverage, 0)

    mask = _glyph_mask(layers, px_per_mm)
    glyphs = find_glyphs(layers, px_per_mm, config)
    lines = group_text_lines(glyphs)

    for line in lines:
        if line.probability < 0.30:
            continue
        for glyph in line.glyphs:
            region = text_map[glyph.top : glyph.bottom, glyph.left : glyph.right]
            np.maximum(region, line.probability, out=region)
        # Only a line of type that runs across the sheet counts towards the
        # row coverage the layout detector consults. A lead label sits *on* its
        # own baseline, and its handful of characters must not be able to
        # suggest that the row is given over to type.
        span = (line.right - line.left) / max(width, 1)
        if span >= WIDE_TEXT_FRACTION and len(line.glyphs) >= LINE_MIN_GLYPHS:
            rows = coverage[line.top : line.bottom]
            np.maximum(rows, min(1.0, span) * line.probability, out=rows)

    # Soften the map so a glyph penalises its immediate surroundings too; a
    # trace grazing the edge of a character is just as ambiguous.
    text_map = cv2.GaussianBlur(text_map, (0, 0), max(1.0, px_per_mm / 4.0))
    text_map = np.clip(text_map, 0.0, 1.0).astype(np.float32)
    return TextDetection(tuple(lines), text_map, mask, coverage, len(glyphs))


# ---------------------------------------------------------------------------
# Semantics, once the lead bands are known
# ---------------------------------------------------------------------------


def assign_lead_labels(
    mask: np.ndarray,
    bands: tuple[LeadBand, ...],
    px_per_mm: float,
) -> dict[int, TextLine]:
    """Locate the printed name of every lead row.

    The names themselves are known - a recorder always prints the same leads in
    the same order - so this is a search, not a reading task, and it can be far
    more aggressive than the general glyph pass. Exactly one label is expected
    per row, in the left margin, sitting on that row's baseline. With a prior
    that strong the only real question is which ink in that small window belongs
    to the name, so faint or broken characters that the page-wide pass discards
    are still accepted here.

    The general pass cannot find these reliably because a lead label is printed
    *on top of* its own trace: removing long horizontal strokes to isolate type
    also cuts through the characters sitting on them.
    """
    if not bands or px_per_mm <= 1:
        return {}
    height, width = mask.shape
    margin = int(width * LEFT_MARGIN_FRACTION)
    if margin < 4:
        return {}

    minimum = max(4, int(round(MIN_GLYPH_MM * px_per_mm * 0.5)))
    maximum = int(round(MAX_GLYPH_MM * px_per_mm))
    assigned: dict[int, TextLine] = {}
    for band in bands:
        top = int(max(0, band.baseline - band.height * 0.55))
        bottom = int(min(height, band.baseline + band.height * 0.55))
        window = mask[top:bottom, :margin]
        if window.size == 0:
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(window, 8)
        glyphs: list[Glyph] = []
        for index in range(1, count):
            left, box_top, box_width, box_height, area = (int(v) for v in stats[index])
            if not minimum <= box_height <= maximum:
                continue
            if box_width > maximum * 1.5 or area < 6:
                continue
            glyphs.append(
                Glyph(
                    left,
                    top + box_top,
                    left + box_width,
                    top + box_top + box_height,
                    area,
                    0.6,
                )
            )
        if not glyphs:
            continue
        # A name is one to three characters standing together; keep the cluster
        # nearest the baseline rather than every mark in the margin.
        tallest = max(glyphs, key=lambda glyph: glyph.height)
        cluster = [
            glyph
            for glyph in glyphs
            if abs(glyph.centre_y - tallest.centre_y) <= tallest.height * 0.7
            and glyph.left <= tallest.right + tallest.height * 2.5
        ]
        assigned[band.index] = _line_from(cluster)
    return assigned


def _longest_runs(window: np.ndarray) -> np.ndarray:
    """Longest contiguous vertical ink run in each column of a window."""
    best = np.zeros(window.shape[1], dtype=np.int32)
    run = np.zeros(window.shape[1], dtype=np.int32)
    for row in range(window.shape[0]):
        run = np.where(window[row] > 0, run + 1, 0)
        np.maximum(best, run, out=best)
    return best


def find_calibration_pulses(
    layers: InkLayers,
    bands: tuple[LeadBand, ...],
    px_per_mm: float,
) -> list[tuple[int, int, int, int, float]]:
    """Locate the rectangular 1 mV reference step printed before a lead.

    The mark is a "П": the pen jumps up, holds level for a couple of
    millimetres, and drops back. That shape - two tall vertical strokes of
    matching height joined by a flat top - is what distinguishes it from a QRS
    flank or a letter, and checking for the whole shape rather than for one tall
    column is what makes the find reliable.

    Returns ``(left, top, right, bottom, height_mm)`` per pulse.

    The measured height is reported but deliberately not used to set the gain.
    At the few pixels per millimetre a photograph of a whole sheet provides, the
    shape test still admits the occasional square-shouldered QRS, and the
    heights that come back do not reproduce the printed 1 mV reliably enough to
    scale every millivolt by. What the find *is* good for is unconditional: the
    mark is not part of any lead's trace, so the path solver is told to stay off
    it.
    """
    if px_per_mm <= 1 or not bands:
        return []
    mask = (layers.strong > 0.5).astype(np.uint8)
    height, width = mask.shape
    margin = int(width * LEFT_MARGIN_FRACTION)
    if margin < 6:
        return []

    min_run = max(4, int(round(4.0 * px_per_mm)))
    min_gap = max(2, int(round(0.5 * px_per_mm)))
    max_gap = max(min_gap + 2, int(round(4.0 * px_per_mm)))

    pulses: list[tuple[int, int, int, int, float]] = []
    for band in bands:
        top = int(max(0, band.baseline - band.height * 1.3))
        bottom = int(min(height, band.baseline + band.height * 1.3))
        window = mask[top:bottom, :margin]
        if window.size == 0:
            continue
        runs = _longest_runs(window)
        tall = np.flatnonzero(runs >= min_run)
        if tall.size < 2:
            continue

        # How far the flat top may wander and still count as level.
        level_tolerance = max(1, int(round(0.7 * px_per_mm)))
        found = None
        for rising in tall:
            for falling in tall:
                gap = int(falling) - int(rising)
                if not min_gap <= gap <= max_gap:
                    continue
                left_run, right_run = float(runs[rising]), float(runs[falling])
                if min(left_run, right_run) < 0.7 * max(left_run, right_run):
                    continue
                between = window[:, int(rising) : int(falling) + 1]
                if not bool(between.any(axis=0).all()):
                    continue
                # The flat top is the whole test. Every column between the two
                # strokes must have its highest ink at the same level; the apex
                # of a QRS falls away immediately either side of the peak and
                # cannot hold a level roof across a millimetre of paper.
                highest = np.argmax(between, axis=0)
                if int(highest.max() - highest.min()) > level_tolerance:
                    continue
                span = max(left_run, right_run)
                if found is None or span > found[0]:
                    found = (span, int(rising), int(falling))
        if found is None:
            continue
        span, rising, falling = found
        column = window[:, rising : falling + 1].any(axis=1)
        rows = np.flatnonzero(column)
        pulses.append(
            (
                int(rising),
                int(top + rows[0]),
                int(falling) + 1,
                int(top + rows[-1]) + 1,
                float(span) / px_per_mm,
            )
        )
    return pulses


def _margin_edges(
    labels: dict[int, TextLine],
    pulses: list[tuple[int, int, int, int, float]],
    bands: tuple[LeadBand, ...],
) -> dict[int, int]:
    """Right edge of the calibration step, per lead row.

    The step is a printed square wave standing on the trace; a solver that
    starts inside it reads a millivolt of signal that the heart never produced,
    so the trace is taken to begin after it.

    The lead's printed *name* is deliberately not treated the same way. It is
    printed over a trace that is already running - measured on
    examples/photo_293, where the recording is visible from x=75 while the names
    end around x=153 - so skipping to the end of the name threw away the first
    0.6 s of eight of the twelve leads. A name is something to read through, not
    a place to start.
    """
    edges: dict[int, int] = {}
    for _left, top, right, bottom, _height in pulses:
        centre = (top + bottom) / 2.0
        near = [
            band
            for band in bands
            if abs(band.baseline - centre) <= band.height * 0.75
        ]
        for band in near:
            edges[band.index] = max(edges.get(band.index, 0), int(right))
    return edges


def _classify(line: TextLine, bands: tuple[LeadBand, ...], width: int) -> tuple[str, int | None]:
    centre_y = (line.top + line.bottom) / 2.0
    if bands:
        first_top = min(band.top for band in bands)
        last_bottom = max(band.bottom for band in bands)
        if line.bottom <= first_top:
            return "header", None
        if line.top >= last_bottom:
            return "footer", None
        nearest = min(bands, key=lambda band: abs(band.baseline - centre_y))
        if line.left <= width * LEFT_MARGIN_FRACTION and len(line.glyphs) <= 4:
            return "lead_label", nearest.index
        return "annotation", nearest.index
    if line.left <= width * LEFT_MARGIN_FRACTION:
        return "lead_label", None
    return "annotation", None


def _try_ocr(image: np.ndarray, objects: list[TextObject]) -> tuple[list[TextObject], bool]:
    try:  # pragma: no cover - exercised only where tesseract is installed
        import pytesseract
    except Exception:
        return objects, False
    try:  # pragma: no cover
        transcribed: list[TextObject] = []
        for obj in objects:
            patch = image[obj.top : obj.bottom, obj.left : obj.right]
            if patch.size == 0:
                transcribed.append(obj)
                continue
            text = pytesseract.image_to_string(patch, config="--psm 7").strip()
            transcribed.append(
                TextObject(
                    obj.left,
                    obj.top,
                    obj.right,
                    obj.bottom,
                    obj.kind,
                    obj.probability,
                    text=text,
                    band_index=obj.band_index,
                )
            )
        return transcribed, True
    except Exception:  # pragma: no cover - a broken tesseract must not break analysis
        return objects, False


def analyze_text(
    image: np.ndarray,
    detection: TextDetection,
    bands: tuple[LeadBand, ...],
    px_per_mm: float,
    config: PipelineConfig,
    layers: InkLayers | None = None,
) -> SemanticReport:
    width = detection.text_map.shape[1]
    notes: list[str] = []

    labels = assign_lead_labels(detection.glyph_mask, bands, px_per_mm)
    # The targeted mask search and the page-wide line detector fail in
    # complementary places.  In particular, subtracting the long trace can
    # split a label printed directly over it, while the intact glyph cluster is
    # still a strong row anchor.  Reuse exactly the candidates accepted by the
    # layout stage and attach each to at most one nearby band.
    for line in _layout_label_lines(detection, width, px_per_mm):
        centre = (line.top + line.bottom) / 2.0
        nearest = min(bands, key=lambda band: abs(band.baseline - centre), default=None)
        if nearest is None or abs(nearest.baseline - centre) > nearest.height * 0.55:
            continue
        previous = labels.get(nearest.index)
        if previous is None or line.probability > previous.probability:
            labels[nearest.index] = line
    pulses = find_calibration_pulses(layers, bands, px_per_mm) if layers is not None else []
    names = STANDARD_LEAD_ORDERS.get(len(bands), ())
    margin = width * LEFT_MARGIN_FRACTION

    objects: list[TextObject] = [
        TextObject(
            line.left,
            line.top,
            line.right,
            line.bottom,
            "lead_label",
            line.probability,
            # The names are known from the print order; this stage locates them.
            text=names[index] if index < len(names) else "",
            band_index=index,
        )
        for index, line in sorted(labels.items())
    ]
    text_map = detection.text_map.copy()
    for left, top, right, bottom, height_mm in pulses:
        objects.append(
            TextObject(
                left,
                top,
                right,
                bottom,
                "calibration",
                0.8,
                text=f"{height_mm:.1f} mm step",
            )
        )
        # Penalise it like type: it is ink, but no lead drew it.
        region = text_map[top:bottom, left:right]
        np.maximum(region, 0.8, out=region)
    for line in labels.values():
        region = text_map[line.top : line.bottom, line.left : line.right]
        np.maximum(region, line.probability, out=region)

    for line in detection.lines:
        if line.probability < 0.30:
            continue
        # The margin belongs to the targeted label search above.
        if line.left <= margin and line.right <= margin:
            continue
        kind, band_index = _classify(line, bands, width)
        objects.append(
            TextObject(
                line.left,
                line.top,
                line.right,
                line.bottom,
                kind,
                line.probability,
                band_index=band_index,
            )
        )

    # The calibration pulse is not measured here on purpose. At the few pixels
    # per millimetre a phone photo of a whole sheet gives, its vertical stroke
    # is indistinguishable from a lead label or the flank of a QRS, and a gain
    # guessed wrong scales every millivolt downstream without any sign that it
    # did. The printed legend read by OCR is trusted; otherwise the gain is
    # declared an assumption.
    gain_mm_mv = config.gain_mm_mv
    gain_measured = False

    objects, ocr_used = _try_ocr(image, objects)
    paper_speed = config.paper_speed_mm_s
    if ocr_used:
        joined = " ".join(obj.text for obj in objects if obj.text)
        speed_match = SPEED_PATTERN.search(joined)
        if speed_match:
            paper_speed = float(speed_match.group(1).replace(",", "."))
            notes.append("speed_from_legend")
        gain_match = GAIN_PATTERN.search(joined)
        if gain_match:
            gain_mm_mv = float(gain_match.group(1).replace(",", "."))
            gain_measured = True
            notes.append("gain_from_legend")
    if not gain_measured:
        notes.append("gain_assumed_standard")
    if "speed_from_legend" not in notes:
        notes.append("speed_assumed_standard")

    multi = sum(1 for line in detection.lines if len(line.glyphs) >= LINE_MIN_GLYPHS)
    found = len(labels)
    probability = clamp01(
        0.15
        + 0.25 * clamp01(multi / 6.0)
        + 0.40 * clamp01(found / max(len(bands), 1))
        + (0.20 if ocr_used else 0.0)
    )
    return SemanticReport(
        objects=tuple(objects),
        text_map=np.clip(text_map, 0.0, 1.0).astype(np.float32),
        paper_speed_mm_s=paper_speed if paper_speed > 0 else DEFAULT_PAPER_SPEED_MM_S,
        gain_mm_mv=gain_mm_mv if gain_mm_mv > 0 else DEFAULT_GAIN_MM_MV,
        gain_measured=gain_measured,
        ocr_used=ocr_used,
        probability=probability,
        labels_found=len(labels),
        line_count=len(detection.lines),
        label_columns=_margin_edges(labels, pulses, bands),
        calibration_marks=len(pulses),
        labels_expected=len(bands),
        notes=tuple(notes),
    )


def text_stage(report: SemanticReport) -> PipelineStageResult:
    kinds: dict[str, int] = {}
    for obj in report.objects:
        kinds[obj.kind] = kinds.get(obj.kind, 0) + 1
    breakdown = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items())) or "none"
    return PipelineStageResult(
        "OCR & semantic analysis",
        report.probability,
        f"{len(report.objects)} text lines kept as cost-map penalties ({breakdown}); "
        "nothing was erased from the image.",
        evidence=(
            f"ocr_engine={'pytesseract' if report.ocr_used else 'geometry_only'}",
            f"lead_labels={report.labels_found}/{report.labels_expected}",
            f"calibration_marks={report.calibration_marks}",
            f"text_lines={report.line_count}",
            f"gain_mm_mv={report.gain_mm_mv:.1f}"
            + ("" if report.gain_measured else " (assumed, not measured)"),
            f"paper_speed_mm_s={report.paper_speed_mm_s:.1f}",
        ),
    )
