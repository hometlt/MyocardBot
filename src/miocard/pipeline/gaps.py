"""Stage 10 - Gap recovery.

Where the image stops supporting the path - the pen skipped, a lead label sits
on top of the trace, glare washed the paper out - the solver still returns its
best guess, but that guess is inference rather than measurement. This stage says
which is which, fills short gaps with a shape-preserving interpolation, and
refuses to invent anything longer than a fraction of a beat.

The reason for each gap is read back off the cost map, so a break caused by text
is reported as text rather than as missing ink.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.series import effective_trace
from miocard.pipeline.state import LeadPath, PipelineStageResult, TextObject, clamp01

REASONS = ("text", "artifact", "noise", "collision", "missing_ink")
SEMANTIC_TEXT_KINDS = frozenset({"header", "annotation", "lead_label"})
SEMANTIC_TEXT_MIN_PROBABILITY = 0.30


@dataclass(frozen=True)
class GapReport:
    probability: float
    statistics: dict[str, float]
    gaps_by_reason: dict[str, int]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) spans of consecutive True values."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, changes.size, 2)]


def _hermite_fill(y: np.ndarray, start: int, end: int) -> None:
    """Catmull-Rom fill between the reliable samples either side of a gap.

    Linear interpolation would put a corner at each end of the gap, which the
    beat detector then reads as a deflection; matching the slope of the
    surrounding trace avoids inventing that feature.
    """
    left = start - 1
    right = end
    if left < 0 or right >= y.size:
        # Nothing to anchor against on one side: hold the nearest value.
        fill = y[right] if left < 0 else y[left]
        y[start:end] = fill
        return
    p1, p2 = float(y[left]), float(y[right])
    p0 = float(y[max(0, left - max(1, (end - start) // 2))])
    p3 = float(y[min(y.size - 1, right + max(1, (end - start) // 2))])
    m1 = 0.5 * (p2 - p0)
    m2 = 0.5 * (p3 - p1)
    steps = np.linspace(0.0, 1.0, end - start + 2, dtype=np.float32)[1:-1]
    t2 = steps * steps
    t3 = t2 * steps
    y[start:end] = (
        (2 * t3 - 3 * t2 + 1) * p1
        + (t3 - 2 * t2 + steps) * m1
        + (-2 * t3 + 3 * t2) * p2
        + (t3 - t2) * m2
    ).astype(y.dtype)


def _gap_reason(
    classes: dict[str, np.ndarray],
    y: np.ndarray,
    start: int,
    end: int,
) -> str:
    rows = np.clip(np.round(y[start:end]).astype(np.int32), 0, classes["text"].shape[0] - 1)
    columns = np.arange(start, end)
    scores = {
        "text": float(classes["text"][rows, columns].mean()),
        "artifact": float(classes["artifact"][rows, columns].mean()),
        "noise": float(classes["noise"][rows, columns].mean()),
    }
    dominant = max(scores, key=scores.get)
    return dominant if scores[dominant] >= 0.12 else "missing_ink"


def _semantic_text_mask(
    path: LeadPath,
    objects: tuple[TextObject, ...],
    px_per_mm: float,
    last_band_index: int,
) -> np.ndarray:
    """Columns where a recognised text region physically covers this path.

    Pixel-wise P(text) only marks the strokes of letters.  Between those
    strokes a shortest path can still snake through the word and be reported
    as measured ECG.  A detected line of print is an ambiguous *region*: if the
    reconstructed pen lies inside its bounding box, the photograph cannot say
    which ink came from the recorder and which came from the printer.
    """
    obscured = np.zeros(path.width, dtype=bool)
    if not objects:
        return obscured
    pad = max(1, int(round(0.35 * px_per_mm))) if px_per_mm > 1 else 1
    for obj in objects:
        if obj.kind not in SEMANTIC_TEXT_KINDS:
            continue
        if obj.probability < SEMANTIC_TEXT_MIN_PROBABILITY:
            continue
        if obj.band_index is not None and obj.band_index != path.index:
            continue
        box_width = obj.right - obj.left
        box_height = obj.bottom - obj.top
        # Targeted label search can accidentally cluster the QRS complexes of
        # a whole lead, and the generic line grouper can call a low-amplitude
        # interior trace an annotation.  Those false semantic boxes are much
        # wider than real lead names or short annotations.  Broad text is kept
        # only on the first/last rows, where this recorder actually overlays
        # the header and footer across the waveform.
        label_width = 9.0 * px_per_mm if px_per_mm > 1 else 40.0
        label_height = 5.5 * px_per_mm if px_per_mm > 1 else 24.0
        annotation_width = 18.0 * px_per_mm if px_per_mm > 1 else 80.0
        if obj.kind == "lead_label" and (
            box_width > label_width or box_height > label_height
        ):
            continue
        if (
            obj.kind == "annotation"
            and obj.band_index not in {0, last_band_index}
            and box_width > annotation_width
        ):
            continue
        left = int(np.clip(obj.left - pad, 0, path.width))
        right = int(np.clip(obj.right + pad, 0, path.width))
        if right <= left:
            continue
        rows = path.y[left:right]
        obscured[left:right] |= (rows >= obj.top - pad) & (rows < obj.bottom + pad)
    return obscured


def recover_gaps(
    paths: tuple[LeadPath, ...],
    classes: dict[str, np.ndarray],
    px_per_mm: float,
    config: PipelineConfig,
    text_objects: tuple[TextObject, ...] = (),
) -> GapReport:
    if not paths:
        return GapReport(0.05, {"gap_fraction": 1.0, "recovered_fraction": 0.0}, {})

    samples_per_second = max(px_per_mm * config.paper_speed_mm_s, 1.0)
    max_gap_px = int(round(config.max_recovered_gap_seconds * samples_per_second))
    reasons: dict[str, int] = dict.fromkeys(REASONS, 0)

    weak_total = 0
    recovered_total = 0
    measured_total = 0
    height = classes["background"].shape[0]
    for path in paths:
        width = path.width
        confidence = cv2.GaussianBlur(
            path.evidence.reshape(1, -1).astype(np.float32), (5, 1), 0
        ).ravel()
        rows = np.clip(np.round(path.y).astype(np.int32), 0, height - 1)
        columns = np.arange(width)
        interference = np.maximum(
            classes["text"][rows, columns],
            classes["artifact"][rows, columns],
        )
        # Dark ink is not automatically ECG ink.  The former implementation
        # used 1-P(background)-P(grid), so a printed glyph was treated as direct
        # measurement precisely when P(text) was high.  Keep the trace evidence
        # instead and discount competing explanations.
        observable = confidence * (1.0 - interference.astype(np.float32))
        obscured_by_text = _semantic_text_mask(
            path,
            text_objects,
            px_per_mm,
            len(paths) - 1,
        )
        observable[obscured_by_text] = 0.0
        inside = np.zeros(width, dtype=bool)
        inside[path.x_start : path.x_end] = True
        weak = (observable < config.gap_evidence_threshold) & inside
        recovered = np.zeros(width, dtype=bool)

        gaps: list[tuple[int, int, str]] = []
        for start, end in _runs(weak):
            reason = (
                "text"
                if bool(np.mean(obscured_by_text[start:end]) >= 0.5)
                else _gap_reason(classes, path.y, start, end)
            )
            reasons[reason] = reasons.get(reason, 0) + 1
            if end - start <= max_gap_px:
                _hermite_fill(path.y, start, end)
                recovered[start:end] = True
                # Interpolated samples are inference: they keep a floor of
                # confidence rather than the measured value of zero.
                confidence[start:end] = np.maximum(confidence[start:end], 0.22)
            gaps.append((start, end, reason))

        # Confidence describes direct observability, not merely darkness.
        confidence = np.minimum(confidence, observable)
        confidence[recovered] = np.maximum(confidence[recovered], 0.22)
        path.confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
        path.recovered = recovered
        path.unreadable = weak
        # Recovered samples are an interpolation of the centre line.  Do not
        # let stage 13 resurrect the old ink envelope and silently replace that
        # interpolation with the very glyph/artifact that was rejected here.
        path.top_envelope[recovered] = path.y[recovered]
        path.bottom_envelope[recovered] = path.y[recovered]
        path.gaps = tuple(gaps)
        weak_total += int(weak.sum())
        recovered_total += int(recovered.sum())
        measured_total += int(inside.sum())

    # At a geometric crossing the photograph contains one connected ink mass
    # and cannot establish which endpoint belongs to which lead.  Keep the
    # one-value-per-column estimates, but do not present either estimate as a
    # direct measurement.  This is deliberately a post-pass: independently
    # interpolating both traces at the crossing would merely invent two smooth
    # curves from the same ambiguous pixels.
    collision_added = 0
    collision_gaps = 0
    smooth_px = max(3, int(round(px_per_mm))) if px_per_mm > 1 else 3
    ordered = sorted(paths, key=lambda item: item.band.baseline)
    for upper, lower in zip(ordered, ordered[1:], strict=False):
        start = max(upper.x_start, lower.x_start)
        end = min(upper.x_end, lower.x_end)
        if end <= start:
            continue
        upper_rows = effective_trace(upper, smooth_px)[start:end]
        lower_rows = effective_trace(lower, smooth_px)[start:end]
        collision = lower_rows - upper_rows < 2.0
        if collision.any():
            collision = cv2.dilate(
                collision.reshape(1, -1).astype(np.uint8),
                np.ones((1, 5), dtype=np.uint8),
            ).ravel().astype(bool)
        for path in (upper, lower):
            if path.unreadable is None or path.confidence is None:
                continue
            new = collision & ~path.unreadable[start:end]
            if not new.any():
                continue
            path.unreadable[start:end] |= collision
            path.confidence[start:end][collision] = np.minimum(
                path.confidence[start:end][collision],
                0.10,
            )
            additions = [
                (start + left, start + right, "collision")
                for left, right in _runs(new)
            ]
            path.gaps = tuple(sorted((*path.gaps, *additions)))
            collision_added += int(new.sum())
            collision_gaps += len(additions)

    if collision_added:
        weak_total += collision_added
        reasons["collision"] += collision_gaps

    gap_fraction = weak_total / max(measured_total, 1)
    recovered_fraction = recovered_total / max(weak_total, 1)
    statistics = {
        "gap_fraction": float(gap_fraction),
        "recovered_fraction": float(recovered_fraction),
        "max_recovered_gap_px": float(max_gap_px),
        "gap_count": float(sum(len(path.gaps) for path in paths)),
    }
    probability = clamp01(
        0.15 + 0.60 * (1.0 - clamp01(gap_fraction * 3.0)) + 0.25 * recovered_fraction
    )
    return GapReport(probability, statistics, reasons)


def gap_stage(report: GapReport) -> PipelineStageResult:
    breakdown = ", ".join(
        f"{reason}={count}" for reason, count in sorted(report.gaps_by_reason.items()) if count
    )
    return PipelineStageResult(
        "Gap recovery",
        report.probability,
        f"{report.statistics['gap_fraction'] * 100:.1f}% of the traced width lacked direct "
        f"evidence; {report.statistics['recovered_fraction'] * 100:.0f}% of that was "
        "interpolated and the rest left flagged.",
        evidence=(
            f"gap_fraction={report.statistics['gap_fraction']:.3f}",
            f"recovered_fraction={report.statistics['recovered_fraction']:.3f}",
            f"gap_causes={breakdown or 'none'}",
        ),
    )
