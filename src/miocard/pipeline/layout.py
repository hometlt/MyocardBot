"""Stage 4 - Layout detector.

A printed ECG puts one lead per horizontal strip, and the isoelectric baseline
of each lead is where its pen spends most of its horizontal travel. Finding
those baselines is what the rest of the pipeline hangs off: the path solver
needs to know which strip belongs to which lead, or it will happily adopt a
neighbour's trace when two waveforms overlap.

The detector works by search rather than by fitting one rigid comb:

1. Every plausible row pitch is proposed in turn. The autocorrelation of the row
   profile is reported as a cross-check but not trusted as the answer - only a
   dozen periods fit on a sheet, and heavily overlapping leads pull that
   estimate long, which makes the tracker step straight over a row.
2. For each pitch, baselines are *tracked* outward from the strongest one,
   predicting the next at the running spacing and snapping to the nearest real
   peak. Tracking rather than fitting is what survives a curled or
   perspective-distorted page, where the spacing drifts and no single pitch fits.
3. The tracked set is trimmed: rows lying under a page-wide block of text are
   dropped, and overshoot past the end of the print is cut back.
4. Each hypothesis is scored on the *total* ink standing on its rows, less a
   fixed cost per row claimed. Scoring the total rather than the average is what
   stops a sparse set from winning by sitting on the few strongest baselines: a
   hypothesis that skips a real lead forfeits that lead's ink.

How much horizontally-continuous ink a baseline carries is scored but never used
to reject one, because a lead at a high heart rate has almost no flat segment.
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.ink import InkLayers, row_ink_profile
from miocard.pipeline.state import LeadBand, PipelineStageResult, clamp01

# Print orders that a 12-lead recorder actually uses. Assigning names is an
# assumption and is reported as one, but it is the assumption every one of these
# machines makes.
STANDARD_LEAD_ORDERS: dict[int, tuple[str, ...]] = {
    3: ("I", "II", "III"),
    6: ("I", "II", "III", "aVR", "aVL", "aVF"),
    12: (
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ),
}

# What a claimed row must be worth in ink before it earns its place, as a
# fraction of the strongest row on the sheet. Too low and the search invents
# rows between the real ones; too high and it settles for the few strongest
# baselines and steps over the rest.
ROW_COST = 0.15
# The near-universal printed layout, given a slight benefit of the doubt when
# two hypotheses otherwise score alike. Deliberately excludes 3 and 6: those are
# exactly what a doubled or trebled pitch produces on a 12-lead sheet, so
# rewarding them would entrench the error this search exists to avoid.
COMMON_ROW_COUNTS = frozenset({12})
COMMON_ROW_BONUS = 1.12
# How far from the predicted position the tracker will accept a real peak.
TRACK_TOLERANCE = 0.35
# Weight of the newest spacing observation, which is what lets the tracker
# follow a page that curls.
SPACING_MEMORY = 0.7
# A perfectly clean sheet concentrates about a third of its row ink onto the
# baselines; the raw ratio is rescaled against that so reported confidence
# spans the useful range instead of hugging zero.
FULL_CONCENTRATION = 0.35
# A printed label sits above its trace.  These broad bounds are used only to
# locate the common offset; the actual rows are snapped to observed ink.
LABEL_OFFSET_FRACTION = (0.12, 0.48)
LABEL_FIT_FRACTION = 0.18
LABEL_SNAP_FRACTION = 0.28


def isoline_profile(layers: InkLayers, px_per_mm: float, span_mm: float = 3.0) -> np.ndarray:
    """Row profile of ink that runs horizontally, i.e. of isoelectric segments.

    Opening with a wide horizontal kernel keeps only strokes that continue for
    several millimetres. The mask is dilated by a row first and eroded back
    after, so a baseline drifting a pixel or two across the sheet still counts
    as continuous - without that tolerance a slightly rotated sheet scores zero.
    """
    mask = (layers.strong > 0.5).astype(np.uint8)
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    mask = cv2.dilate(mask, vertical)
    span = max(9, int(round(span_mm * max(px_per_mm, 3.0))))
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1))
    )
    horizontal = cv2.erode(horizontal, vertical)
    profile = horizontal.mean(axis=1).astype(np.float32)
    return cv2.GaussianBlur(profile.reshape(-1, 1), (1, 7), 0).ravel()


def _smooth(profile: np.ndarray, width: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(profile.reshape(-1, 1), (1, width), 0).ravel()


def row_repeat(profile: np.ndarray, low: int, high: int) -> int:
    """Dominant repeat of the row profile, by autocorrelation.

    Unlike the millimetre grid, only a dozen rows fit on a sheet, so the search
    window reaches a third of the image height and the estimate rests on very
    few periods. That is why the result is treated as a hypothesis to be tested
    rather than as the answer.
    """
    high = int(min(high, profile.size // 3))
    if high <= low or profile.size < 32:
        return 0
    window = int(high) * 2 + 1
    detrended = profile - cv2.blur(
        profile.reshape(1, -1), (window, 1), borderType=cv2.BORDER_REPLICATE
    ).ravel()
    padded = 1 << int(np.ceil(np.log2(profile.size * 2)))
    spectrum = np.fft.rfft(detrended, n=padded)
    correlation = np.fft.irfft((spectrum * np.conjugate(spectrum)).real, n=padded)
    if correlation[0] <= 0:
        return 0
    correlation = correlation[: high + 2] / correlation[0]
    best_lag, best_value = 0, -np.inf
    for lag in range(low, high + 1):
        if correlation[lag] < correlation[lag - 1] or correlation[lag] < correlation[lag + 1]:
            continue
        if correlation[lag] > best_value:
            best_lag, best_value = lag, float(correlation[lag])
    return best_lag


def _candidates(smoothed: np.ndarray, excluded: np.ndarray) -> list[int]:
    """Local maxima of the row profile that are not inside a printed text line."""
    floor = 0.08 * float(smoothed.max())
    return [
        index
        for index in range(1, smoothed.size - 1)
        if smoothed[index] >= smoothed[index - 1]
        and smoothed[index] >= smoothed[index + 1]
        and smoothed[index] > floor
        and not excluded[index]
    ]


def track_baselines(
    smoothed: np.ndarray,
    candidates: list[int],
    pitch: float,
) -> list[float]:
    """Walk out from the strongest baseline, adapting the spacing as it drifts."""
    if not candidates or pitch <= 1:
        return []
    height = smoothed.size
    seed = max(candidates, key=lambda index: smoothed[index])

    def walk(direction: int) -> list[float]:
        found: list[float] = []
        current = float(seed)
        spacing = float(pitch)
        while True:
            predicted = current + direction * spacing
            if predicted < -spacing * 0.5 or predicted > height + spacing * 0.5:
                break
            near = [c for c in candidates if abs(c - predicted) <= TRACK_TOLERANCE * spacing]
            if near:
                # Prefer a strong peak, but not one far from where it was due.
                nearest = max(near, key=lambda c: smoothed[c] - 0.002 * abs(c - predicted))
                spacing = SPACING_MEMORY * spacing + (1 - SPACING_MEMORY) * abs(nearest - current)
                current = float(nearest)
            else:
                current = float(predicted)
            if not 0 <= current < height:
                break
            found.append(current)
        return found

    return sorted([*walk(-1)[::-1], float(seed), *walk(1)])


def _score_hypothesis(
    baselines: list[float],
    isoline: np.ndarray,
    profile: np.ndarray,
    excluded: np.ndarray,
    text: np.ndarray,
) -> tuple[list[float], list[float], float]:
    """Trim the tracked set to the rows the sheet supports, and rate it.

    Only two things justify discarding a tracked baseline: it lies under a
    page-wide block of text, or it is an overshoot past the end of the print.
    Weak horizontal ink does not - a lead at a high heart rate has almost no
    flat segment - so isoelectric support is scored, never used as a gate.
    """
    if not baselines:
        return [], [], 0.0
    height = isoline.size
    reference = max(float(isoline.max()), 1e-9)

    def at(array: np.ndarray, baseline: float) -> float:
        row = int(round(np.clip(baseline, 0, height - 1)))
        # Look a little either side: the tracked position can sit a pixel off
        # the ridge without the row being any less of a baseline.
        return float(array[max(0, row - 2) : min(height, row + 3)].max())

    inside = [
        baseline
        for baseline in baselines
        if not excluded[int(round(np.clip(baseline, 0, height - 1)))]
    ]
    if not inside:
        return [], [], 0.0

    # Trim the ends only. The tracker walks until it leaves the image, so the
    # outer entries can land on bare paper past the last row, or on the header
    # block - full of ink, but with no baseline in it. An entry is dropped only
    # when it fails *both* tests, has no flat segment and no ink at all, since a
    # busy lead legitimately has one without the other. At most two are cut from
    # each end, so a sheet whose leads all run hot cannot collapse.
    flatness = [at(isoline, baseline) for baseline in inside]
    ink = [at(profile, baseline) for baseline in inside]
    flat_floor = 0.20 * float(np.median(flatness))
    ink_floor = 0.30 * float(np.median(ink))

    typed = [at(text, baseline) for baseline in inside]

    def spurious(position: int) -> bool:
        if flatness[position] >= flat_floor:
            return False
        # Nothing there at all, or a block of type: both are ends to cut, and
        # the header is the second kind - plenty of ink, no baseline in it.
        return ink[position] < ink_floor or typed[position] > 0.15

    first, last = 0, len(inside) - 1
    for _ in range(2):
        if first <= last and spurious(first):
            first += 1
    for _ in range(2):
        if last >= first and spurious(last):
            last -= 1
    kept = [float(baseline) for baseline in inside[first : last + 1]]
    supports = [at(isoline, baseline) / reference for baseline in kept]
    if len(kept) < 2:
        return kept, supports, 0.0
    spacing = np.diff(kept)
    regularity = clamp01(1.0 - float(np.std(spacing)) / max(float(np.mean(spacing)), 1e-6) * 2.0)

    # Each claimed row earns the ink standing on it and pays a fixed cost, so a
    # row is only worth claiming if the sheet actually supports it. Scoring the
    # *total* this way rather than the average is what stops a sparse set from
    # winning by sitting on the few strongest baselines and ignoring the rest -
    # skipping a real lead now forfeits its ink.
    ink_reference = max(float(profile.max()), 1e-9)
    strength = [at(profile, baseline) / ink_reference for baseline in kept]
    score = float(sum(value - ROW_COST for value in strength)) * (0.5 + 0.5 * regularity)
    if len(kept) in COMMON_ROW_COUNTS:
        score *= COMMON_ROW_BONUS
    return kept, supports, score


def _text_only_rows(
    text_coverage: np.ndarray | None,
    isoline: np.ndarray,
    height: int,
) -> np.ndarray:
    """Rows given over to type, with no lead baseline running through them.

    Text coverage alone is not enough to disqualify a row. On a sheet whose
    header sits across the first lead, the top rows are simultaneously full of
    type and carrying a real trace; discarding them there costs a whole lead.
    Only a row that is both covered by type and free of horizontally-continuous
    ink is genuinely text and nothing else.
    """
    if text_coverage is None or text_coverage.shape != (height,):
        return np.zeros(height, dtype=bool)
    reference = max(float(isoline.max()), 1e-9)
    baseline_free = (isoline / reference) < 0.10
    return (text_coverage > 0.15) & baseline_free


def _label_guided_baselines(
    anchors: np.ndarray,
    candidates: list[int],
    profile: np.ndarray,
    isoline: np.ndarray,
    height: int,
    row_count: int,
    low_pitch: int,
    high_pitch: int,
) -> tuple[list[float], list[float], dict[str, float]] | None:
    """Recover a complete row set from an ordered subset of printed labels.

    Curled Bionet sheets have a row pitch that changes smoothly from top to
    bottom.  A quadratic in the row index captures that drift without assuming
    equal spacing.  Which labels were found is unknown, so every ordered subset
    of the expected rows is tried.  Only geometrically feasible, low-residual
    fits survive; their predicted baselines are then shifted below the labels
    and snapped to nearby observed ink.

    This is a fallback, not another weighted feature in the ordinary score.  It
    is invoked only when the ink tracker failed to return the expected number of
    rows, avoiding the scalar-score failure documented for photo_296.
    """
    values = np.asarray(anchors, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = np.unique(np.round(values, 2))
    if row_count < 3 or values.size < 6 or values.size > row_count:
        return None

    positions = np.arange(row_count, dtype=np.float64)
    candidate_rows = np.asarray(candidates, dtype=np.float64)
    profile_reference = max(float(profile.max()), 1e-9)
    models: list[tuple[float, float, np.ndarray]] = []
    for assignment in combinations(range(row_count), int(values.size)):
        coefficients = np.polyfit(np.asarray(assignment, dtype=np.float64), values, 2)
        predicted = np.polyval(coefficients, positions)
        spacing = np.diff(predicted)
        if predicted[0] < 0.0 or predicted[-1] >= height:
            continue
        if spacing.min() < low_pitch * 0.70 or spacing.max() > high_pitch * 1.05:
            continue
        residual = np.abs(predicted[np.asarray(assignment)] - values)
        pitch = max(float(np.median(spacing)), 1.0)
        if float(residual.max()) > LABEL_FIT_FRACTION * pitch:
            continue
        models.append((float(residual.max()), float(residual.mean()), predicted))

    best: tuple[tuple[float, ...], np.ndarray, float, float] | None = None
    for maximum_residual, mean_residual, label_rows in models:
        pitch = max(float(np.median(np.diff(label_rows))), 1.0)
        offset_low = LABEL_OFFSET_FRACTION[0] * pitch
        offset_high = LABEL_OFFSET_FRACTION[1] * pitch
        for offset in np.arange(offset_low, offset_high + 0.5, 1.0):
            targets = label_rows + offset
            snapped: list[float] = []
            distances: list[float] = []
            for target in targets:
                near = candidate_rows[
                    np.abs(candidate_rows - target) <= LABEL_SNAP_FRACTION * pitch
                ]
                if near.size == 0:
                    snapped.append(float(target))
                    distances.append(LABEL_SNAP_FRACTION * pitch)
                    continue
                # Distance is the primary evidence.  Ink only breaks near-ties,
                # so a large QRS cannot pull the row away from its label model.
                rows = np.clip(np.round(near).astype(np.int32), 0, height - 1)
                objective = np.abs(near - target) / pitch
                objective -= 0.03 * profile[rows] / profile_reference
                chosen = int(np.argmin(objective))
                snapped.append(float(near[chosen]))
                distances.append(abs(float(near[chosen]) - float(target)))

            rows = np.asarray(snapped, dtype=np.float64)
            if np.diff(rows).min() <= 0.35 * pitch:
                continue
            # Leave room for half of the final strip above the footer.  A model
            # ending inside the footer is the common false translation.
            if rows[-1] > height - 0.35 * pitch:
                continue
            row_indices = np.clip(np.round(rows).astype(np.int32), 0, height - 1)
            distance = np.asarray(distances, dtype=np.float64)
            key = (
                maximum_residual / pitch,
                mean_residual / pitch,
                float(np.percentile(distance, 80)) / pitch,
                float(distance.mean()) / pitch,
                -float(np.mean(isoline[row_indices])),
            )
            if best is None or key < best[0]:
                best = (key, rows, maximum_residual, float(offset))

    if best is None:
        return None
    _, rows, residual, offset = best
    reference = max(float(isoline.max()), 1e-9)
    supports = [float(isoline[int(round(row))]) / reference for row in rows]
    return (
        [float(row) for row in rows],
        supports,
        {
            "label_anchor_count": float(values.size),
            "label_fit_residual_px": float(residual),
            "label_to_baseline_offset_px": float(offset),
        },
    )


def detect_layout(
    layers: InkLayers,
    config: PipelineConfig,
    text_coverage: np.ndarray | None = None,
    label_anchors: np.ndarray | None = None,
) -> tuple[tuple[LeadBand, ...], dict[str, object]]:
    profile = row_ink_profile(layers)
    height = profile.size
    smoothed = _smooth(profile)
    isoline = isoline_profile(layers, _px_per_mm_hint(layers))
    excluded = _text_only_rows(text_coverage, isoline, height)
    coverage = (
        text_coverage
        if text_coverage is not None and text_coverage.shape == (height,)
        else np.zeros(height, dtype=np.float32)
    )

    summary: dict[str, object] = {
        "pixel_density": float(layers.strong.mean()),
        "row_coverage": float((profile > 0.04).mean()),
        "col_coverage": float((layers.strong.mean(axis=0) > 0.04).mean()),
        "text_rows_excluded": int(excluded.sum()),
    }

    low = max(4, int(height / (config.max_lead_rows + 1)))
    high = max(low + 2, int(height / max(config.min_lead_rows - 0.5, 1.0)))
    repeat = row_repeat(profile, low, high)
    candidates = _candidates(smoothed, excluded)

    best: tuple[list[float], list[float], float, float] | None = None
    # Every plausible row pitch is tried, not just the autocorrelation peak.
    # That peak rests on only a dozen periods and is readily pulled long by a
    # sheet whose leads overlap - and a pitch even slightly too long makes the
    # tracker step straight over a row. Tracking is cheap, so the pitch is
    # chosen by which hypothesis actually explains the sheet best.
    for pitch in range(low, high + 1):
        tracked = track_baselines(smoothed, candidates, float(pitch))
        kept, supports, score = _score_hypothesis(
            tracked, isoline, smoothed, excluded, coverage
        )
        if best is None or score > best[2]:
            best = (kept, supports, score, float(pitch))

    if best is None or len(best[0]) < 2:
        summary.update(
            {
                "row_pitch_px": 0.0,
                "row_count": 0,
                "comb_score": 0.0,
                "baseline_regularity": 0.0,
                "assumed_lead_order": False,
            }
        )
        return (), summary

    layout_method = "ink_tracker"
    expected = config.expected_lead_rows
    if expected is not None and len(best[0]) != expected and label_anchors is not None:
        guided = _label_guided_baselines(
            label_anchors,
            candidates,
            smoothed,
            isoline,
            height,
            int(expected),
            low,
            high,
        )
        if guided is not None:
            guided_rows, guided_supports, guided_summary = guided
            guided_spacing = np.diff(guided_rows)
            guided_regularity = clamp01(
                1.0
                - float(np.std(guided_spacing))
                / max(float(np.mean(guided_spacing)), 1e-6)
                * 2.0
            )
            # The fit is already constrained by ordered labels and page bounds;
            # retain a score only for reporting and confidence comparison.
            best = (
                guided_rows,
                guided_supports,
                float(sum(guided_supports)) * (0.5 + 0.5 * guided_regularity),
                float(np.median(guided_spacing)),
            )
            summary.update(guided_summary)
            layout_method = "ordered_label_fallback"

    baselines, supports, score, pitch = best
    spacing = np.diff(baselines)
    pitch = float(np.median(spacing))
    regularity = clamp01(
        1.0 - float(np.std(spacing)) / max(float(np.mean(spacing)), 1e-6) * 2.0
    )

    # Band edges are the midpoints between neighbouring baselines, so uneven
    # spacing on a curled page gives uneven bands rather than overlapping ones.
    labels = STANDARD_LEAD_ORDERS.get(len(baselines), ())
    bands: list[LeadBand] = []
    for index, baseline in enumerate(baselines):
        above = baselines[index - 1] if index > 0 else baseline - pitch
        below = baselines[index + 1] if index + 1 < len(baselines) else baseline + pitch
        top = int(max(0, round((above + baseline) / 2)))
        bottom = int(min(height, round((baseline + below) / 2)))
        bands.append(
            LeadBand(
                index=index,
                top=top,
                bottom=max(top + 1, bottom),
                baseline=float(baseline),
                ink=float(supports[index]),
                label=labels[index] if index < len(labels) else f"row {index + 1}",
            )
        )

    summary.update(
        {
            "row_pitch_px": pitch,
            "row_count": len(bands),
            "comb_score": clamp01(float(np.mean(supports))),
            "baseline_regularity": regularity,
            "assumed_lead_order": len(bands) in STANDARD_LEAD_ORDERS,
            # How many rows short of the near-universal twelve-lead print this
            # sheet came out. Rows are left unnamed when the count does not
            # match a layout - see the note on assign_lead_names in HANDOFF.md -
            # but the shortfall itself is worth stating.
            "rows_short_of_twelve": max(0, 12 - len(bands)),
            "row_repeat_px": float(repeat),
            "mean_row_support": float(np.mean(supports)),
            "hypothesis_score": float(score),
            "layout_method": layout_method,
            "label_anchor_count": float(
                summary.get(
                    "label_anchor_count",
                    0 if label_anchors is None else np.asarray(label_anchors).size,
                )
            ),
        }
    )
    return tuple(bands), summary


def _px_per_mm_hint(layers: InkLayers) -> float:
    """Millimetre scale implied by the background kernel stage 6 chose."""
    return max(3.0, layers.kernel_px / 3.5)


def layout_probability(summary: dict[str, object]) -> float:
    bands = int(summary.get("row_count", 0) or 0)
    if bands <= 0:
        return 0.05
    score = clamp01(float(summary.get("comb_score", 0.0) or 0.0) / FULL_CONCENTRATION)
    regularity = float(summary.get("baseline_regularity", 0.0) or 0.0)
    completeness = clamp01(bands / 12.0) if bands <= 12 else 1.0
    return clamp01(0.10 + 0.50 * score + 0.25 * regularity + 0.15 * completeness)


def detect_column_blocks(layers: InkLayers, bands: tuple[LeadBand, ...]) -> int:
    """How many side-by-side lead columns the sheet uses (1 for a rhythm layout).

    A 4x3 sheet leaves a narrow ink-free corridor between blocks that repeats in
    every band; a 12x1 sheet has no such corridor.
    """
    if not bands:
        return 1
    stacked = np.zeros(layers.strong.shape[1], dtype=np.float32)
    for band in bands:
        stacked += layers.strong[band.top : band.bottom].max(axis=0)
    stacked /= max(len(bands), 1)
    smoothed = cv2.GaussianBlur(stacked.reshape(1, -1), (9, 1), 0).ravel()
    width = smoothed.size
    interior = smoothed[width // 12 : width - width // 12]
    if interior.size == 0:
        return 1
    empty = interior < 0.15
    # Count corridors that are wide enough not to be an isoelectric pause.
    runs = 0
    run_length = 0
    minimum = max(6, width // 90)
    for value in empty:
        if value:
            run_length += 1
        else:
            if run_length >= minimum:
                runs += 1
            run_length = 0
    if run_length >= minimum:
        runs += 1
    return int(np.clip(runs + 1, 1, 4))


def layout_stage(
    bands: tuple[LeadBand, ...],
    summary: dict[str, object],
) -> PipelineStageResult:
    if not bands:
        return PipelineStageResult(
            "Layout detector",
            layout_probability(summary),
            "No regular lead rows found; the sheet is treated as a single strip.",
            evidence=(f"text_rows_excluded={int(summary.get('text_rows_excluded', 0))}",),
        )
    short = int(summary.get("rows_short_of_twelve", 0) or 0)
    if summary.get("assumed_lead_order"):
        labelled = "assumed standard order"
    elif short:
        labelled = f"unnamed rows, {short} short of a twelve-lead sheet"
    else:
        labelled = "unnamed rows"
    return PipelineStageResult(
        "Layout detector",
        layout_probability(summary),
        f"{len(bands)} isoelectric baselines at {float(summary['row_pitch_px']):.1f} px "
        f"spacing ({labelled}).",
        evidence=(
            f"row_count={len(bands)}",
            f"rows_short_of_twelve={short}",
            f"row_pitch_px={float(summary['row_pitch_px']):.1f}",
            f"isoline_support={float(summary['comb_score']):.2f}",
            f"baseline_regularity={float(summary['baseline_regularity']):.2f}",
            f"text_rows_excluded={int(summary.get('text_rows_excluded', 0))}",
            f"columns={int(summary.get('column_blocks', 1))}",
        ),
    )
