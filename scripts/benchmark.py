#!/usr/bin/env python

"""Score the reconstruction against what is actually printed on the sheets.

Five photographs with no marked-up reference is a recipe for tuning thresholds
until one picture improves and another quietly breaks. This script exists so
that every change can be answered with a number, and so that the number comes
from something outside the pipeline wherever such a thing exists.

Three kinds of evidence are used, in decreasing order of authority:

1.  **Transcribed print** - ``examples/ground_truth.json`` holds what a person
    read off the paper: where each lead label sits, the footer's speed and gain,
    and the recorder's own measurement block. Only one of the five sheets has
    the whole block in frame, so exactly one printed heart rate exists; the
    script reports that honestly instead of inventing agreement.
2.  **Reference-free measurements** - whether the reconstructed pen sits on
    printed ink, whether two leads' paths collide, how far a path strays from
    its own baseline. These need no annotation and catch the failure that
    matters most: a confident line drawn through blank paper.
3.  **Self-reported confidence** - the stage probabilities and quality grade,
    included only so a run can be compared with itself.

    python scripts/benchmark.py                       # table + summary
    python scripts/benchmark.py --out runs/after.json
    python scripts/benchmark.py --baseline runs/before.json --out runs/after.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from miocard.imaging import ProcessedImage, process_ecg_image  # noqa: E402
from miocard.pipeline import ECGPipelineState, map_source_points, median_beats  # noqa: E402
from miocard.pipeline.series import effective_trace  # noqa: E402

DEFAULT_GROUND_TRUTH = ROOT_DIR / "examples" / "ground_truth.json"
DEFAULT_IMAGE_DIR = ROOT_DIR / "examples"

# A hand-placed lead label is a few pixels off the baseline it names, and the
# recorder prints it slightly above the trace. Half a row pitch is the widest
# tolerance that still cannot reach a neighbouring lead.
LEAD_MATCH_TOLERANCE = 0.5
# How far the printed lead name may sit from the baseline it names, as a
# fraction of the row pitch. This is a fact about how a recorder prints, not a
# tuning knob: a name a third of a row away from its trace would be sitting on
# the neighbouring lead. Bounding it stops the alignment from "explaining" a
# comb that has drifted by a whole row as a large typographic offset.
MAX_LABEL_OFFSET = 0.35
# How close the reconstructed pen has to come to a dark pixel to count as
# standing on the printed trace. One millimetre-ish; the printed stroke is about
# two pixels wide at these resolutions and the path is quantised to whole rows.
INK_HIT_RADIUS_PX = 2
# Two paths this close are drawing the same ink, whoever it belongs to.
COLLISION_PX = 2.0
# A reconstruction this near the border of the picture is on the edge of the
# sheet or on whatever the sheet was photographed against.
EDGE_MARGIN_PX = 3
# Background-corrected darkness below which the paper is simply blank. Not a
# tuning knob: at these exposures the printed grid alone is darker than this.
BLANK_PAPER_INK = 10.0
# Ink dark enough to be pen rather than grid, as a fraction of the sheet's 99th
# percentile. Chosen by looking at which pixels the mask keeps, not by tuning a
# score: below this the millimetre grid in a shadowed patch starts counting.
UNCOVERED_INK_FRACTION = 0.5
UNCOVERED_TEXT_CUTOFF = 0.20
UNCOVERED_GRID_CUTOFF = 0.35


# ---------------------------------------------------------------------------
# Reference-free measurements
# ---------------------------------------------------------------------------


def ink_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dark strokes, background-corrected, as a boolean mask.

    Deliberately not the pipeline's own ink stage: scoring a reconstruction
    against the very layer that produced it would only prove it is consistent.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    ink = cv2.subtract(background, gray)
    strong = float(np.percentile(ink, 99.0))
    if strong <= 1.0:
        return np.zeros(gray.shape, dtype=bool), ink
    return ink > max(12.0, 0.35 * strong), ink


def _reconstructed_rows(state: ECGPipelineState, path) -> np.ndarray:
    px_per_mm = float(state.grid.small_cell_px) if state.grid is not None else 0.0
    smooth_px = max(3, int(round(px_per_mm))) if px_per_mm > 1 else 3
    return effective_trace(path, smooth_px)


def ink_hit_rate(
    state: ECGPipelineState,
    mask: np.ndarray,
    ink: np.ndarray,
) -> dict[str, object]:
    """How much of each reconstructed pen stands on printed ink.

    This is the check that a picture makes obvious and a confidence number
    hides: a path that has wandered onto blank paper still reports a cost. The
    worst lead is reported next to the mean, because eleven good leads will
    otherwise bury one that was drawn through nothing - which is exactly the
    failure a reader would notice first.
    """
    height, width = mask.shape
    hits = 0
    total = 0
    worst = (1.0, "")
    edge = 0
    blank = 0
    for path in state.paths:
        rows = _reconstructed_rows(state, path)
        lead_hits = 0
        lead_total = 0
        for column in range(path.x_start, min(path.x_end, width)):
            row = int(round(rows[column]))
            low = max(0, row - INK_HIT_RADIUS_PX)
            high = min(height, row + INK_HIT_RADIUS_PX + 1)
            lead_total += 1
            if mask[low:high, column].any():
                lead_hits += 1
            # A path lying along the border of the photograph is following the
            # edge of the paper or the table it sits on, not a trace.
            if row <= EDGE_MARGIN_PX or row >= height - 1 - EDGE_MARGIN_PX:
                edge += 1
            # The hit rate above is a yes/no against one threshold, which makes
            # it blind to the difference between "on a faint stroke tip" and
            # "on nothing". This counts only the second: paper as bright as the
            # background, where no pen ever went.
            if ink[row, column] < BLANK_PAPER_INK:
                blank += 1
        hits += lead_hits
        total += lead_total
        if lead_total and lead_hits / lead_total < worst[0]:
            worst = (lead_hits / lead_total, path.label)
    return {
        "rate": hits / total if total else 0.0,
        "columns": total,
        "worst_rate": worst[0] if total else 0.0,
        "worst_lead": worst[1],
        "edge_rate": edge / total if total else 0.0,
        "blank_rate": blank / total if total else 0.0,
    }


def uncovered_ink_rate(state: ECGPipelineState, image: np.ndarray) -> float:
    """Printed ink inside the lead rows that no reconstruction accounts for.

    The on-ink rate asks whether the pen is standing on something; this asks the
    opposite and much harder question - whether anything printed was left
    unread. It is the only metric here that can see a truncated peak, because a
    QRS stroke the envelope never reached is ink lying in plain sight with no
    reconstruction over it.

    Text and grid are excluded using the pipeline's own posteriors, so the
    number is not fully independent, and some label and grid pixels survive the
    cut on every sheet. That contamination is a constant of the photograph: it
    does not move when the reconstruction changes, which is what makes the
    metric usable for before-and-after even though its absolute value is not a
    percentage of anything meaningful.
    """
    if not state.paths:
        return 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ink = cv2.subtract(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel), gray).astype(np.float32)
    strong = ink > max(20.0, UNCOVERED_INK_FRACTION * float(np.percentile(ink, 99)))

    height, width = strong.shape
    covered = np.zeros_like(strong)
    for path in state.paths:
        for column in range(path.x_start, min(path.x_end, width)):
            top = int(np.clip(path.top_envelope[column], 0, height - 1))
            bottom = int(np.clip(path.bottom_envelope[column], 0, height - 1))
            covered[max(0, top - 1) : min(height, bottom + 2), column] = True

    inside = np.zeros_like(strong)
    inside[
        min(path.band.top for path in state.paths) : max(path.band.bottom for path in state.paths),
        min(path.x_start for path in state.paths) : max(path.x_end for path in state.paths),
    ] = True
    maps = state.probability_maps
    printed = (
        strong
        & inside
        & (maps["text"] < UNCOVERED_TEXT_CUTOFF)
        & (maps["grid"] < UNCOVERED_GRID_CUTOFF)
    )
    total = int(printed.sum())
    return float(np.count_nonzero(printed & ~covered) / total) if total else 0.0


def collision_rate(state: ECGPipelineState) -> tuple[float, float]:
    """Fraction of columns where two lead paths sit on top of each other."""
    if len(state.paths) < 2:
        return 0.0, 0.0
    deepest = 0.0
    ordered = sorted(state.paths, key=lambda path: path.band.baseline)
    columns = 0
    collisions = 0
    for upper, lower in zip(ordered, ordered[1:], strict=False):
        start = max(upper.x_start, lower.x_start)
        end = min(upper.x_end, lower.x_end)
        if end <= start:
            continue
        top = _reconstructed_rows(state, upper)[start:end]
        bottom = _reconstructed_rows(state, lower)[start:end]
        columns += end - start
        collisions += int(np.count_nonzero(bottom - top < COLLISION_PX))
        # A rate alone hides how bad a collision is. Two leads that swap places
        # for six columns at a QRS score almost nothing here, and that is
        # exactly the failure a reader sees: measured on examples/photo_293 the
        # lower lead's line rises 56 px above the upper lead's.
        deepest = max(deepest, float(np.max(top - bottom)) if end > start else 0.0)
    return collisions / columns if columns else 0.0, max(0.0, deepest)


def baseline_excursion(state: ECGPipelineState) -> float:
    """Median distance of a path from its own baseline, in row pitches.

    A lead that has adopted its neighbour's trace shows up here as an excursion
    approaching one whole pitch, where a real waveform stays well under half.
    """
    pitch = float(state.layout.get("row_pitch_px", 0.0) or 0.0)
    if pitch <= 1 or not state.paths:
        return 0.0
    excursions = [
        float(np.median(np.abs(path.y[path.x_start : path.x_end] - path.band.baseline)))
        for path in state.paths
        if path.x_end > path.x_start
    ]
    return float(np.median(excursions)) / pitch if excursions else 0.0


# ---------------------------------------------------------------------------
# Scoring against the transcription
# ---------------------------------------------------------------------------


def _align(
    bands: np.ndarray,
    leads: np.ndarray,
    offset: float,
    skip_cost: float,
) -> tuple[float, list[int]]:
    """Cheapest order-preserving match of detected rows to printed leads.

    Both sequences run down the sheet, and the detected rows are a subsequence
    of the printed ones - a row can be missed, and a spurious row can be found,
    but they cannot swap places. Matching each printed lead to whichever band
    happens to be nearest ignores that, and on a sheet whose comb has drifted by
    half a pitch it silently pairs half the leads with their neighbour's row.

    Returns the total distance and, per band, the index of the lead it answers
    (-1 for a band that answers none).
    """
    n, m = bands.size, leads.size
    infinity = float("inf")
    cost = np.full((n + 1, m + 1), infinity)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[0, :] = 0.0
    for i in range(1, n + 1):
        for j in range(m + 1):
            # 2: this band answers no printed lead at all.
            best, choice = cost[i - 1, j] + skip_cost, 2
            if j > 0:
                paired = cost[i - 1, j - 1] + abs(bands[i - 1] - offset - leads[j - 1])
                if paired < best:
                    best, choice = paired, 1
                # 0: this printed lead was not found.
                if cost[i, j - 1] < best:
                    best, choice = cost[i, j - 1], 0
            cost[i, j] = best
            back[i, j] = choice
    assignment = [-1] * n
    i, j = n, m
    while i > 0:
        choice = back[i, j]
        if choice == 0:
            j -= 1
        elif choice == 1:
            assignment[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            i -= 1
    return float(cost[n, m]), assignment


def match_leads(
    state: ECGPipelineState,
    label_y_source: dict[str, float],
) -> dict[str, object]:
    """Which printed leads the layout detector actually found, and under what name.

    The transcription lives in the coordinates of the photograph; the bands live
    in the coordinates of the deskewed, cropped, rescaled picture. The pipeline
    publishes the affine between them, so the comparison is exact rather than
    approximate.
    """
    pitch = float(state.layout.get("row_pitch_px", 0.0) or 0.0)
    matrix = state.source_transform
    if matrix is None or not state.bands or pitch <= 1:
        return {
            "found": 0,
            "named": 0,
            "expected": len(label_y_source),
            "missing": sorted(label_y_source),
            "misnamed": [],
            "spurious_rows": 0,
            "label_offset_px": 0.0,
            "residual_px": None,
        }

    names = list(label_y_source)
    points = np.array([[0.0, float(label_y_source[name])] for name in names])
    mapped = map_source_points(matrix, points)[:, 1]

    baselines = np.array([band.baseline for band in state.bands])

    # The recorder prints a lead's name a little above the trace it names, by a
    # constant that depends on the machine, not on the reconstruction. Take that
    # constant out before scoring, or a systematic 20 px of typography gets
    # counted as 20 px of layout error. What stays under test is the *spacing*
    # of the printed leads, which is the thing being claimed. The offset is
    # found by trying every plausible one and keeping whichever explains the
    # sheet best, so a drifted comb cannot fake agreement at one end.
    skip_cost = LEAD_MATCH_TOLERANCE * pitch
    best: tuple[float, list[int], float] | None = None
    limit = MAX_LABEL_OFFSET * pitch
    for offset in np.arange(-limit, limit + 1e-9, 1.0):
        total, assignment = _align(baselines, mapped, float(offset), skip_cost)
        if best is None or total < best[0]:
            best = (total, assignment, float(offset))
    assert best is not None
    _, assignment, offset = best

    lead_to_band = {lead: band for band, lead in enumerate(assignment) if lead >= 0}
    missing = [name for index, name in enumerate(names) if index not in lead_to_band]
    misnamed = [
        f"{names[lead]}->{state.bands[band].label}"
        for lead, band in sorted(lead_to_band.items())
        if state.bands[band].label != names[lead]
    ]
    residuals = [
        abs(float(baselines[band]) - offset - float(mapped[lead]))
        for lead, band in lead_to_band.items()
    ]
    return {
        "found": len(lead_to_band),
        "named": len(lead_to_band) - len(misnamed),
        "expected": len(names),
        "missing": missing,
        "misnamed": misnamed,
        # Rows that answered no printed lead at all: a header block or a footer
        # taken for a waveform.
        "spurious_rows": int(sum(1 for lead in assignment if lead < 0)),
        "label_offset_px": round(offset, 1),
        # What is left once the printing convention is removed: how well the
        # detected rows line up with the printed ones.
        "residual_px": round(float(np.max(residuals)), 1) if residuals else None,
    }


def amplitude_check(state: ECGPipelineState, printed: list | None) -> dict[str, object]:
    """The recorder's own S(V1) and R(V5) against the ones we reconstructed.

    This is the only external check the project has on the whole amplitude
    chain - grid pitch, millimetres per millivolt, path, envelope - and unlike
    the printed heart rate it exists on more than one sheet. The recorder prints
    the two numbers it used for the Sokolow-Lyon index, and they are amplitudes
    of named waves in named leads, which is exactly what a median beat gives.
    """
    beats = {beat.label: beat for beat in median_beats(state)}
    measured_sv1 = abs(beats["V1"].trough_mv) if "V1" in beats else None
    measured_rv5 = beats["V5"].peak_mv if "V5" in beats else None
    result: dict[str, object] = {
        "sv1_mv": round(measured_sv1, 3) if measured_sv1 is not None else None,
        "rv5_mv": round(measured_rv5, 3) if measured_rv5 is not None else None,
        "sv1_printed_mv": None,
        "rv5_printed_mv": None,
        "amplitude_error_mv": None,
    }
    if not printed or len(printed) < 2:
        return result
    result["sv1_printed_mv"] = float(printed[0])
    result["rv5_printed_mv"] = float(printed[1])
    errors = [
        abs(measured - float(expected))
        for measured, expected in ((measured_sv1, printed[0]), (measured_rv5, printed[1]))
        if measured is not None
    ]
    if errors:
        result["amplitude_error_mv"] = round(float(np.mean(errors)), 3)
    return result


def measure(record: dict, result: ProcessedImage, processed: np.ndarray) -> dict[str, object]:
    state = result.pipeline
    if state is None:
        return {"file": record["file"], "error": "pipeline did not run"}

    truth = record.get("printed_measurements") or {}
    paper = record.get("paper") or {}
    mask, ink_image = ink_mask(processed)
    ink = ink_hit_rate(state, mask, ink_image)
    collisions, depth = collision_rate(state)
    leads = match_leads(state, record.get("lead_label_y", {}))

    beats = state.beats
    measured_rate = float(beats.heart_rate_bpm) if beats is not None else None
    printed_rate = truth.get("rate_bpm")
    rate_error = (
        round(measured_rate - float(printed_rate), 1)
        if measured_rate is not None and printed_rate
        else None
    )

    reported_gain = float(state.text.get("gain_mm_mv", 0.0) or 0.0)
    printed_gain = paper.get("gain_mm_mv")
    return {
        **amplitude_check(state, truth.get("sv1_rv5_sum_mv")),
        "file": record["file"],
        "rows": int(state.layout.get("row_count", 0) or 0),
        "leads_found": leads["found"],
        "leads_named": leads["named"],
        "leads_expected": leads["expected"],
        "leads_missing": leads["missing"],
        "leads_misnamed": leads["misnamed"],
        "leads_spurious_rows": leads["spurious_rows"],
        "label_offset_px": leads["label_offset_px"],
        "lead_residual_px": leads["residual_px"],
        "labels_located": int(state.text.get("labels_found", 0) or 0),
        "row_pitch_px": round(float(state.layout.get("row_pitch_px", 0.0) or 0.0), 1),
        "baseline_regularity": round(
            float(state.layout.get("baseline_regularity", 0.0) or 0.0), 3
        ),
        "px_per_mm": round(float(state.grid.small_cell_px), 2) if state.grid else None,
        "grid_confidence": state.grid.confidence if state.grid else "none",
        "ink_hit_rate": round(float(ink["rate"]), 3),
        "worst_lead_ink": round(float(ink["worst_rate"]), 3),
        "worst_lead": ink["worst_lead"],
        "edge_rate": round(float(ink["edge_rate"]), 3),
        "pen_on_blank": round(float(ink["blank_rate"]), 4),
        "traced_columns": ink["columns"],
        "collision_rate": round(collisions, 3),
        "crossing_depth_px": round(depth, 1),
        "uncovered_ink": round(uncovered_ink_rate(state, processed), 3),
        "baseline_excursion_pitch": round(baseline_excursion(state), 3),
        "rate_bpm": round(measured_rate, 1) if measured_rate is not None else None,
        "rate_printed_bpm": printed_rate,
        "rate_error_bpm": rate_error,
        "gain_mm_mv": reported_gain,
        "gain_printed_mm_mv": printed_gain,
        "gain_measured": bool(state.text.get("gain_measured", False)),
        "chest_gain_printed_mm_mv": paper.get("chest_gain_mm_mv"),
        "quality_grade": state.quality_grade,
        "quality_probability": round(float(state.quality.get("probability", 0.0) or 0.0), 3),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# name, header, width, formatter. Higher is better unless noted in COMPARE.
COLUMNS: tuple[tuple[str, str, int, str], ...] = (
    ("file", "sheet", 12, "s"),
    ("rows", "rows", 5, "d"),
    ("leads_found", "leads", 6, "lead"),
    ("lead_residual_px", "resid", 6, ".1f"),
    ("label_offset_px", "offset", 7, "+.0f"),
    ("labels_located", "lbl", 4, "d"),
    ("ink_hit_rate", "on-ink", 7, ".3f"),
    ("worst_lead_ink", "worst", 6, ".3f"),
    ("edge_rate", "onedge", 7, ".3f"),
    ("pen_on_blank", "blank", 7, ".4f"),
    ("uncovered_ink", "unread", 7, ".3f"),
    ("collision_rate", "collide", 8, ".3f"),
    ("crossing_depth_px", "cross", 6, ".0f"),
    ("baseline_excursion_pitch", "excurs", 7, ".3f"),
    ("baseline_regularity", "reg", 6, ".2f"),
    ("row_pitch_px", "pitch", 6, ".1f"),
    ("px_per_mm", "px/mm", 6, ".2f"),
    ("rate_bpm", "bpm", 6, ".1f"),
    ("rate_error_bpm", "vs print", 9, "+.1f"),
    ("amplitude_error_mv", "mV err", 7, ".3f"),
    ("quality_grade", "grade", 9, "s"),
)

# Which direction counts as an improvement, for the baseline diff.
COMPARE: dict[str, int] = {
    "leads_found": +1,
    "leads_named": +1,
    "lead_residual_px": -1,
    "labels_located": +1,
    "ink_hit_rate": +1,
    "worst_lead_ink": +1,
    # More of the printed record read is an improvement, as long as the columns
    # added are standing on ink - which the two rates above are there to check.
    "traced_columns": +1,
    "edge_rate": -1,
    "pen_on_blank": -1,
    "uncovered_ink": -1,
    "baseline_regularity": +1,
    "collision_rate": -1,
    "crossing_depth_px": -1,
    "baseline_excursion_pitch": -1,
    "amplitude_error_mv": -1,
    "quality_probability": +1,
}


def _cell(row: dict, name: str, spec: str, width: int) -> str:
    value = row.get(name)
    if spec == "lead":
        found, expected = row.get("leads_found"), row.get("leads_expected")
        text = "-" if found is None else f"{found}/{expected}"
    elif value is None:
        text = "-"
    elif spec == "s":
        text = str(value)
        if name == "file":
            text = text.split("@")[0].replace("photo_", "")
    else:
        text = format(value, spec)
    return text[:width].rjust(width)


def print_table(rows: list[dict]) -> None:
    header = "  ".join(name.rjust(width) for _, name, width, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['file'][:12]:>12}  {row['error']}")
            continue
        print("  ".join(_cell(row, key, spec, width) for key, _, width, spec in COLUMNS))


def print_notes(rows: list[dict]) -> None:
    print()
    for row in rows:
        if "error" in row:
            continue
        name = row["file"].split("@")[0]
        notes: list[str] = []
        if row["leads_missing"]:
            notes.append("not found: " + ", ".join(row["leads_missing"]))
        if row["leads_misnamed"]:
            notes.append("misnamed: " + ", ".join(row["leads_misnamed"]))
        if row["edge_rate"] > 0.02:
            notes.append(
                f"{row['edge_rate'] * 100:.0f}% of the reconstruction runs along the picture border"
            )
        if row["worst_lead_ink"] < 0.85:
            notes.append(
                f"lead {row['worst_lead']} stands on ink in only "
                f"{row['worst_lead_ink'] * 100:.0f}% of its columns"
            )
        if row["leads_spurious_rows"]:
            notes.append(f"{row['leads_spurious_rows']} detected row(s) answer no printed lead")
        if row["rows"] != row["leads_expected"]:
            notes.append(f"{row['rows']} rows claimed for {row['leads_expected']} printed leads")
        printed_gain = row.get("gain_printed_mm_mv")
        if printed_gain and abs(row["gain_mm_mv"] - float(printed_gain)) > 0.1:
            notes.append(f"gain {row['gain_mm_mv']:.1f} reported, {printed_gain:.1f} printed")
        if row.get("chest_gain_printed_mm_mv"):
            notes.append(
                f"sheet prints two gains ({printed_gain:.0f}/"
                f"{float(row['chest_gain_printed_mm_mv']):.0f} mm/mV); "
                "the chest leads are scaled wrong"
            )
        if row.get("rate_printed_bpm") is None:
            notes.append("no printed rate on this sheet")
        if notes:
            print(f"{name}: " + "; ".join(notes))


def print_summary(rows: list[dict]) -> None:
    scored = [row for row in rows if "error" not in row]
    if not scored:
        return
    found = sum(row["leads_found"] for row in scored)
    expected = sum(row["leads_expected"] for row in scored)
    named = sum(row["leads_named"] for row in scored)
    print()
    print(f"leads located      {found}/{expected}")
    print(f"leads named right  {named}/{expected}")
    print(f"mean on-ink        {np.mean([row['ink_hit_rate'] for row in scored]):.3f}")
    print(f"worst lead on-ink  {min(row['worst_lead_ink'] for row in scored):.3f}")
    print(f"mean on-edge       {np.mean([row['edge_rate'] for row in scored]):.3f}")
    print(f"mean unread ink    {np.mean([row['uncovered_ink'] for row in scored]):.3f}")
    print(f"mean collision     {np.mean([row['collision_rate'] for row in scored]):.3f}")
    print(f"worst crossing     {max(row['crossing_depth_px'] for row in scored):.0f} px")
    print(f"mean excursion     {np.mean([row['baseline_excursion_pitch'] for row in scored]):.3f}")
    errors = [row["rate_error_bpm"] for row in scored if row["rate_error_bpm"] is not None]
    if errors:
        print(f"rate error         {np.mean(np.abs(errors)):.1f} bpm over {len(errors)} sheet(s)")
    amplitudes = [
        row["amplitude_error_mv"] for row in scored if row["amplitude_error_mv"] is not None
    ]
    if amplitudes:
        print(
            f"amplitude error    {np.mean(amplitudes):.3f} mV over {len(amplitudes)} sheet(s) "
            "(S in V1, R in V5, against the recorder's own print)"
        )


def print_diff(rows: list[dict], baseline: list[dict]) -> None:
    previous = {row["file"]: row for row in baseline}
    print()
    print("change against baseline (worse marked !):")
    any_change = False
    for row in rows:
        before = previous.get(row["file"])
        if before is None or "error" in row or "error" in before:
            continue
        deltas: list[str] = []
        for key, direction in COMPARE.items():
            new, old = row.get(key), before.get(key)
            if not isinstance(new, int | float) or not isinstance(old, int | float):
                continue
            change = new - old
            if abs(change) < 1e-9:
                continue
            mark = "!" if change * direction < 0 else " "
            deltas.append(f"{mark}{key} {old:g}->{new:g}")
        if deltas:
            any_change = True
            print(f"  {row['file'].split('@')[0]}: " + ", ".join(deltas))
    if not any_change:
        print("  nothing moved")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--only", default="", help="Substring filter on the file name.")
    parser.add_argument("--out", default="", help="Write the measurements here as JSON.")
    parser.add_argument("--baseline", default="", help="Compare against an earlier --out file.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    image_dir = Path(args.image_dir)

    rows: list[dict] = []
    for record in truth["records"]:
        if args.only and args.only not in record["file"]:
            continue
        path = image_dir / record["file"]
        if not path.exists():
            rows.append({"file": record["file"], "error": "image missing"})
            continue
        result = process_ecg_image(path.read_bytes(), args.model, args.max_bytes)
        processed = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        rows.append(measure(record, result, processed))

    if not rows:
        print("Nothing to measure.")
        return 2

    print_table(rows)
    print_notes(rows)
    print_summary(rows)

    if args.baseline:
        print_diff(rows, json.loads(Path(args.baseline).read_text(encoding="utf-8"))["rows"])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"model": args.model, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
