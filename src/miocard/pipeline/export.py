"""Stage 15 - Export.

Packages the blackboard into things other systems can consume: a JSON-safe
summary, the digitised leads as CSV, and the compact note block that goes into
the model prompt. Measured quantities are only exported when the stage that
produced them was confident enough for them to mean anything - a heart rate from
an irregular, half-invented reconstruction is worse than no heart rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from miocard.pipeline.series import moving_average
from miocard.pipeline.state import ECGPipelineState, PipelineStageResult, clamp01

# Below this the beat detector is guessing, and a printed bpm would be taken at
# face value by whoever reads the report.
MIN_RATE_CONFIDENCE = 0.45
EXPORT_DECIMATION_HZ = 250.0

# The median beat handed to the model. A whole record is tens of thousands of
# numbers and would swamp the request; one averaged beat per lead is where the
# morphology actually lives, and it is what a reader looks at anyway. The window
# reaches back far enough for P and PR and forward far enough for ST and T.
BEAT_WINDOW_S = (-0.25, 0.45)
# 10 ms a point: fine enough that a Q wave and the width of a QRS survive, and
# still only a few hundred numbers for the whole record.
BEAT_POINTS = 70
MIN_BEATS_FOR_MEDIAN = 3


@dataclass(frozen=True)
class ExportReport:
    payload: dict[str, object]
    probability: float


def _decimate(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, float]:
    if sample_rate_hz <= EXPORT_DECIMATION_HZ or values.size < 4:
        return values, sample_rate_hz
    factor = int(np.ceil(sample_rate_hz / EXPORT_DECIMATION_HZ))
    return values[::factor], sample_rate_hz / factor


def _decimation_factor(sample_rate_hz: float, size: int) -> int:
    if sample_rate_hz <= EXPORT_DECIMATION_HZ or size < 4:
        return 1
    return int(np.ceil(sample_rate_hz / EXPORT_DECIMATION_HZ))


def _sample_status(state: ECGPipelineState, signal) -> np.ndarray:
    """Per-image-column provenance for one exported lead signal."""
    status = np.full(signal.values_mv.size, "outside", dtype="<U12")
    start = int(np.clip(signal.start_sample, 0, status.size))
    end = int(np.clip(signal.end_sample, start, status.size))
    status[start:end] = "measured"
    path = next((item for item in state.paths if item.index == signal.index), None)
    if path is None:
        return status
    if path.unreadable is not None:
        unreadable = path.unreadable[: status.size]
        status[np.flatnonzero(unreadable)] = "unreadable"
    if path.recovered is not None:
        recovered = path.recovered[: status.size]
        status[np.flatnonzero(recovered)] = "interpolated"
    return status


@dataclass(frozen=True)
class MedianBeat:
    """One lead's averaged complex, on a fixed time grid around the R peak."""

    label: str
    index: int
    values: np.ndarray
    beats_used: int
    confidence: float
    # Extremes of the complex before it was resampled for display. Point
    # sampling a narrow QRS clips its peak, so an amplitude that is going to be
    # compared with a printed one has to be taken at full resolution.
    peak_mv: float = 0.0
    trough_mv: float = 0.0


def median_beats(state: ECGPipelineState, points: int = BEAT_POINTS) -> tuple[MedianBeat, ...]:
    """Median complex per lead, resampled onto a common window around R.

    Taken from the optimised signal rather than from stage 12's template,
    because the template is built on the provisional deflection and is in
    pixels: what a reader needs is millivolts, on the same footing as the
    amplitudes reported next to it.
    """
    beats: list[MedianBeat] = []
    for signal in state.lead_signals:
        rate = float(signal.sample_rate_hz)
        peaks = np.asarray(state.beats_per_lead.get(signal.index, ()), dtype=np.int64)
        if rate <= 1 or peaks.size < MIN_BEATS_FOR_MEDIAN:
            continue
        pre = int(round(-BEAT_WINDOW_S[0] * rate))
        post = int(round(BEAT_WINDOW_S[1] * rate))
        values = signal.values_mv
        windows = [
            values[peak - pre : peak + post]
            for peak in peaks
            if peak - pre >= signal.start_sample and peak + post <= signal.end_sample
        ]
        if len(windows) < MIN_BEATS_FOR_MEDIAN:
            continue
        stacked = np.stack(windows).astype(np.float32)
        median = np.median(stacked, axis=0)
        # Report the deflection against this lead's own isoelectric level, so a
        # baseline that wandered does not read as ST shift.
        median = median - float(np.median(median[: max(1, pre // 3)]))
        # Average within each output step before sampling it. Plain point
        # sampling of a QRS at half the rate drops whichever pixel held the peak
        # and turns one spike into a ragged pair.
        factor = median.size / max(1, points)
        smoothed = moving_average(median, int(round(factor))) if factor > 1.5 else median
        grid = np.linspace(0.0, median.size - 1, points)
        window = signal.confidence[signal.start_sample : signal.end_sample]
        beats.append(
            MedianBeat(
                label=signal.label,
                index=signal.index,
                values=np.interp(grid, np.arange(smoothed.size), smoothed).astype(np.float32),
                beats_used=len(windows),
                confidence=float(np.mean(window)) if window.size else 0.0,
                peak_mv=float(median.max()),
                trough_mv=float(median.min()),
            )
        )
    return tuple(beats)


def signal_text(state: ECGPipelineState) -> str:
    """The digitised waveform, compact enough to put in a prompt.

    The point of sending this alongside the photograph is that a reader can
    measure a number off it that no amount of looking at a low-resolution phone
    picture will give. The point of the preamble is that the numbers are a
    reconstruction and have to be read as one.
    """
    if not bool(state.signals.get("calibrated", False)):
        return ""
    beats = median_beats(state)
    if not beats:
        return ""

    gain = float(state.text.get("gain_mm_mv", 0.0) or 0.0)
    measured = bool(state.text.get("gain_measured", False))
    start, end = BEAT_WINDOW_S
    step = (end - start) / max(1, BEAT_POINTS - 1)
    lines = [
        "- Digitised signal (reconstructed from the print by the pipeline above, not a recording "
        "from the machine). Median complex per lead, in mV, sampled every "
        f"{step * 1000:.0f} ms from {start:+.2f} s to {end:+.2f} s around the R peak, "
        "with the isoelectric level of each lead subtracted.",
        f"  Amplitude scale: {gain:.1f} mm/mV, "
        + ("read from the sheet." if measured else "assumed - every mV below is wrong by "
           "whatever ratio the sheet actually printed."),
        "  Treat these as a second opinion on the picture, not as ground truth: where they "
        "disagree with what you can see, say so and trust the picture.",
        "  Format: lead [mean confidence, beats averaged]: values",
    ]
    for beat in beats:
        values = " ".join(f"{value:.2f}" for value in beat.values)
        lines.append(f"  {beat.label} [{beat.confidence:.2f}, n={beat.beats_used}]: {values}")
    return "\n".join(lines)


def build_export(state: ECGPipelineState) -> ExportReport:
    grid = state.grid
    signals_summary = dict(state.signals)
    bands = {band.index: band for band in state.bands}
    median_by_index = {beat.index: beat for beat in median_beats(state)}
    leads: list[dict[str, object]] = []
    for signal in state.lead_signals:
        traced = signal.traced
        window = signal.confidence[signal.start_sample : signal.end_sample]
        traced_status = _sample_status(state, signal)[signal.start_sample : signal.end_sample]
        source_values = traced if traced.size else signal.values_mv
        source_confidence = window if traced.size else signal.confidence
        source_status = traced_status if traced.size else _sample_status(state, signal)
        values, rate = _decimate(source_values, signal.sample_rate_hz)
        factor = _decimation_factor(signal.sample_rate_hz, source_values.size)
        band = bands.get(signal.index)
        median = median_by_index.get(signal.index)
        leads.append(
            {
                "label": signal.label,
                "index": signal.index,
                "sample_rate_hz": round(float(rate), 3),
                "amplitude_mv": round(float(signal.amplitude_mv), 4),
                "confidence": round(float(np.mean(window)) if window.size else 0.0, 3),
                "probability": round(float(signal.probability), 3),
                "traced_columns": [int(signal.start_sample), int(signal.end_sample)],
                "line_start_probability": round(float(signal.start_probability), 3),
                "line_end_probability": round(float(signal.end_probability), 3),
                # Where on the picture this lead was read from, so a sample can
                # be pointed at on the sheet it came from.
                "band": (
                    {
                        "top": int(band.top),
                        "bottom": int(band.bottom),
                        "baseline": round(float(band.baseline), 2),
                    }
                    if band is not None
                    else None
                ),
                "r_peak_columns": [
                    int(peak) for peak in state.beats_per_lead.get(signal.index, ())
                ],
                "median_beat_mv": (
                    [round(float(value), 4) for value in median.values]
                    if median is not None
                    else None
                ),
                "samples": [round(float(value), 4) for value in values],
                "sample_confidence": [
                    round(float(value), 3) for value in source_confidence[::factor]
                ],
                "sample_status": source_status[::factor].tolist(),
            }
        )

    rhythm: dict[str, object] = {"measured": False}
    if state.beats is not None and state.beats.probability >= MIN_RATE_CONFIDENCE:
        rhythm = {
            "measured": True,
            "heart_rate_bpm": round(float(state.beats.heart_rate_bpm), 1),
            "beat_count": int(state.beats.count),
            "rr_regularity": round(float(state.beats.regularity), 3),
            "source_lead": state.beats.source_lead,
            "confidence": round(float(state.beats.probability), 3),
        }
    elif state.beats is not None:
        rhythm = {
            "measured": False,
            "reason": "beat detection confidence below reporting threshold",
            "confidence": round(float(state.beats.probability), 3),
        }

    payload: dict[str, object] = {
        "schema": "miocard.pipeline/1",
        "image": {
            "height": int(state.image_shape[0]),
            "width": int(state.image_shape[1]),
        },
        "grid": (
            {
                "px_per_mm": round(float(grid.small_cell_px), 3),
                "large_cell_px": round(float(grid.large_cell_px), 3),
                "confidence": grid.confidence,
                "axis_ratio": round(float(grid.axis_ratio), 4),
            }
            if grid is not None
            else None
        ),
        "layout": {
            "rows": int(state.layout.get("row_count", 0) or 0),
            "row_pitch_px": round(float(state.layout.get("row_pitch_px", 0.0) or 0.0), 2),
            "columns": int(state.layout.get("column_blocks", 1) or 1),
            "assumed_lead_order": bool(state.layout.get("assumed_lead_order", False)),
            "rows_short_of_twelve": int(state.layout.get("rows_short_of_twelve", 0) or 0),
        },
        "paper": {
            "speed_mm_s": float(state.text.get("paper_speed_mm_s", 0.0) or 0.0),
            "gain_mm_mv": float(state.text.get("gain_mm_mv", 0.0) or 0.0),
            "gain_measured": bool(state.text.get("gain_measured", False)),
        },
        "rhythm": rhythm,
        "quality": {
            "grade": state.quality_grade,
            "probability": round(float(state.quality.get("probability", 0.0) or 0.0), 3),
            "sub_scores": {
                name: round(float(value), 3)
                for name, value in (state.quality.get("sub_scores", {}) or {}).items()
            },
            "limitations": list(state.quality.get("limitations", ()) or ()),
        },
        "signals": {
            "units": signals_summary.get("units", "px"),
            "calibrated": bool(signals_summary.get("calibrated", False)),
            "median_beat": {
                "window_s": list(BEAT_WINDOW_S),
                "points": BEAT_POINTS,
            },
            "leads": leads,
        },
        # Kept as objects rather than erased from the picture, which is the
        # whole premise of the cost map; a consumer of this file should be able
        # to see what the pipeline decided was print rather than signal.
        "text_objects": [
            {
                "kind": obj.kind,
                "box": [int(obj.left), int(obj.top), int(obj.right), int(obj.bottom)],
                "probability": round(float(obj.probability), 3),
                "text": obj.text,
                "band_index": obj.band_index,
            }
            for obj in state.text_objects
        ],
        # Maps a point of the uploaded photograph onto the picture every
        # coordinate here refers to, as a 2x3 affine in row-major order.
        "source_transform": (
            [round(float(value), 6) for value in np.asarray(state.source_transform).ravel()]
            if state.source_transform is not None
            else None
        ),
        "stages": [
            {
                "stage": stage.stage,
                "probability": round(float(stage.probability), 3),
                "confidence": stage.confidence,
                "summary": stage.summary,
                "evidence": list(stage.evidence),
            }
            for stage in state.stages
        ],
    }

    # Export is bookkeeping: it is confident when there is something to package.
    probability = clamp01(0.25 + 0.45 * clamp01(len(leads) / 12.0) + 0.30 * (1.0 if rhythm["measured"] else 0.0))
    return ExportReport(payload=payload, probability=probability)


def to_json(state: ECGPipelineState, indent: int | None = None) -> str:
    return json.dumps(build_export(state).payload, ensure_ascii=False, indent=indent)


def signals_to_csv(state: ECGPipelineState) -> str:
    """One value per image column, with confidence and provenance beside it."""
    if not state.lead_signals:
        return ""
    length = max(signal.values_mv.size for signal in state.lead_signals)
    header = ["sample", "seconds", *[signal.label for signal in state.lead_signals]]
    header.extend(f"{signal.label}_confidence" for signal in state.lead_signals)
    header.extend(f"{signal.label}_status" for signal in state.lead_signals)
    rate = state.lead_signals[0].sample_rate_hz
    statuses = {
        signal.index: _sample_status(state, signal) for signal in state.lead_signals
    }
    rows = [",".join(header)]
    for index in range(length):
        seconds = index / rate if rate > 0 else float("nan")
        cells = [str(index), f"{seconds:.5f}" if rate > 0 else ""]
        for signal in state.lead_signals:
            if index < signal.values_mv.size:
                cells.append(f"{float(signal.values_mv[index]):.5f}")
            else:
                cells.append("")
        for signal in state.lead_signals:
            cells.append(
                f"{float(signal.confidence[index]):.3f}"
                if index < signal.confidence.size
                else ""
            )
        for signal in state.lead_signals:
            status = statuses[signal.index]
            cells.append(str(status[index]) if index < status.size else "")
        rows.append(",".join(cells))
    return "\n".join(rows) + "\n"


def measurement_text(state: ECGPipelineState) -> str:
    """Measured facts for the model prompt, stated only when they are trustworthy."""
    lines: list[str] = []
    grid = state.grid
    if grid is not None and grid.confidence != "low":
        lines.append(
            f"- Measured grid: {grid.small_cell_px:.2f} px/mm "
            f"(5 mm block {grid.large_cell_px:.1f} px), confidence {grid.confidence}"
        )
    rows = int(state.layout.get("row_count", 0) or 0)
    if rows:
        order = "standard lead order assumed" if state.layout.get("assumed_lead_order") else "rows unnamed"
        lines.append(f"- Detected {rows} lead rows ({order})")
    short = int(state.layout.get("rows_short_of_twelve", 0) or 0)
    if short:
        # Otherwise a reader counts the rows that are present and concludes the
        # sheet only ever had those. Which leads are missing is not known - see
        # HANDOFF.md - so the shortfall is stated without naming them.
        lines.append(
            f"- {short} fewer rows than a twelve-lead sheet: some printed leads are absent from "
            "the reconstruction, and which ones is not established. Rows are therefore unnamed; "
            "read the lead labels from the image."
        )
    gain = float(state.text.get("gain_mm_mv", 0.0) or 0.0)
    if state.text.get("gain_measured"):
        lines.append(f"- Gain read from the printed legend: {gain:.1f} mm/mV")
    elif gain:
        lines.append(
            f"- Gain was not read from the sheet; amplitudes assume the standard "
            f"{gain:.0f} mm/mV and are wrong by that ratio if the print says otherwise"
        )
    if state.beats is not None and state.beats.probability >= MIN_RATE_CONFIDENCE:
        lines.append(
            f"- Reconstructed rate: {state.beats.heart_rate_bpm:.0f} bpm from "
            f"{state.beats.count} beats on {state.beats.source_lead} "
            f"(RR regularity {state.beats.regularity:.2f})"
        )
    else:
        lines.append("- Rate was not measured reliably from the reconstruction; read it from the image")
    limits = state.quality.get("limitations", ()) or ()
    if limits:
        lines.append("- Reconstruction limits: " + "; ".join(limits))
    return "\n".join(lines)


def export_stage(report: ExportReport) -> PipelineStageResult:
    payload = report.payload
    leads = payload["signals"]["leads"]  # type: ignore[index]
    rhythm = payload["rhythm"]  # type: ignore[index]
    return PipelineStageResult(
        "Export",
        report.probability,
        f"Packaged {len(leads)} digitised leads, stage confidences and quality limits.",
        evidence=(
            f"leads={len(leads)}",
            f"units={payload['signals']['units']}",  # type: ignore[index]
            f"rate_reported={str(bool(rhythm.get('measured'))).lower()}",
            f"stages={len(payload['stages'])}",  # type: ignore[arg-type]
        ),
    )
