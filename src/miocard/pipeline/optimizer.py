"""Stage 13 - Signal optimizer.

Stage 9 solved for continuity only. With beats and templates on the blackboard
there is more to go on, so the path is re-solved inside a narrow corridor around
itself against the full objective MyocardBot2.md asks for: image evidence,
continuity, curvature and template similarity. Carrying the previous offset in
the dynamic-programming state is what makes the curvature term exact rather than
a smoothing pass bolted on afterwards.

The stage then places the isoelectric baseline on the PR segments of the
detected beats - which leaves the ST segment where the recorder drew it - and
converts pixels to millivolts using the measured grid and gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.series import effective_trace, median_filter
from miocard.pipeline.state import (
    LeadPath,
    LeadSignal,
    LeadTemplate,
    PipelineStageResult,
    clamp01,
)

PR_START_SECONDS = 0.13
PR_END_SECONDS = 0.05
PLAUSIBLE_AMPLITUDE_MV = (0.15, 6.0)
# Large enough to dominate any real traversal cost, finite so the dynamic
# program still has a defined answer when a lead sits against the sheet edge.
OFF_SHEET_PENALTY = 1e4


@dataclass(frozen=True)
class OptimizerReport:
    signals: tuple[LeadSignal, ...]
    summary: dict[str, object]
    probability: float


def baseline_rows(
    trace_rows: np.ndarray,
    peaks: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    """Isoelectric line, anchored on PR segments when beats are known.

    A plain median filter would drag the baseline into a depressed or elevated
    ST segment and quietly erase the finding; the PR segment is isoelectric by
    definition, so anchoring there does not.
    """
    length = trace_rows.size
    if peaks.size >= 2 and sample_rate_hz > 1:
        start = int(round(PR_START_SECONDS * sample_rate_hz))
        end = int(round(PR_END_SECONDS * sample_rate_hz))
        anchors_x: list[float] = []
        anchors_y: list[float] = []
        for peak in peaks:
            low = int(peak) - start
            high = int(peak) - end
            if low < 0 or high <= low or high > length:
                continue
            anchors_x.append(float((low + high) / 2))
            anchors_y.append(float(np.median(trace_rows[low:high])))
        if len(anchors_x) >= 2:
            return np.interp(
                np.arange(length, dtype=np.float32),
                np.asarray(anchors_x, dtype=np.float32),
                np.asarray(anchors_y, dtype=np.float32),
            ).astype(np.float32)
    window = max(3, int(round(0.60 * sample_rate_hz))) if sample_rate_hz > 1 else max(
        3, length // 12
    )
    return median_filter(trace_rows, window)


def provisional_deflection(
    path: LeadPath,
    sample_rate_hz: float,
    smooth_px: int,
) -> np.ndarray:
    """Deflection in pixels, good enough to find beats with."""
    trace = effective_trace(path, smooth_px)
    baseline = baseline_rows(trace, np.zeros(0, dtype=np.int32), sample_rate_hz)
    return (baseline - trace).astype(np.float32)


def refine_paths(
    cost: np.ndarray,
    bases: np.ndarray,
    targets: np.ndarray | None,
    supports: np.ndarray | None,
    config: PipelineConfig,
) -> np.ndarray:
    """Re-solve every lead inside +/- refine_radius_px with a curvature term.

    The dynamic-programming state is the pair (offset at x-1, offset at x), so
    the second difference - the curvature - is available exactly at every
    transition instead of being approximated by smoothing afterwards. Every lead
    shares the loop over columns, which is what keeps an exact solve affordable.
    """
    leads, width = bases.shape
    radius = max(1, int(config.refine_radius_px))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    count = offsets.size
    if width < 3 or leads == 0:
        return bases.astype(np.float32)

    height = cost.shape[0]
    positions = bases[:, None, :] + offsets[None, :, None]
    exact = np.round(positions).astype(np.int32)
    rows = np.clip(exact, 0, height - 1)
    unary = cost[rows, np.arange(width)[None, None, :]].astype(np.float32)
    # An offset that would leave the sheet is not a candidate; without this the
    # refinement can walk a path off the top or bottom edge.
    unary = np.where(exact == rows, unary, unary + OFF_SHEET_PENALTY)
    if targets is not None and supports is not None:
        mismatch = np.abs(positions - targets[:, None, :]) * supports[:, None, :]
        unary = unary + config.cost_weights.template * mismatch.astype(np.float32)

    lam1 = float(config.cost_weights.continuity)
    lam2 = float(config.cost_weights.curvature)

    # accumulated[b, p, c]: best cost for a lead whose offsets at x-1 and x are p, c.
    delta = positions[:, None, :, 1] - positions[:, :, None, 0]
    accumulated = (
        unary[:, :, None, 0] + unary[:, None, :, 1] + lam1 * np.abs(delta)
    ).astype(np.float32)
    back = np.zeros((width, leads, count, count), dtype=np.int8)

    for x in range(2, width):
        previous = positions[:, :, x - 2]
        current = positions[:, :, x - 1]
        following = positions[:, :, x]
        second = (
            following[:, None, None, :]
            - 2.0 * current[:, None, :, None]
            + previous[:, :, None, None]
        )
        candidates = accumulated[:, :, :, None] + lam2 * np.abs(second)
        chosen = np.argmin(candidates, axis=1)
        back[x] = chosen.astype(np.int8)
        inner = np.take_along_axis(candidates, chosen[:, None, :, :], axis=1)[:, 0]
        first = following[:, None, :] - current[:, :, None]
        accumulated = inner + lam1 * np.abs(first) + unary[:, None, :, x]

    lead_index = np.arange(leads)
    flat = np.argmin(accumulated.reshape(leads, -1), axis=1)
    indices = np.empty((leads, width), dtype=np.int32)
    indices[:, width - 1] = flat % count
    indices[:, width - 2] = flat // count
    for x in range(width - 1, 1, -1):
        indices[:, x - 2] = back[x, lead_index, indices[:, x - 1], indices[:, x]]
    return np.clip(bases + offsets[indices], 0.0, height - 1).astype(np.float32)


def optimize_signals(
    paths: tuple[LeadPath, ...],
    cost: np.ndarray,
    templates: dict[int, LeadTemplate],
    template_targets: dict[int, tuple[np.ndarray, np.ndarray]],
    peaks_by_lead: dict[int, np.ndarray],
    px_per_mm: float,
    gain_mm_mv: float,
    sample_rate_hz: float,
    config: PipelineConfig,
) -> OptimizerReport:
    calibrated = px_per_mm > 1.0 and gain_mm_mv > 0.0
    scale = 1.0 / (px_per_mm * gain_mm_mv) if calibrated else 1.0
    smooth_px = max(3, int(round(px_per_mm))) if px_per_mm > 1 else 3

    if not paths:
        return OptimizerReport((), {"units": "none", "calibrated": False}, 0.05)

    width = cost.shape[1]
    bases = np.stack([path.y.astype(np.float32) for path in paths])
    targets = np.zeros((len(paths), width), dtype=np.float32)
    supports = np.zeros((len(paths), width), dtype=np.float32)
    for row, path in enumerate(paths):
        if path.index not in template_targets:
            continue
        values, support = template_targets[path.index]
        # The template lives in deflection space; turn it back into rows.
        provisional = effective_trace(path, smooth_px)
        reference = baseline_rows(
            provisional, peaks_by_lead.get(path.index, np.zeros(0, np.int32)), sample_rate_hz
        )
        targets[row] = (reference - values).astype(np.float32)
        supports[row] = support.astype(np.float32)

    refined = refine_paths(cost, bases, targets, supports, config)
    movements = [float(np.mean(np.abs(refined[row] - bases[row]))) for row in range(len(paths))]

    signals: list[LeadSignal] = []
    for row, path in enumerate(paths):
        path.y = refined[row]

        trace = effective_trace(path, smooth_px)
        peaks = peaks_by_lead.get(path.index, np.zeros(0, dtype=np.int32))
        baseline = baseline_rows(trace, peaks, sample_rate_hz)
        deflection = (baseline - trace).astype(np.float32)
        values_mv = (deflection * scale).astype(np.float32)

        confidence = (
            path.confidence
            if path.confidence is not None
            else np.clip(path.evidence, 0.0, 1.0).astype(np.float32)
        ).copy()
        # Outside the traced span the sheet is blank margin: the solver still
        # returned a row there, but it stands for nothing.
        confidence[: path.x_start] = 0.0
        confidence[path.x_end :] = 0.0
        window = values_mv[path.x_start : path.x_end]
        if window.size < 8:
            window = values_mv
        amplitude = float(np.percentile(window, 99) - np.percentile(window, 1))
        template_quality = (
            float(np.median(templates[path.index].correlations))
            if path.index in templates
            else 0.0
        )
        plausible = (
            clamp01(
                1.0
                - abs(np.log(max(amplitude, 1e-3) / 1.2))
                / np.log(PLAUSIBLE_AMPLITUDE_MV[1] / 1.2)
            )
            if calibrated
            else 0.35
        )
        traced = confidence[path.x_start : path.x_end]
        if traced.size == 0:
            traced = confidence
        probability = clamp01(
            0.10
            + 0.40 * float(np.mean(traced))
            + 0.30 * clamp01(template_quality)
            + 0.20 * plausible
        )
        signals.append(
            LeadSignal(
                index=path.index,
                label=path.label,
                values_mv=values_mv,
                confidence=confidence.astype(np.float32),
                sample_rate_hz=float(sample_rate_hz),
                amplitude_mv=amplitude,
                probability=probability,
                start_sample=int(path.x_start),
                end_sample=int(path.x_end),
                start_probability=float(path.start_probability),
                end_probability=float(path.end_probability),
            )
        )

    if not signals:
        return OptimizerReport((), {"units": "none", "calibrated": False}, 0.05)

    mean_confidence = float(np.mean([float(np.mean(item.confidence)) for item in signals]))
    mean_probability = float(np.mean([item.probability for item in signals]))
    summary: dict[str, object] = {
        "units": "mV" if calibrated else "px",
        "calibrated": calibrated,
        "sample_rate_hz": float(sample_rate_hz),
        "gain_mm_mv": float(gain_mm_mv),
        "px_per_mm": float(px_per_mm),
        "lead_count": len(signals),
        "mean_confidence": mean_confidence,
        "mean_amplitude_mv": float(np.mean([item.amplitude_mv for item in signals])),
        "mean_refinement_px": float(np.mean(movements)) if movements else 0.0,
        "optimized_probability": mean_probability,
    }
    probability = clamp01(0.65 * mean_probability + 0.35 * mean_confidence)
    return OptimizerReport(tuple(signals), summary, probability)


def optimizer_stage(report: OptimizerReport) -> PipelineStageResult:
    summary = report.summary
    if not report.signals:
        return PipelineStageResult(
            "Signal optimizer",
            report.probability,
            "No lead could be turned into a signal.",
            evidence=("leads=0",),
        )
    return PipelineStageResult(
        "Signal optimizer",
        report.probability,
        f"{summary['lead_count']} leads refined against evidence, curvature and template, "
        f"then scaled to {summary['units']}"
        + (" using the measured grid." if summary["calibrated"] else " (grid unknown)."),
        evidence=(
            f"units={summary['units']}",
            f"sample_rate_hz={float(summary['sample_rate_hz']):.1f}",
            f"mean_amplitude_mv={float(summary['mean_amplitude_mv']):.2f}",
            f"mean_confidence={float(summary['mean_confidence']):.2f}",
            f"mean_refinement_px={float(summary['mean_refinement_px']):.2f}",
        ),
    )
