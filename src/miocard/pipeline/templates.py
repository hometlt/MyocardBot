"""Stage 12 - Template builder.

Beats of the same lead should look alike. Stacking them on their R peaks and
taking the median gives a template that is immune to a single bad beat, and the
correlation of each beat against that template is both a per-beat quality score
and the ectopic-beat flag.

The template is also what the signal optimizer leans on in stage 13: where the
image evidence is ambiguous, the shape the rest of the record agrees on is a
better prior than smoothness alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.state import LeadTemplate, PipelineStageResult, clamp01

MIN_BEATS_FOR_TEMPLATE = 3
OUTLIER_CORRELATION = 0.80


@dataclass(frozen=True)
class TemplateReport:
    templates: tuple[LeadTemplate, ...]
    statistics: dict[str, float]
    probability: float


def _stack_beats(
    signal: np.ndarray,
    peaks: np.ndarray,
    pre: int,
    post: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    used = []
    for peak in peaks:
        start = int(peak) - pre
        end = int(peak) + post
        if start < 0 or end > signal.size:
            continue
        windows.append(signal[start:end])
        used.append(int(peak))
    if not windows:
        return np.zeros((0, pre + post), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.stack(windows).astype(np.float32), np.asarray(used, dtype=np.int32)


def _correlations(windows: np.ndarray, template: np.ndarray) -> np.ndarray:
    centred_template = template - template.mean()
    norm = float(np.linalg.norm(centred_template))
    if norm <= 1e-9:
        return np.zeros(windows.shape[0], dtype=np.float32)
    centred = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (centred @ centred_template) / (norms * norm)
    return np.nan_to_num(values, nan=0.0).astype(np.float32)


def build_templates(
    signals: dict[int, np.ndarray],
    peaks_by_lead: dict[int, np.ndarray],
    labels: dict[int, str],
    sample_rate_hz: float,
    config: PipelineConfig,
) -> TemplateReport:
    pre = max(2, int(round(config.template_pre_seconds * sample_rate_hz)))
    post = max(2, int(round(config.template_post_seconds * sample_rate_hz)))

    templates: list[LeadTemplate] = []
    for index, signal in signals.items():
        peaks = peaks_by_lead.get(index, np.zeros(0, dtype=np.int32))
        if peaks.size < MIN_BEATS_FOR_TEMPLATE:
            continue
        windows, _ = _stack_beats(signal, peaks, pre, post)
        if windows.shape[0] < MIN_BEATS_FOR_TEMPLATE:
            continue
        template = np.median(windows, axis=0).astype(np.float32)
        correlations = _correlations(windows, template)
        probability = clamp01((float(np.median(correlations)) - 0.35) / 0.60)
        templates.append(
            LeadTemplate(
                index=index,
                label=labels.get(index, f"row {index + 1}"),
                samples_mv=template,
                correlations=correlations,
                offset_samples=pre,
                probability=probability,
            )
        )

    if not templates:
        return TemplateReport(
            templates=(),
            statistics={"lead_count": 0.0, "median_correlation": 0.0, "outlier_beats": 0.0},
            probability=0.10,
        )

    medians = np.asarray([float(np.median(item.correlations)) for item in templates])
    outliers = sum(int((item.correlations < OUTLIER_CORRELATION).sum()) for item in templates)
    total_beats = sum(int(item.correlations.size) for item in templates)
    statistics = {
        "lead_count": float(len(templates)),
        "median_correlation": float(np.median(medians)),
        "outlier_beats": float(outliers),
        "outlier_fraction": float(outliers / max(total_beats, 1)),
        "template_samples": float(pre + post),
    }
    probability = clamp01(
        0.10
        + 0.60 * clamp01((statistics["median_correlation"] - 0.40) / 0.55)
        + 0.30 * clamp01(len(templates) / max(len(signals), 1))
    )
    return TemplateReport(tuple(templates), statistics, probability)


def reconstruct_from_template(
    template: LeadTemplate,
    peaks: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lay the median beat back down at every R peak.

    Returns the reconstruction and a mask of where it actually says anything, so
    the optimizer only trusts it inside the beats it was built from.
    """
    values = np.zeros(length, dtype=np.float32)
    support = np.zeros(length, dtype=bool)
    window = template.samples_mv.size
    for peak in peaks:
        start = int(peak) - template.offset_samples
        end = start + window
        low = max(0, start)
        high = min(length, end)
        if high <= low:
            continue
        values[low:high] = template.samples_mv[low - start : high - start]
        support[low:high] = True
    return values, support


def template_stage(report: TemplateReport) -> PipelineStageResult:
    if not report.templates:
        return PipelineStageResult(
            "Template builder",
            report.probability,
            "Too few clean beats to build a median beat for any lead.",
            evidence=("templates=0",),
        )
    return PipelineStageResult(
        "Template builder",
        report.probability,
        f"Median beat built for {len(report.templates)} leads; "
        f"beat-to-template correlation {report.statistics['median_correlation']:.2f}.",
        evidence=(
            f"templates={len(report.templates)}",
            f"median_correlation={report.statistics['median_correlation']:.3f}",
            f"outlier_beats={report.statistics['outlier_beats']:.0f}",
            f"outlier_fraction={report.statistics['outlier_fraction']:.3f}",
        ),
    )
