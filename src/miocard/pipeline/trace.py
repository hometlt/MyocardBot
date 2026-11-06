"""Stage 6 - ECG trace probability estimator.

Two things distinguish pen ink from everything else on the sheet: it is much
darker than the printed grid, and it is horizontally continuous because it was
drawn by a pen moving across moving paper. Neither test alone is enough - a
letter is dark, a grid line is continuous - so the estimator multiplies them.

The output is evidence in [0, 1], not a posterior. Deciding between the classes
is the cost map builder's job in stage 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from miocard.pipeline.ink import InkLayers
from miocard.pipeline.state import PipelineStageResult, clamp01


@dataclass(frozen=True)
class TraceReport:
    evidence: np.ndarray
    continuity: np.ndarray
    statistics: dict[str, float]
    probability: float


def _darkness_score(layers: InkLayers) -> np.ndarray:
    """Logistic on ink strength, centred on the ink/grid split found by Otsu."""
    centre = layers.threshold
    slope = max(0.045, centre * 0.30)
    return (1.0 / (1.0 + np.exp(-(layers.ink - centre) / slope))).astype(np.float32)


def _continuity_score(layers: InkLayers, px_per_mm: float) -> np.ndarray:
    """How much the ink at a pixel is part of a *stroke*, in any direction.

    A pen stroke is elongated along its own direction, and the direction changes
    constantly: the isoelectric segment runs horizontally, the flank of a QRS
    runs very nearly vertically, and both are the same pen. Measuring elongation
    along one axis only - as an earlier version did, horizontally - suppresses
    exactly the steep edges that carry the amplitude, breaking the ink run at
    every R wave and truncating the reconstruction there.

    Taking the strongest response over four orientations keeps the test that
    matters, which is elongation against an isolated speck of JPEG noise or a
    grid intersection: those are elongated along no direction at all.
    """
    span = max(3, int(round(1.2 * max(px_per_mm, 3.0))))
    if span % 2 == 0:
        span += 1
    line = cv2.getGaussianKernel(span, span / 4.0).ravel().astype(np.float32)

    kernels = [line.reshape(1, -1), line.reshape(-1, 1)]
    # The two diagonals, so a stroke at 45 degrees is not left in between.
    diagonal = np.zeros((span, span), dtype=np.float32)
    diagonal[np.arange(span), np.arange(span)] = line
    kernels.append(diagonal)
    kernels.append(diagonal[:, ::-1].copy())

    score = np.zeros_like(layers.ink)
    for kernel in kernels:
        np.maximum(score, cv2.filter2D(layers.ink, -1, kernel), out=score)
    reference = max(float(np.percentile(score, 99.0)), 1e-3)
    return np.clip(score / reference, 0.0, 1.0).astype(np.float32)


def estimate_trace(layers: InkLayers, px_per_mm: float) -> TraceReport:
    darkness = _darkness_score(layers)
    continuity = _continuity_score(layers, px_per_mm)
    evidence = np.clip(darkness * (0.35 + 0.65 * continuity), 0.0, 1.0).astype(np.float32)

    statistics = {
        "p50": float(np.percentile(evidence, 50)),
        "p75": float(np.percentile(evidence, 75)),
        "p95": float(np.percentile(evidence, 95)),
        "mean": float(evidence.mean()),
        "coverage": float((evidence > 0.5).mean()),
        "ink_threshold": float(layers.threshold),
        "ink_scale": float(layers.scale),
    }

    # A readable sheet devotes a small but non-trivial share of its pixels to
    # pen ink. Far below that means faded print; far above means the image is
    # too dark to tell ink from paper.
    coverage = statistics["coverage"]
    coverage_score = clamp01(1.0 - abs(np.log(max(coverage, 1e-4) / 0.03)) / 2.2)
    contrast_score = clamp01((statistics["p95"] - statistics["p50"]) / 0.55)
    probability = clamp01(0.12 + 0.48 * coverage_score + 0.40 * contrast_score)
    return TraceReport(
        evidence=evidence,
        continuity=continuity,
        statistics=statistics,
        probability=probability,
    )


def trace_stage(report: TraceReport) -> PipelineStageResult:
    return PipelineStageResult(
        "Trace probability estimator",
        report.probability,
        "Pen-ink likelihood from darkness against the local paper and "
        "horizontal stroke continuity.",
        evidence=(
            f"coverage={report.statistics['coverage']:.4f}",
            f"p95={report.statistics['p95']:.3f}",
            f"p50={report.statistics['p50']:.3f}",
            f"ink_threshold={report.statistics['ink_threshold']:.2f}",
        ),
    )
