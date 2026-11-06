"""Stage 8 - Cost map builder.

Every pixel gets a distribution over what it is - trace, grid, text, noise,
artifact or bare paper - and a traversal cost derived from it. This is where the
"never erase" rule pays off: text and grid are still in the picture, they just
carry a price, and the path solver is free to cross them when the alternative is
worse.

The data term is -log P(trace), so the solver maximises the posterior
probability of the whole path rather than chasing the darkest pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.grid import GridMetrics
from miocard.pipeline.ink import InkLayers
from miocard.pipeline.state import CLASS_NAMES, PipelineStageResult, clamp01

# How much accumulated evidence it takes before a pixel stops being bare paper.
BACKGROUND_SCALE = 0.18
LOG_FLOOR = 1e-3


@dataclass(frozen=True)
class CostMapReport:
    classes: dict[str, np.ndarray]
    unary: np.ndarray
    summary: dict[str, float]
    probability: float


def _grid_evidence(
    grid_prior: np.ndarray,
    layers: InkLayers,
    trace_evidence: np.ndarray,
) -> np.ndarray:
    """Grid needs both the geometric model and faint ink actually being there."""
    if float(grid_prior.max()) <= 0.0:
        return np.zeros_like(layers.ink, dtype=np.float32)
    faint = np.clip(layers.ink / max(layers.threshold * 0.55, 1e-3), 0.0, 1.0)
    return np.clip(grid_prior * faint * (1.0 - 0.85 * trace_evidence), 0.0, 1.0).astype(
        np.float32
    )


def build_cost_map(
    layers: InkLayers,
    grid: GridMetrics | None,
    grid_prior: np.ndarray,
    trace_evidence: np.ndarray,
    text_evidence: np.ndarray,
    noise_evidence: np.ndarray,
    artifact_evidence: np.ndarray,
    config: PipelineConfig,
) -> CostMapReport:
    weights = config.class_weights
    grid_evidence = _grid_evidence(grid_prior, layers, trace_evidence)

    scaled = {
        "trace": weights.trace * np.clip(trace_evidence, 0.0, 1.0),
        "grid": weights.grid * grid_evidence,
        "text": weights.text * np.clip(text_evidence, 0.0, 1.0),
        "noise": weights.noise * np.clip(noise_evidence, 0.0, 1.0),
        "artifact": weights.artifact * np.clip(artifact_evidence, 0.0, 1.0),
    }
    total = sum(scaled.values())
    # Bare paper is the default explanation and loses ground smoothly as
    # evidence accumulates, so the six classes always sum to exactly one.
    background = (1.0 / (1.0 + total / BACKGROUND_SCALE)).astype(np.float32)
    share = np.where(total > 1e-9, (1.0 - background) / np.maximum(total, 1e-9), 0.0)

    classes = {name: (value * share).astype(np.float32) for name, value in scaled.items()}
    classes["background"] = background
    # Guard against float drift so the layers provably sum to one.
    stack = np.stack([classes[name] for name in CLASS_NAMES], axis=0)
    stack = np.clip(stack, 0.0, 1.0)
    stack /= np.maximum(stack.sum(axis=0, keepdims=True), 1e-9)
    classes = {name: stack[index].astype(np.float32) for index, name in enumerate(CLASS_NAMES)}

    cost_weights = config.cost_weights
    unary = (
        cost_weights.trace * -np.log(np.maximum(classes["trace"], LOG_FLOOR))
        + cost_weights.text * classes["text"]
        + cost_weights.grid * classes["grid"]
        + cost_weights.noise * classes["noise"]
        + cost_weights.artifact * classes["artifact"]
    ).astype(np.float32)

    summary = {name: float(classes[name].mean()) for name in CLASS_NAMES}
    summary["unary_mean"] = float(unary.mean())
    summary["unary_min"] = float(unary.min())
    summary["grid_model"] = 1.0 if grid is not None else 0.0

    # The map is useful when the trace class actually stands out from the paper
    # and is not smothered by text or artifact mass.
    separation = clamp01(
        (float(np.percentile(classes["trace"], 99.5)) - summary["trace"]) / 0.55
    )
    interference = clamp01((summary["text"] + summary["artifact"]) / 0.25)
    probability = clamp01(0.15 + 0.70 * separation - 0.25 * interference)
    return CostMapReport(
        classes=classes,
        unary=unary,
        summary=summary,
        probability=probability,
    )


def cost_map_stage(report: CostMapReport) -> PipelineStageResult:
    return PipelineStageResult(
        "Cost map builder",
        report.probability,
        "Per-pixel posterior over trace/grid/text/noise/artifact/background "
        "turned into a traversal cost; nothing was removed from the image.",
        evidence=(
            f"p_trace_mean={report.summary['trace']:.4f}",
            f"p_grid_mean={report.summary['grid']:.4f}",
            f"p_text_mean={report.summary['text']:.4f}",
            f"p_noise_mean={report.summary['noise']:.4f}",
            f"p_artifact_mean={report.summary['artifact']:.4f}",
            f"unary_mean={report.summary['unary_mean']:.2f}",
        ),
    )
