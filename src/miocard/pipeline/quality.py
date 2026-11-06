"""Stage 14 - Quality analyzer.

The grade is a weighted combination of things that were actually measured, and
each sub-score is kept so a poor grade can be explained rather than merely
announced. The weighting is deliberately unforgiving about the two failures that
make a digitisation misleading rather than merely rough: an unknown grid, which
makes every interval wrong, and large unrecovered gaps, which make a lead's
shape partly invented.
"""

from __future__ import annotations

from miocard.pipeline.state import PipelineStageResult, clamp01

GRADE_THRESHOLDS = ((0.72, "diagnostic"), (0.44, "limited"))

SUB_SCORE_WEIGHTS = {
    "geometry": 0.10,
    "grid": 0.20,
    "layout": 0.15,
    "evidence": 0.15,
    "completeness": 0.20,
    "consistency": 0.20,
}


def analyze_quality(
    geometry_probability: float,
    grid_probability: float,
    layout_probability: float,
    trace_probability: float,
    cost_probability: float,
    gap_statistics: dict[str, float],
    beat_probability: float,
    template_statistics: dict[str, float],
    signal_summary: dict[str, object],
    artifact_statistics: dict[str, float],
    gain_measured: bool,
) -> dict[str, object]:
    completeness = clamp01(1.0 - 2.0 * float(gap_statistics.get("gap_fraction", 1.0)))
    consistency = clamp01(
        0.40 * beat_probability
        + 0.40 * clamp01(float(template_statistics.get("median_correlation", 0.0)))
        + 0.20 * clamp01(1.0 - float(template_statistics.get("outlier_fraction", 1.0)))
    )
    evidence = clamp01(0.5 * trace_probability + 0.5 * cost_probability)
    artifact_load = clamp01(float(artifact_statistics.get("mean", 0.0)) * 4.0)

    sub_scores = {
        "geometry": clamp01(geometry_probability),
        "grid": clamp01(grid_probability),
        "layout": clamp01(layout_probability),
        "evidence": evidence,
        "completeness": completeness,
        "consistency": consistency,
    }
    overall = sum(SUB_SCORE_WEIGHTS[name] * value for name, value in sub_scores.items())
    # An artifact-covered sheet cannot be graded above what the visible part
    # supports, however well the visible part reconstructed.
    overall = clamp01(overall * (1.0 - 0.35 * artifact_load))
    if not signal_summary.get("calibrated", False):
        overall = min(overall, 0.43)

    limits: list[str] = []
    if sub_scores["grid"] < 0.45:
        limits.append("grid pitch uncertain, intervals unreliable")
    if completeness < 0.6:
        limits.append("parts of the trace had no image evidence")
    if consistency < 0.45:
        limits.append("beat-to-beat shape did not repeat")
    if artifact_load > 0.35:
        limits.append("glare, shadow or blur covers part of the sheet")
    if sub_scores["layout"] < 0.45:
        limits.append("lead rows could not be separated cleanly")
    if not gain_measured:
        # Timing comes from the measured grid, but amplitude needs the gain, and
        # a sheet printed at 5 mm/mV read as 10 halves every millivolt reported.
        limits.append("gain was not read from the sheet, so amplitudes in mV are assumed")

    grade = "non_diagnostic"
    for threshold, name in GRADE_THRESHOLDS:
        if overall >= threshold:
            grade = name
            break
    # Calling a reconstruction diagnostic while listing reasons not to trust it
    # would contradict this stage's own findings.
    if grade == "diagnostic" and limits:
        grade = "limited"

    return {
        "grade": grade,
        "probability": float(overall),
        "sub_scores": sub_scores,
        "artifact_load": float(artifact_load),
        "limitations": tuple(limits),
    }


def quality_stage(quality: dict[str, object]) -> PipelineStageResult:
    sub_scores: dict[str, float] = quality["sub_scores"]  # type: ignore[assignment]
    limits = quality.get("limitations", ())
    summary = f"Overall reconstruction quality {float(quality['probability']):.2f} ({quality['grade']})."
    if limits:
        summary += " Limits: " + "; ".join(limits) + "."
    return PipelineStageResult(
        "Quality analyzer",
        float(quality["probability"]),
        summary,
        evidence=tuple(
            f"{name}={value:.2f}" for name, value in sorted(sub_scores.items())
        )
        + (f"artifact_load={float(quality['artifact_load']):.2f}",),
    )
