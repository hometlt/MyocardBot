"""Orchestration of the fifteen MyocardBot2 stages.

Each stage reads the blackboard, writes its own slot and records a result with a
confidence. The order is the one in MyocardBot2.md, and the dependencies are
real: the grid sets the millimetre scale that the layout, glyph and beat stages
all measure in, the cost map is what the solver walks, and the beats and
templates found on the first reconstruction are what the optimizer re-solves
against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miocard.pipeline.beats import beat_stage, detect_beats
from miocard.pipeline.config import DEFAULT_CONFIG, PipelineConfig
from miocard.pipeline.costmap import build_cost_map, cost_map_stage
from miocard.pipeline.export import build_export, export_stage
from miocard.pipeline.gaps import gap_stage, recover_gaps
from miocard.pipeline.geometry import GeometryReport, geometry_stage
from miocard.pipeline.grid import grid_probability_map, grid_stage, measure_grid
from miocard.pipeline.ink import extract_ink
from miocard.pipeline.layout import detect_column_blocks, detect_layout, layout_stage
from miocard.pipeline.loader import LoadedImage, loader_stage
from miocard.pipeline.noise import detect_noise, noise_stage
from miocard.pipeline.optimizer import (
    optimize_signals,
    optimizer_stage,
    provisional_deflection,
)
from miocard.pipeline.quality import analyze_quality, quality_stage
from miocard.pipeline.solver import solve_paths, solver_stage
from miocard.pipeline.state import ECGPipelineState
from miocard.pipeline.templates import (
    build_templates,
    reconstruct_from_template,
    template_stage,
)
from miocard.pipeline.text import analyze_text, detect_text, layout_label_anchors, text_stage
from miocard.pipeline.trace import estimate_trace, trace_stage

# A pitch this far from square means the sheet is distorted enough that turning
# pixels into seconds would invent precision that is not there.
CALIBRATION_AXIS_RATIO = (0.85, 1.18)


@dataclass(frozen=True)
class Preprocessing:
    """What stages 1 and 2 produced, before the model-facing rescale."""

    loaded: LoadedImage
    geometry: GeometryReport


def run_pipeline(
    image: np.ndarray,
    preprocessing: Preprocessing,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> ECGPipelineState:
    state = ECGPipelineState(image_shape=image.shape[:3], image=image)

    # 1. Image loader ------------------------------------------------------
    state.record(loader_stage(preprocessing.loaded))

    # 2. Geometry analyzer -------------------------------------------------
    report = preprocessing.geometry
    state.geometry = {
        "deskew_angle_deg": report.angle_deg,
        "angle_sharpness": report.angle_sharpness,
        "rotation_applied": report.rotation_applied,
        "crop_applied": report.crop_applied,
        "retained_fraction": report.retained_fraction,
        "brightness": report.brightness,
        "contrast": report.contrast,
        "blur_score": report.blur_score,
        "source_size": (preprocessing.loaded.source_width, preprocessing.loaded.source_height),
    }
    # The caller may have rescaled the deskewed sheet to fit a model's image
    # profile; that is the last step between the photograph and these pixels.
    # Stage 1 may also have dropped dead border lines, which shifts the origin.
    transform = report.source_matrix(image.shape[1] / max(1, report.width))
    origin = np.asarray(preprocessing.loaded.origin, dtype=np.float64)
    transform[:, 2] -= transform[:, :2] @ origin
    state.source_transform = transform
    state.record(geometry_stage(report))

    # 3. Grid detector -----------------------------------------------------
    grid = measure_grid(image, config)
    grid_prior = grid_probability_map(image, grid)
    state.grid = grid
    state.record(grid_stage(grid))

    px_per_mm = float(grid.small_cell_px) if grid is not None else 0.0
    calibrated = grid is not None and (
        CALIBRATION_AXIS_RATIO[0] <= grid.axis_ratio <= CALIBRATION_AXIS_RATIO[1]
    )

    # 4. Layout detector ---------------------------------------------------
    layers = extract_ink(image, px_per_mm if px_per_mm > 1 else None)
    # Stage 5's glyph pass runs first even though its result is reported in
    # plan order: the layout detector needs to know which image rows are
    # printed text, or it will mistake the header block for a lead baseline.
    detection = detect_text(layers, px_per_mm, config)
    label_anchors = layout_label_anchors(detection, image.shape[1], px_per_mm)
    bands, layout_summary = detect_layout(
        layers,
        config,
        detection.row_text_coverage,
        label_anchors,
    )
    layout_summary["column_blocks"] = detect_column_blocks(layers, bands)
    state.bands = bands
    state.layout = layout_summary
    state.record(layout_stage(bands, layout_summary))

    # 5. OCR & semantic analysis ------------------------------------------
    semantics = analyze_text(image, detection, bands, px_per_mm, config, layers)
    config = config.with_paper(semantics.paper_speed_mm_s, semantics.gain_mm_mv)
    state.text_objects = semantics.objects
    state.text = {
        "object_count": len(semantics.objects),
        "line_count": semantics.line_count,
        "labels_found": semantics.labels_found,
        "calibration_marks": semantics.calibration_marks,
        "margin_edges": dict(semantics.label_columns),
        "glyph_count": detection.glyph_count,
        "ocr_used": semantics.ocr_used,
        "paper_speed_mm_s": semantics.paper_speed_mm_s,
        "gain_mm_mv": semantics.gain_mm_mv,
        "gain_measured": semantics.gain_measured,
        "notes": semantics.notes,
    }
    state.record(text_stage(semantics))

    # 6. ECG trace probability estimator ----------------------------------
    trace = estimate_trace(layers, px_per_mm if px_per_mm > 1 else 3.0)
    state.trace_probability = dict(trace.statistics)
    state.record(trace_stage(trace))

    # 7. Noise / artifact detector ----------------------------------------
    noise = detect_noise(
        image,
        layers,
        trace.evidence,
        px_per_mm if px_per_mm > 1 else 3.0,
        report.blur_score,
        report.contrast,
    )
    state.noise = dict(noise.noise_statistics)
    state.artifacts = dict(noise.artifact_statistics)
    state.record(noise_stage(noise))

    # 8. Cost map builder --------------------------------------------------
    cost = build_cost_map(
        layers,
        grid,
        grid_prior,
        trace.evidence,
        semantics.text_map,
        noise.noise,
        noise.artifact,
        config,
    )
    state.probability_maps = cost.classes
    state.cost_map = {
        "unary": cost.unary,
        "signal_probability": cost.summary["trace"],
        "line_probability": cost.summary["trace"],
        "grid_probability": cost.summary["grid"],
        "text_probability": cost.summary["text"],
        "noise_probability": cost.summary["noise"],
        "artifact_probability": cost.summary["artifact"],
        "background_probability": cost.summary["background"],
        "unary_mean": cost.summary["unary_mean"],
        "weights": config.cost_weights,
    }
    state.record(cost_map_stage(cost))

    # 9. Path solver -------------------------------------------------------
    solver = solve_paths(
        cost.unary,
        cost.classes["trace"],
        bands,
        config,
        px_per_mm,
        semantics.label_columns,
        cost.classes["grid"],
    )
    state.paths = solver.paths
    state.record(solver_stage(solver))

    # 10. Gap recovery -----------------------------------------------------
    gaps = recover_gaps(
        solver.paths,
        cost.classes,
        px_per_mm,
        config,
        semantics.objects,
    )
    state.record(gap_stage(gaps))

    # 11. Beat detector ----------------------------------------------------
    sample_rate_hz = px_per_mm * config.paper_speed_mm_s if calibrated else 0.0
    smooth_px = max(3, int(round(px_per_mm))) if px_per_mm > 1 else 3
    provisional = {
        path.index: provisional_deflection(path, sample_rate_hz, smooth_px)
        for path in solver.paths
    }
    labels = {path.index: path.label for path in solver.paths}
    beats = detect_beats(provisional, labels, sample_rate_hz, config)
    state.beats = beats.beats
    state.beats_per_lead = dict(beats.per_lead)
    state.record(beat_stage(beats))

    # 12. Template builder -------------------------------------------------
    templates = build_templates(provisional, beats.per_lead, labels, sample_rate_hz, config)
    state.templates = templates.templates
    state.record(template_stage(templates))

    # 13. Signal optimizer -------------------------------------------------
    template_by_index = {item.index: item for item in templates.templates}
    template_targets = {
        index: reconstruct_from_template(
            template, beats.per_lead.get(index, np.zeros(0, dtype=np.int32)), image.shape[1]
        )
        for index, template in template_by_index.items()
    }
    optimized = optimize_signals(
        solver.paths,
        cost.unary,
        template_by_index,
        template_targets,
        beats.per_lead,
        px_per_mm if calibrated else 0.0,
        config.gain_mm_mv,
        sample_rate_hz,
        config,
    )
    state.lead_signals = optimized.signals
    state.signals = dict(optimized.summary)
    state.record(optimizer_stage(optimized))

    # 14. Quality analyzer -------------------------------------------------
    state.quality = analyze_quality(
        geometry_probability=report.probability,
        grid_probability=grid.probability if grid is not None else 0.05,
        layout_probability=state.stages[3].probability,
        trace_probability=trace.probability,
        cost_probability=cost.probability,
        gap_statistics=gaps.statistics,
        beat_probability=beats.probability,
        template_statistics=templates.statistics,
        signal_summary=optimized.summary,
        artifact_statistics=noise.artifact_statistics,
        gain_measured=semantics.gain_measured,
    )
    state.record(quality_stage(state.quality))

    # 15. Export -----------------------------------------------------------
    export = build_export(state)
    state.record(export_stage(export))
    # Rebuilt so the payload contains the export stage itself.
    state.signals["export"] = build_export(state).payload
    return state
