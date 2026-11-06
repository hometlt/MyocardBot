"""MyocardBot2 ECG reconstruction pipeline.

Fifteen replaceable stages write to one blackboard, in the order given by
MyocardBot2.md. Digitisation is treated as probabilistic path reconstruction:
the image contributes a per-pixel distribution over trace/grid/text/noise/
artifact/background, and the pen position is the minimum-cost path through it.
Nothing is erased - text and grid become costs, not deletions - and every stage
publishes a confidence next to its result.
"""

from __future__ import annotations

from miocard.pipeline.config import ClassWeights, CostWeights, PipelineConfig
from miocard.pipeline.export import (
    build_export,
    measurement_text,
    median_beats,
    signal_text,
    signals_to_csv,
    to_json,
)
from miocard.pipeline.geometry import (
    GeometryReport,
    analyze_geometry,
    crop_empty_margins,
    map_source_points,
)
from miocard.pipeline.grid import GridMetrics, grid_probability_map, measure_grid
from miocard.pipeline.gridscan import GridScan, GridScanConfig, TileGrid, scan_grid
from miocard.pipeline.loader import ImageProcessingError, LoadedImage, load_image
from miocard.pipeline.runner import Preprocessing, run_pipeline
from miocard.pipeline.state import (
    CLASS_NAMES,
    STAGE_NAMES,
    BeatSet,
    ECGPipelineState,
    LeadBand,
    LeadPath,
    LeadSignal,
    LeadTemplate,
    PipelineStageResult,
    TextObject,
)

__all__ = [
    "CLASS_NAMES",
    "STAGE_NAMES",
    "BeatSet",
    "ClassWeights",
    "CostWeights",
    "ECGPipelineState",
    "GeometryReport",
    "GridMetrics",
    "GridScan",
    "GridScanConfig",
    "ImageProcessingError",
    "LeadBand",
    "LeadPath",
    "LeadSignal",
    "LeadTemplate",
    "LoadedImage",
    "PipelineConfig",
    "PipelineStageResult",
    "Preprocessing",
    "TextObject",
    "TileGrid",
    "analyze_geometry",
    "build_export",
    "crop_empty_margins",
    "grid_probability_map",
    "load_image",
    "map_source_points",
    "measure_grid",
    "measurement_text",
    "median_beats",
    "run_pipeline",
    "scan_grid",
    "signal_text",
    "signals_to_csv",
    "to_json",
]
