"""Blackboard state shared by every MyocardBot2 pipeline stage.

MyocardBot2.md asks for a blackboard architecture: a stage reads what earlier
stages published, writes its own result next to it, and always publishes a
confidence alongside the result. Nothing is ever destructively removed from the
picture - text and grid become probability layers instead of being erased.

The dataclasses here are deliberately plain: a stage is replaceable as long as
it fills in the same slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Per-pixel classes of the cost map. MyocardBot2.md lists trace/text/grid/noise/
# artifact; ``background`` is the sixth class that lets the five sum to one
# instead of being renormalised against nothing on blank paper.
CLASS_NAMES: tuple[str, ...] = ("trace", "grid", "text", "noise", "artifact", "background")

STAGE_NAMES: tuple[str, ...] = (
    "Image loader",
    "Geometry analyzer",
    "Grid detector",
    "Layout detector",
    "OCR & semantic analysis",
    "Trace probability estimator",
    "Noise / artifact detector",
    "Cost map builder",
    "Path solver",
    "Gap recovery",
    "Beat detector",
    "Template builder",
    "Signal optimizer",
    "Quality analyzer",
    "Export",
)

# Standard clinical paper settings. They are defaults, not assumptions: stage 5
# overrides them whenever it reads the printed speed/gain legend.
DEFAULT_PAPER_SPEED_MM_S = 25.0
DEFAULT_GAIN_MM_MV = 10.0


def clamp01(value: float) -> float:
    """Clip to a probability range, mapping NaN to zero rather than propagating it."""
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def confidence_label(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


@dataclass(frozen=True)
class PipelineStageResult:
    """What a stage published: an outcome, how much it is trusted, and why."""

    stage: str
    probability: float
    summary: str
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def confidence(self) -> str:
        return confidence_label(self.probability)

    def format(self) -> str:
        details = ", ".join(self.evidence) if self.evidence else self.summary
        return f"- {self.stage}: {self.confidence} (p={self.probability:.2f}) {details}"


@dataclass(frozen=True)
class LeadBand:
    """One horizontal strip of the sheet that carries a single lead."""

    index: int
    top: int
    bottom: int
    baseline: float
    ink: float
    label: str = ""

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


@dataclass(frozen=True)
class TextObject:
    """A glyph-like blob kept as a semantic object, never painted over.

    ``kind`` is inferred from where the blob sits relative to the lead bands, so
    a lead label at the left edge can penalise the cost map differently from a
    footer legend that no trace will ever cross.
    """

    left: int
    top: int
    right: int
    bottom: int
    kind: str
    probability: float
    text: str = ""
    band_index: int | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass
class LeadPath:
    """The reconstructed pen position for one lead, one y per image column."""

    index: int
    label: str
    band: LeadBand
    y: np.ndarray
    evidence: np.ndarray
    top_envelope: np.ndarray
    bottom_envelope: np.ndarray
    x_start: int
    x_end: int
    mean_cost: float
    # How sharply the trace begins and ends at those columns. A printed lead
    # starts abruptly on blank paper; a low value means the span was guessed
    # from fading evidence rather than seen.
    start_probability: float = 0.0
    end_probability: float = 0.0
    # Filled in by gap recovery (stage 10).
    confidence: np.ndarray | None = None
    recovered: np.ndarray | None = None
    # Samples for which the solver has a best guess but the photograph does
    # not contain enough unambiguous trace evidence to call it measured.
    unreadable: np.ndarray | None = None
    gaps: tuple[tuple[int, int, str], ...] = ()

    @property
    def width(self) -> int:
        return int(self.y.size)


@dataclass
class BeatSet:
    """R-peak positions and the rhythm statistics derived from them."""

    r_peaks: np.ndarray
    rr_seconds: np.ndarray
    heart_rate_bpm: float
    regularity: float
    probability: float
    source_lead: str = ""

    @property
    def count(self) -> int:
        return int(self.r_peaks.size)


@dataclass
class LeadTemplate:
    """Median beat for one lead plus how well each beat matches it."""

    index: int
    label: str
    samples_mv: np.ndarray
    correlations: np.ndarray
    offset_samples: int
    probability: float


@dataclass
class LeadSignal:
    """Digitised lead: millivolts over time with a per-sample confidence."""

    index: int
    label: str
    values_mv: np.ndarray
    confidence: np.ndarray
    sample_rate_hz: float
    amplitude_mv: float
    probability: float
    # Sample range that actually carries a printed trace; outside it the array
    # keeps its image-column alignment but means nothing.
    start_sample: int = 0
    end_sample: int = 0
    # How sharply the printed line begins and ends at those samples.
    start_probability: float = 0.0
    end_probability: float = 0.0

    @property
    def traced(self) -> np.ndarray:
        return self.values_mv[self.start_sample : self.end_sample]


@dataclass
class ECGPipelineState:
    """Blackboard written by the fifteen MyocardBot2 stages, in order."""

    image_shape: tuple[int, int, int]
    image: np.ndarray | None = None
    geometry: dict[str, object] = field(default_factory=dict)
    # 2x3 affine taking a point of the photograph as it was decoded onto
    # ``image``. Deskew, crop and the model-profile rescale all happen before
    # stage 3, so without this nothing measured here can be pointed at on the
    # picture the user actually sent.
    source_transform: np.ndarray | None = None
    grid: object | None = None
    layout: dict[str, object] = field(default_factory=dict)
    text: dict[str, object] = field(default_factory=dict)
    trace_probability: dict[str, object] = field(default_factory=dict)
    noise: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)
    cost_map: dict[str, object] = field(default_factory=dict)
    paths: tuple[LeadPath, ...] = ()
    signals: dict[str, object] = field(default_factory=dict)
    quality: dict[str, object] = field(default_factory=dict)
    probability_maps: dict[str, np.ndarray] = field(default_factory=dict)
    bands: tuple[LeadBand, ...] = ()
    text_objects: tuple[TextObject, ...] = ()
    beats: BeatSet | None = None
    # R-peak columns per lead index. The reported rhythm comes from one lead,
    # but every lead was searched, and marking the complexes a reader can check
    # needs all of them.
    beats_per_lead: dict[int, np.ndarray] = field(default_factory=dict)
    templates: tuple[LeadTemplate, ...] = ()
    lead_signals: tuple[LeadSignal, ...] = ()
    stages: tuple[PipelineStageResult, ...] = field(default_factory=tuple)

    def record(self, result: PipelineStageResult) -> PipelineStageResult:
        self.stages = (*self.stages, result)
        return result

    def stage(self, name: str) -> PipelineStageResult | None:
        for result in self.stages:
            if result.stage == name:
                return result
        return None

    @property
    def quality_grade(self) -> str:
        return str(self.quality.get("grade", "limited"))

    @property
    def unary_cost(self) -> np.ndarray | None:
        cost = self.cost_map.get("unary")
        return cost if isinstance(cost, np.ndarray) else None

    def metadata_text(self) -> str:
        """Compact stage notes for the model prompt."""
        lines = [
            f"- Pipeline quality grade: {self.quality_grade}",
            "- Processing notes (MyocardBot2):",
        ]
        lines.extend(stage.format() for stage in self.stages)
        if self.probability_maps:
            lines.append("- Probability map means:")
            lines.extend(
                f"  - {name}: {float(np.asarray(probability).mean()):.3f}"
                for name, probability in sorted(self.probability_maps.items())
            )
        return "\n".join(lines) + "\n"
