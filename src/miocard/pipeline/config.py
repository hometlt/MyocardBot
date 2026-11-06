"""Tunables for the MyocardBot2 reconstruction pipeline.

Everything an algorithm would otherwise hard-code lives here so a stage can be
swapped or retuned without touching the code that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassWeights:
    """Relative weight of each evidence layer when the classes are normalised.

    The weights only matter against each other: a pixel that looks equally like
    ink and like grid resolves towards whichever class is weighted higher.
    """

    trace: float = 1.00
    grid: float = 0.85
    text: float = 0.95
    noise: float = 0.55
    artifact: float = 0.70


@dataclass(frozen=True)
class CostWeights:
    """How each class turns into a traversal penalty for the path solver."""

    # -log P(trace) is the data term; the rest are additive penalties so the
    # solver prefers to route around text and glare instead of through them.
    trace: float = 1.00
    # Type is ink, but no lead drew it. The penalty is set near the cost of bare
    # paper so the solver rules a straight line under a lead label rather than
    # tracing the strokes of the character, and stage 10 then reports that
    # stretch as interpolated instead of measured.
    text: float = 5.00
    grid: float = 0.45
    noise: float = 0.90
    artifact: float = 1.60
    # Continuity and curvature, in cost units per pixel of vertical movement.
    continuity: float = 0.055
    curvature: float = 0.030
    template: float = 0.45
    # Charged per *empty* pixel the pen would have to cross to change rows. A
    # genuine vertical QRS stroke inks the whole column and pays nothing; hopping
    # over blank paper onto the neighbouring lead pays for every pixel of the
    # gap, which is what stops a path swapping traces at a crossing.
    gap_crossing: float = 0.55
    # Hard cap on how far the pen may move between neighbouring columns,
    # expressed as a fraction of the lead band height.
    max_step_fraction: float = 0.60


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single reconstruction run."""

    class_weights: ClassWeights = field(default_factory=ClassWeights)
    cost_weights: CostWeights = field(default_factory=CostWeights)

    # Stage 3 - millimetre grid search window, in pixels of the analysed image.
    min_grid_period_px: int = 4
    max_grid_period_px: int = 60
    grid_peak_threshold: float = 0.07

    # Stage 4 - a printed sheet carries between three and fifteen lead rows.
    min_lead_rows: int = 3
    max_lead_rows: int = 15
    # The Bionet EKG2000 workflow is a simultaneous 12x1 print.  Keeping this
    # explicit lets the layout stage use a device prior without pretending that
    # an arbitrary three- or six-row print is also twelve rows.
    expected_lead_rows: int | None = 12
    band_margin_fraction: float = 0.55

    # Stage 5 - glyph-like blob acceptance.
    min_text_area_px: int = 24
    max_text_area_fraction: float = 0.02

    # Stage 10 - a gap longer than this is left as missing data instead of
    # being invented; 0.4 s is well under one beat at any plausible rate.
    max_recovered_gap_seconds: float = 0.40
    gap_evidence_threshold: float = 0.30

    # Stage 11 - refractory period and integration window of the R-peak search.
    refractory_seconds: float = 0.22
    integration_seconds: float = 0.12

    # Stage 12 - median beat window around each R peak.
    template_pre_seconds: float = 0.25
    template_post_seconds: float = 0.45

    # Stage 13 - half-width of the refinement search, in pixels.
    refine_radius_px: int = 3

    # Paper defaults, overridden by stage 5 when the legend is readable.
    paper_speed_mm_s: float = 25.0
    gain_mm_mv: float = 10.0

    # Cap on the width used by the path solver. Wider sheets are solved on a
    # decimated column grid and interpolated back, which keeps a 12-lead photo
    # inside a second or two without changing the reconstruction visibly.
    max_solver_width: int = 1600

    def with_paper(self, speed_mm_s: float, gain_mm_mv: float) -> PipelineConfig:
        from dataclasses import replace

        return replace(self, paper_speed_mm_s=speed_mm_s, gain_mm_mv=gain_mm_mv)


DEFAULT_CONFIG = PipelineConfig()
