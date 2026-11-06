from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NON_ECG_IMPRESSION = (
    "Изображение не является диагностической 12-канальной ЭКГ. "
    "Клиническая интерпретация ритма, интервалов и морфологии невозможна."
)


def _lower_enum(value: object) -> object:
    return value.lower() if isinstance(value, str) else value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageQuality(StrictModel):
    grade: Literal["diagnostic", "limited", "non_diagnostic"]
    issues: list[str]
    visible_leads: list[str]
    usable_leads: list[str]
    unreadable_leads: list[str]

    normalize_grade = field_validator("grade", mode="before")(_lower_enum)


class Calibration(StrictModel):
    raw_text: str
    speed_mm_s: float | None
    gain_mm_mv: float | None
    gain_values_mm_mv: list[float]
    gain_notes: str
    confidence: Literal["high", "medium", "low"]
    source: str

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class Observation(StrictModel):
    value: str
    confidence: Literal["high", "medium", "low"]
    evidence: str

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class ECGObservations(StrictModel):
    rhythm: Observation
    heart_rate: Observation
    pr_interval: Observation
    qrs_complex: Observation
    electrical_axis: Observation
    q_waves: Observation
    r_progression: Observation
    st_t: Observation
    qt_qtc: Observation


class Interpretation(StrictModel):
    label: str
    confidence: Literal["high", "medium", "low"]
    supporting_evidence: list[str]
    contradicting_or_missing_evidence: list[str]

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class UrgentFinding(StrictModel):
    finding: str
    confidence: Literal["high", "medium", "low"]
    rationale: str
    recommended_action: str

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class FindingBox(StrictModel):
    """Normalized coordinates relative to the exact image shown to the model."""

    top: int = Field(ge=0, le=1000)
    left: int = Field(ge=0, le=1000)
    bottom: int = Field(ge=0, le=1000)
    right: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def validate_extent(self) -> FindingBox:
        if self.top >= self.bottom:
            raise ValueError("bottom must be greater than top")
        if self.left >= self.right:
            raise ValueError("right must be greater than left")
        return self


class LocalizedFinding(StrictModel):
    finding: str = Field(max_length=80)
    related_observation: Literal[
        "rhythm",
        "heart_rate",
        "pr_interval",
        "qrs_complex",
        "electrical_axis",
        "q_waves",
        "r_progression",
        "st_t",
        "qt_qtc",
        "other",
    ]
    box: FindingBox


class ECGReport(StrictModel):
    image_quality: ImageQuality
    calibration: Calibration
    observations: ECGObservations
    possible_interpretations: list[Interpretation]
    urgent_findings: list[UrgentFinding]
    limitations: list[str]
    recommended_checks: list[str]
    overall_confidence: Literal["high", "medium", "low"]
    concise_impression: str = Field(description="Short preliminary impression in Russian")
    localized_findings: list[LocalizedFinding] = Field(default_factory=list, max_length=3)

    normalize_confidence = field_validator("overall_confidence", mode="before")(_lower_enum)


def compact_schema(schema: object) -> object:
    """Drop Pydantic's generated `title` keys before the schema is sent to a model.

    Pydantic titles every model and every field, which is roughly a quarter of the
    serialized schema and tells the model nothing the key name does not already say.
    A property genuinely called `title` maps to an object, not a string, so checking the
    value type keeps real fields intact.
    """
    if isinstance(schema, dict):
        return {
            key: compact_schema(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [compact_schema(item) for item in schema]
    return schema


REPORT_JSON_SCHEMA = ECGReport.model_json_schema()


class InferredLead(StrictModel):
    lead: Literal["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    confidence: Literal["high", "medium"]
    evidence: str

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class ECGImageFacts(StrictModel):
    is_ecg_image: bool
    non_ecg_reason: str
    visible_lead_labels: list[str]
    inferred_leads: list[InferredLead]
    missing_standard_leads: list[str]
    readable_waveform_leads: list[str]
    unreadable_waveform_leads: list[str]
    calibration_text: str
    speed_mm_s: float | None
    gain_values_mm_mv: list[float]


IMAGE_FACTS_JSON_SCHEMA = compact_schema(ECGImageFacts.model_json_schema())


class ECGAnalysisResponse(StrictModel):
    image_facts: ECGImageFacts
    report: ECGReport | None


ANALYSIS_JSON_SCHEMA = compact_schema(ECGAnalysisResponse.model_json_schema())
