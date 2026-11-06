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
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ImageQuality(StrictModel):
    grade: Literal["diagnostic", "limited", "non_diagnostic"] = Field(
        alias="g", description="grade"
    )
    issues: list[str] = Field(alias="i", description="issues")
    visible_leads: list[str] = Field(alias="v", description="visible_leads")
    usable_leads: list[str] = Field(alias="u", description="usable_leads")
    unreadable_leads: list[str] = Field(alias="x", description="unreadable_leads")

    normalize_grade = field_validator("grade", mode="before")(_lower_enum)


class Calibration(StrictModel):
    raw_text: str = Field(alias="r", description="raw_text")
    speed_mm_s: float | None = Field(alias="s", description="speed_mm_s")
    gain_mm_mv: float | None = Field(alias="g", description="gain_mm_mv")
    gain_values_mm_mv: list[float] = Field(alias="gs", description="gain_values_mm_mv")
    gain_notes: str = Field(alias="n", description="gain_notes")
    confidence: Literal["high", "medium", "low"] = Field(
        alias="c", description="confidence"
    )
    source: str = Field(alias="src", description="source")

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class Observation(StrictModel):
    value: str = Field(alias="v", description="value")
    confidence: Literal["high", "medium", "low"] = Field(
        alias="c", description="confidence"
    )
    evidence: str = Field(alias="e", description="evidence")

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class ECGObservations(StrictModel):
    rhythm: Observation = Field(alias="r", description="rhythm")
    heart_rate: Observation = Field(alias="hr", description="heart_rate")
    pr_interval: Observation = Field(alias="pr", description="pr_interval")
    qrs_complex: Observation = Field(alias="qrs", description="qrs_complex")
    electrical_axis: Observation = Field(alias="ax", description="electrical_axis")
    q_waves: Observation = Field(alias="q", description="q_waves")
    r_progression: Observation = Field(alias="rp", description="r_progression")
    st_t: Observation = Field(alias="st", description="st_t")
    qt_qtc: Observation = Field(alias="qt", description="qt_qtc")


class Interpretation(StrictModel):
    label: str = Field(alias="l", description="label")
    confidence: Literal["high", "medium", "low"] = Field(
        alias="c", description="confidence"
    )
    supporting_evidence: list[str] = Field(alias="s", description="supporting_evidence")
    contradicting_or_missing_evidence: list[str] = Field(
        alias="x", description="contradicting_or_missing_evidence"
    )

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class UrgentFinding(StrictModel):
    finding: str = Field(alias="f", description="finding")
    confidence: Literal["high", "medium", "low"] = Field(
        alias="c", description="confidence"
    )
    rationale: str = Field(alias="r", description="rationale")
    recommended_action: str = Field(alias="a", description="recommended_action")

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class FindingBox(StrictModel):
    """Normalized coordinates relative to the exact image shown to the model."""

    top: int = Field(ge=0, le=1000, alias="t", description="top")
    left: int = Field(ge=0, le=1000, alias="l", description="left")
    bottom: int = Field(ge=0, le=1000, alias="b", description="bottom")
    right: int = Field(ge=0, le=1000, alias="r", description="right")

    @model_validator(mode="after")
    def validate_extent(self) -> FindingBox:
        if self.top >= self.bottom:
            raise ValueError("bottom must be greater than top")
        if self.left >= self.right:
            raise ValueError("right must be greater than left")
        return self


class LocalizedFinding(StrictModel):
    finding: str = Field(max_length=80, alias="f", description="finding")
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
    ] = Field(alias="o", description="related_observation")
    box: FindingBox = Field(alias="b", description="box")


class CandidateVerdict(StrictModel):
    """Ruling of the full-sheet pass on one candidate from the trusted context."""

    candidate: int = Field(
        ge=1, le=20, alias="n", description="number of the Cn candidate being judged"
    )
    verdict: Literal["confirmed", "rejected", "uncertain"] = Field(
        alias="v", description="verdict"
    )
    reason: str = Field(max_length=160, alias="r", description="short visual reason")

    normalize_verdict = field_validator("verdict", mode="before")(_lower_enum)


class ECGReport(StrictModel):
    image_quality: ImageQuality = Field(alias="q", description="image_quality")
    calibration: Calibration = Field(alias="cal", description="calibration")
    observations: ECGObservations = Field(alias="o", description="observations")
    possible_interpretations: list[Interpretation] = Field(
        alias="dx", description="possible_interpretations"
    )
    urgent_findings: list[UrgentFinding] = Field(alias="urg", description="urgent_findings")
    limitations: list[str] = Field(alias="lim", description="limitations")
    recommended_checks: list[str] = Field(alias="chk", description="recommended_checks")
    overall_confidence: Literal["high", "medium", "low"] = Field(
        alias="c", description="overall_confidence"
    )
    concise_impression: str = Field(
        alias="sum", description="concise_impression in Russian"
    )
    localized_findings: list[LocalizedFinding] = Field(
        default_factory=list,
        max_length=3,
        alias="m",
        description="localized_findings",
    )
    candidate_verdicts: list[CandidateVerdict] = Field(
        default_factory=list,
        max_length=12,
        alias="cv",
        description=(
            "one entry per Cn candidate of the trusted context; empty when no such "
            "context was supplied"
        ),
    )

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


class RhythmScoutCandidate(StrictModel):
    finding: str = Field(max_length=80, alias="f", description="finding")
    confidence: Literal["high", "medium", "low"] = Field(
        alias="c", description="confidence"
    )
    evidence: str = Field(max_length=240, alias="e", description="evidence")
    qrs_width: Literal["narrow", "wide", "uncertain"] = Field(
        alias="w", description="qrs_width"
    )
    kind: Literal["premature", "pause"] = Field(
        default="premature",
        alias="k",
        description="premature complex, or a pause/dropped beat",
    )
    box: FindingBox = Field(alias="b", description="box")

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)
    normalize_kind = field_validator("kind", mode="before")(_lower_enum)


class RhythmScoutResponse(StrictModel):
    candidates: list[RhythmScoutCandidate] = Field(
        default_factory=list,
        max_length=5,
        alias="c",
        description="candidates",
    )


RHYTHM_SCOUT_JSON_SCHEMA = compact_schema(RhythmScoutResponse.model_json_schema())


REPORT_JSON_SCHEMA = ECGReport.model_json_schema()


class InferredLead(StrictModel):
    lead: Literal["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"] = Field(
        alias="l", description="lead"
    )
    confidence: Literal["high", "medium"] = Field(alias="c", description="confidence")
    evidence: str = Field(alias="e", description="evidence")

    normalize_confidence = field_validator("confidence", mode="before")(_lower_enum)


class ECGImageFacts(StrictModel):
    is_ecg_image: bool = Field(alias="e", description="is_ecg_image")
    non_ecg_reason: str = Field(alias="n", description="non_ecg_reason")
    visible_lead_labels: list[str] = Field(alias="v", description="visible_lead_labels")
    inferred_leads: list[InferredLead] = Field(alias="i", description="inferred_leads")
    missing_standard_leads: list[str] = Field(alias="m", description="missing_standard_leads")
    readable_waveform_leads: list[str] = Field(
        alias="r", description="readable_waveform_leads"
    )
    unreadable_waveform_leads: list[str] = Field(
        alias="u", description="unreadable_waveform_leads"
    )
    calibration_text: str = Field(alias="ct", description="calibration_text")
    speed_mm_s: float | None = Field(alias="s", description="speed_mm_s")
    gain_values_mm_mv: list[float] = Field(alias="g", description="gain_values_mm_mv")


IMAGE_FACTS_JSON_SCHEMA = compact_schema(ECGImageFacts.model_json_schema())


class ECGAnalysisResponse(StrictModel):
    image_facts: ECGImageFacts = Field(alias="f", description="image_facts")
    report: ECGReport | None = Field(alias="r", description="report")


ANALYSIS_JSON_SCHEMA = compact_schema(ECGAnalysisResponse.model_json_schema())
