from miocard.models import NON_ECG_IMPRESSION
from miocard.prompt import (
    NON_ECG_PLACEHOLDER,
    SPECIFICATION_FILE,
    SYSTEM_PROMPT,
    build_user_prompt,
    load_system_prompt,
)


def test_prompt_is_loaded_from_the_specification_file() -> None:
    assert SPECIFICATION_FILE.name == "analyze.md"
    assert load_system_prompt() == SYSTEM_PROMPT
    assert NON_ECG_PLACEHOLDER not in SYSTEM_PROMPT
    assert NON_ECG_IMPRESSION in SYSTEM_PROMPT


def test_prompt_requires_a_russian_answer_in_one_json_object() -> None:
    assert "must be written" in SYSTEM_PROMPT
    assert "in Russian" in SYSTEM_PROMPT
    assert "Return exactly one JSON object" in SYSTEM_PROMPT
    assert "No text, Markdown, code fences or comments before or after it." in SYSTEM_PROMPT
    assert "Do not add fields that are absent from the schema" in SYSTEM_PROMPT
    assert "Весь пользовательский текст в report пиши по-русски" in build_user_prompt("meta", None)


def test_prompt_locks_image_facts_and_handles_non_ecg_images() -> None:
    assert "treat it as locked" in SYSTEM_PROMPT
    assert "set `report=null`" in SYSTEM_PROMPT
    assert "must never be called missing" in SYSTEM_PROMPT
    assert "must become `usable_leads` unchanged" in SYSTEM_PROMPT


def test_prompt_requires_rhythm_differential() -> None:
    assert "always compare sinus arrhythmia against premature complexes" in SYSTEM_PROMPT
    assert "If the P wave before the early complex is not distinguishable" in SYSTEM_PROMPT
    assert "keep sinus arrhythmia as an\n  alternative" in SYSTEM_PROMPT
    assert "confidence in\nsupraventricular ectopy must be low" in SYSTEM_PROMPT
    assert "medium is forbidden" in SYSTEM_PROMPT


def test_prompt_scans_entire_recording_for_isolated_abnormal_complexes() -> None:
    assert "scan the entire visible recording horizontally from start" in SYSTEM_PROMPT
    assert "compare every QRS with its neighbours and with the dominant template" in SYSTEM_PROMPT
    assert "horizontal position must not\n  affect attention" in SYSTEM_PROMPT
    assert "in at least two readable leads" in SYSTEM_PROMPT
    assert "do not require a visible compensatory pause" in SYSTEM_PROMPT
    assert "must be reflected in `concise_impression` and" in SYSTEM_PROMPT
    assert "`observations.rhythm`, even if the remaining complexes are regular" in SYSTEM_PROMPT
    assert "do not turn local noise in one row into an ectopic\n  beat" in SYSTEM_PROMPT


def test_prompt_requests_normalized_localized_findings() -> None:
    assert "up to 3 of the most significant visible abnormalities" in SYSTEM_PROMPT
    assert "coordinates are integers 0–1000" in SYSTEM_PROMPT
    assert "0<=left<right<=1000" in SYSTEM_PROMPT
    assert "must tightly enclose the visible feature" in SYSTEM_PROMPT
    assert "Do not mark normal rhythm" in SYSTEM_PROMPT


def test_prompt_uses_strict_early_repolarization_criteria() -> None:
    assert "visible terminal notch or slurring of the QRS" in SYSTEM_PROMPT
    assert "concave ST or tall positive T waves alone" in SYSTEM_PROMPT
    assert "возможный доброкачественный или\n  неспецифический вариант реполяризации" in SYSTEM_PROMPT


def test_prompt_excludes_common_causes_before_rare_ones() -> None:
    assert "before atrial fibrillation exclude sinus arrhythmia" in SYSTEM_PROMPT
    assert "before\nan acute ischemic pattern exclude early repolarization" in SYSTEM_PROMPT
    assert "before Brugada or another rare pattern exclude a normal variant" in SYSTEM_PROMPT
    assert "not in a separate section" in SYSTEM_PROMPT


def test_prompt_uses_reported_history_only_as_context() -> None:
    assert "previous examinations and the conclusions of earlier ECGs" in SYSTEM_PROMPT
    assert "never as measured data from the current image" in SYSTEM_PROMPT
    assert "new, unchanged or resolved" in SYSTEM_PROMPT
    assert "trust the image and note" in SYSTEM_PROMPT


def test_prompt_rejects_false_image_quality_claims() -> None:
    assert "do not prove poor electrode contact" in SYSTEM_PROMPT
    assert "Line\n  thickness alone is not clipping" in SYSTEM_PROMPT
    assert "unique in meaning: do\n  not paraphrase the same defect twice" in SYSTEM_PROMPT
    assert "Do not mention hypertrophy or a hypertrophic pattern" in SYSTEM_PROMPT
    assert "Высокий вольтаж грудных комплексов" in SYSTEM_PROMPT


def test_prompt_cross_checks_heart_rate_two_ways() -> None:
    assert "cross-check the heart rate two independent ways" in SYSTEM_PROMPT
    assert "number of QRS complexes on a full row" in SYSTEM_PROMPT
    assert "differ by more\n  than 10/min" in SYSTEM_PROMPT


def test_prompt_forbids_repeated_rhythm_interpretation() -> None:
    assert "justify it in\ndetail only in `observations.rhythm`" in SYSTEM_PROMPT
    assert (
        "do not create a separate `possible_interpretations` entry\nfor that same rhythm"
        in SYSTEM_PROMPT
    )


def test_prompt_avoids_repeated_generic_warnings() -> None:
    assert "the general disclaimer is\n  already shown at bot start" in SYSTEM_PROMPT
    assert "is not a standalone image limitation" in SYSTEM_PROMPT
    assert 'mandatory "клиническая корреляция"' in SYSTEM_PROMPT
