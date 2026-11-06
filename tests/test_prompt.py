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
    assert "For an irregular rhythm compare sinus arrhythmia, premature complexes" in SYSTEM_PROMPT
    assert "A premature P may be\n  hidden in the preceding T wave" in SYSTEM_PROMPT
    assert "preserve the abnormal event count and lower only" in SYSTEM_PROMPT
    assert "confidence in supraventricular ectopy must be low" in SYSTEM_PROMPT
    assert "medium is forbidden" in SYSTEM_PROMPT


def test_prompt_scans_entire_recording_for_isolated_abnormal_complexes() -> None:
    assert "left 0–40%, middle 30–75%,\n   and right 65–100%" in SYSTEM_PROMPT
    assert "identify the QRS sequence from left to right" in SYSTEM_PROMPT
    assert "Compare every\n   beat with both neighbours" in SYSTEM_PROMPT
    assert "at least two leads supports a true event" in SYSTEM_PROMPT
    assert "A compensatory pause supports it but is not required" in SYSTEM_PROMPT
    assert "Run three independent searches" in SYSTEM_PROMPT
    assert "compare the last four visible R–R intervals" in SYSTEM_PROMPT
    assert "Perform a second pass from right to left" in SYSTEM_PROMPT
    assert "Count distinct temporal events" in SYSTEM_PROMPT
    assert "Reflect a clinically meaningful event count in `concise_impression`" in SYSTEM_PROMPT
    assert "trusted multiscale rhythm context" in SYSTEM_PROMPT
    assert "independent analyses of\noverlapping" in SYSTEM_PROMPT
    assert "coordinates relative to that view" in SYSTEM_PROMPT


def test_prompt_returns_a_verdict_for_every_trusted_candidate() -> None:
    assert "return exactly one `candidate_verdicts` entry per listed candidate" in SYSTEM_PROMPT
    assert "unverified rather than as rejected" in SYSTEM_PROMPT
    assert "only a confirmed one may change the count you report" in SYSTEM_PROMPT


def test_prompt_scans_for_pauses_and_dropped_beats() -> None:
    assert "then for abruptly long intervals" in SYSTEM_PROMPT
    assert "An abruptly long interval is a conduction event, not ectopy" in SYSTEM_PROMPT
    assert "atrioventricular block or a blocked atrial complex" in SYSTEM_PROMPT
    assert "never merge the two numbers" in SYSTEM_PROMPT
    assert "never\n  report it as an extrasystole" in SYSTEM_PROMPT


def test_prompt_sweeps_st_territories_and_must_not_miss_patterns() -> None:
    assert "### Territorial ST–T sweep" in SYSTEM_PROMPT
    assert "inferior (II, III, aVF), lateral" in SYSTEM_PROMPT
    assert "check the remaining groups for a\nreciprocal change" in SYSTEM_PROMPT
    assert "must not be dismissed as an isolated aVR artifact" in SYSTEM_PROMPT
    assert "### Conduction and must-not-miss sweep" in SYSTEM_PROMPT
    assert "a PR interval under 120 ms with a slurred QRS upstroke" in SYSTEM_PROMPT
    assert "under 5 mm in every limb lead" in SYSTEM_PROMPT
    assert "alternating beat to beat" in SYSTEM_PROMPT
    # A silent negative must not turn into a list of everything that was ruled out.
    assert "mention it only where something is actually visible" in SYSTEM_PROMPT


def test_user_prompt_separates_trusted_multiscale_context_from_user_text() -> None:
    prompt = build_user_prompt("meta", "история пациента", "C1 x≈215/1000")

    assert "Краткий клинический контекст от пользователя:\nистория пациента" in prompt
    assert "Доверенный контекст приложения для текущего анализа:\nC1 x≈215/1000" in prompt
    assert "не является диагнозом или готовым заключением" in prompt


def test_prompt_compares_atrial_fibrillation_and_flutter_symmetrically() -> None:
    assert "Assess atrial fibrillation and atrial flutter symmetrically" in SYSTEM_PROMPT
    assert "including exactly 149 or 150/min" in SYSTEM_PROMPT
    assert "carries no diagnostic weight by itself" in SYSTEM_PROMPT
    assert "non-repetitive irregular R–R sequence" in SYSTEM_PROMPT
    assert "Atrial flutter requires organized atrial deflections" in SYSTEM_PROMPT
    assert "Never treat the millimetre grid, compression ripples" in SYSTEM_PROMPT
    assert "do not call flutter probable" in SYSTEM_PROMPT
    assert "QRS amplitude variation is not a discriminator" in SYSTEM_PROMPT
    assert "different printed gain as evidence of atrial fibrillation" in SYSTEM_PROMPT
    assert "A sinus label requires one discrete sinus-compatible P wave" in SYSTEM_PROMPT
    assert "Do not join intervals from\n  different sequential lead segments" in SYSTEM_PROMPT
    assert "explicitly test atrial flutter with 2:1" not in SYSTEM_PROMPT


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
    assert "Apply equal evidentiary thresholds to atrial fibrillation" in SYSTEM_PROMPT
    assert "before atrial flutter exclude grid/noise" in SYSTEM_PROMPT
    assert "Never reject atrial fibrillation merely because" in SYSTEM_PROMPT
    assert "Before an acute ischemic pattern exclude early repolarization" in SYSTEM_PROMPT
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
    assert "Do not mention ventricular hypertrophy or a hypertrophic pattern" in SYSTEM_PROMPT
    assert "Высокий вольтаж грудных комплексов" in SYSTEM_PROMPT


def test_prompt_uses_measured_atrial_and_ventricular_voltage_criteria() -> None:
    assert "Call **P mitrale** only" in SYSTEM_PROMPT
    assert "total P duration >=120 ms" in SYSTEM_PROMPT
    assert "Call **P pulmonale** only" in SYSTEM_PROMPT
    assert "greater than 2.5 mm in lead II" in SYSTEM_PROMPT
    assert "Sokolow-Lyon S(V1) + the taller" in SYSTEM_PROMPT
    assert "Cornell S(V3) + R(aVL) >28 mm" in SYSTEM_PROMPT
    assert "R/S in V1 >1 with R(V1) >6 mm" in SYSTEM_PROMPT
    assert "R(V1) + S(V5 or V6)\n  >10.5 mm" in SYSTEM_PROMPT
    assert "Relative differences in wave amplitude may be described" in SYSTEM_PROMPT


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
