from miocard.formatting import format_report, format_report_messages, split_html_message
from miocard.models import (
    NON_ECG_IMPRESSION,
    ECGReport,
    Interpretation,
    LocalizedFinding,
    UrgentFinding,
)


def sample_report() -> ECGReport:
    empty_observation = {
        "value": "Не оценивается",
        "confidence": "low",
        "evidence": "Недостаточно данных",
    }
    return ECGReport.model_validate(
        {
            "image_quality": {
                "grade": "limited",
                "issues": ["Наклон снимка"],
                "visible_leads": ["I", "II", "aVL"],
                "usable_leads": ["I", "II"],
                "unreadable_leads": ["aVL"],
            },
            "calibration": {
                "raw_text": "25 mm/s 10 mm/mV",
                "speed_mm_s": 25,
                "gain_mm_mv": 10,
                "gain_values_mm_mv": [10],
                "gain_notes": "Одинаковое усиление",
                "confidence": "medium",
                "source": "Подпись внизу",
            },
            "observations": {
                "rhythm": {
                    "value": "Регулярный",
                    "confidence": "medium",
                    "evidence": "R–R примерно равны",
                },
                "heart_rate": dict(empty_observation),
                "pr_interval": dict(empty_observation),
                "qrs_complex": dict(empty_observation),
                "electrical_axis": dict(empty_observation),
                "q_waves": dict(empty_observation),
                "r_progression": dict(empty_observation),
                "st_t": dict(empty_observation),
                "qt_qtc": dict(empty_observation),
            },
            "possible_interpretations": [],
            "urgent_findings": [],
            "limitations": ["Фото не заменяет цифровой сигнал"],
            "recommended_checks": ["Измерить интервалы на оригинале"],
            "overall_confidence": "low",
            "concise_impression": "Нужна проверка врачом <очно>.",
        }
    )


def test_format_report_escapes_html() -> None:
    text = format_report(sample_report())
    assert "&lt;очно&gt;" in text
    assert "не диагноз" not in text
    assert "⚠️" not in text
    assert "🟡" in text
    assert "[средняя]" not in text
    assert len(text) <= 3900


def test_format_report_shows_dual_gain() -> None:
    report = sample_report()
    report.calibration.gain_mm_mv = None
    report.calibration.gain_values_mm_mv = [10, 5]
    report.calibration.raw_text = "10.0/5.0mm/mV 25.0mm/sec"
    text = format_report(report)
    assert "10/5 мм/мВ (двойное усиление)" in text
    assert "10.0/5.0mm/mV" in text


def test_non_ecg_report_stops_after_concise_message() -> None:
    report = sample_report()
    report.image_quality.grade = "non_diagnostic"
    report.concise_impression = NON_ECG_IMPRESSION

    text = format_report(report)

    assert text == (
        "⚕️ <b>Предварительный разбор ЭКГ</b>\n\n"
        f"<b>Кратко:</b> {NON_ECG_IMPRESSION}"
    )
    assert "Качество:" not in text
    assert "Параметры:" not in text
    assert "Предположения:" not in text
    assert "Ограничения:" not in text
    assert "Проверить:" not in text


def test_unavailable_parameters_are_not_repeated() -> None:
    text = format_report(sample_report())

    assert "<b>Ритм:</b>" in text
    assert "<b>ST–T:</b>" not in text
    assert "Вывод модели отклонен" not in text


def test_rhythm_interpretation_is_not_repeated_in_assumptions() -> None:
    report = sample_report()
    report.observations.rhythm.value = (
        "Синусовый ритм с возможными наджелудочковыми экстрасистолами"
    )
    report.possible_interpretations = [
        Interpretation.model_validate({
            "label": "Синусовый ритм с возможными наджелудочковыми экстрасистолами",
            "confidence": "low",
            "supporting_evidence": ["Узкие QRS"],
            "contradicting_or_missing_evidence": ["P перед ранним QRS не различим"],
        }),
        Interpretation.model_validate({
            "label": "Неспецифический вариант реполяризации",
            "confidence": "low",
            "supporting_evidence": ["Высокие положительные T"],
            "contradicting_or_missing_evidence": ["J-точка не измерена"],
        }),
    ]

    text = format_report(report)

    assert text.count("Синусовый ритм с возможными наджелудочковыми экстрасистолами") == 1
    assert "Неспецифический вариант реполяризации" in text


def test_internal_analysis_terms_are_replaced() -> None:
    report = sample_report()
    report.observations.rhythm.evidence = "Проверено через visual preflight и LOCKED_IMAGE_FACTS"

    text = format_report(report)

    assert "visual preflight" not in text
    assert "LOCKED_IMAGE_FACTS" not in text
    assert "проверка изображения" in text


def test_localized_findings_add_matching_footnotes() -> None:
    report = sample_report()
    report.localized_findings = [
        LocalizedFinding.model_validate({
            "finding": "Одиночный широкий комплекс",
            "related_observation": "rhythm",
            "box": {"top": 100, "left": 700, "bottom": 500, "right": 850},
        }),
        LocalizedFinding.model_validate({
            "finding": "Изменение ST–T",
            "related_observation": "st_t",
            "box": {"top": 600, "left": 200, "bottom": 750, "right": 450},
        }),
    ]

    text = format_report(report)

    assert "<b>Ритм:</b> [1]" in text
    assert "<b>[1] Одиночный широкий комплекс</b>" in text
    assert "<b>[2] Изменение ST–T</b>" in text
    assert "<b>Отметки на снимке:</b>" in text


def test_minor_quality_issues_are_hidden_for_diagnostic_image() -> None:
    report = sample_report()
    report.image_quality.grade = "diagnostic"
    report.image_quality.issues = ["Небольшой наклон"]
    report.limitations = []

    text = format_report(report)

    assert "Небольшой наклон" not in text


def test_lead_message_carries_the_impression_and_urgent_findings() -> None:
    report = sample_report()
    report.urgent_findings = [
        UrgentFinding.model_validate({
            "finding": "Подъем ST в II, III, aVF",
            "confidence": "medium",
            "rationale": "Элевация более 1 мм",
            "recommended_action": "Срочная очная оценка",
        })
    ]

    messages = format_report_messages(report)

    lead = messages[0]
    assert "Кратко:" in lead
    assert "Подъем ST в II, III, aVF" in lead
    # The lead has to fit a notification preview, so the supporting detail moves on.
    assert len(lead) < 700
    assert "Параметры:" not in lead
    assert "Ограничения:" not in lead
    assert "Параметры:" in "\n".join(messages[1:])


def test_absence_of_urgent_findings_is_stated_rather_than_left_blank() -> None:
    messages = format_report_messages(sample_report())

    assert "не выявлено" in messages[0]


def test_non_ecg_report_is_a_single_message() -> None:
    report = sample_report()
    report.concise_impression = NON_ECG_IMPRESSION

    messages = format_report_messages(report)

    assert len(messages) == 1
    assert NON_ECG_IMPRESSION in messages[0]


def test_long_expandable_section_is_split_without_breaking_its_tags() -> None:
    report = sample_report()
    for field_name in report.observations.__class__.model_fields:
        observation = getattr(report.observations, field_name)
        observation.value = "Очень подробное значение параметра " * 20
        observation.evidence = "Подробное визуальное основание вывода " * 20

    messages = format_report_messages(report)

    assert len(messages) > 2
    for message in messages:
        assert len(message) <= 3900
        assert message.count("<blockquote expandable>") == message.count("</blockquote>")
        assert message.count("<b>") == message.count("</b>")


def test_split_message_respects_limit() -> None:
    chunks = split_html_message(("раздел\n\n" * 1000), limit=200)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_split_message_does_not_break_html_or_entities() -> None:
    chunks = split_html_message("<b>" + ("<опасно>&" * 100) + "</b>", limit=120)
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all("<b>" not in chunk and "</b>" not in chunk for chunk in chunks)
    assert all(chunk.count("&") == chunk.count(";") for chunk in chunks)


def test_long_report_is_split_without_truncating_fields() -> None:
    report = sample_report()
    markers: list[str] = []
    for field_name in report.observations.__class__.model_fields:
        marker = f"КОНЕЦ-{field_name}"
        markers.append(marker)
        observation = getattr(report.observations, field_name)
        observation.value = ("Полное значение параметра " * 8) + marker
        observation.evidence = ("Полное визуальное основание " * 8) + marker
    report.concise_impression = ("Полное краткое описание без сокращения " * 12) + "КОНЕЦ-КРАТКО"
    markers.append("КОНЕЦ-КРАТКО")

    formatted = format_report(report)
    chunks = split_html_message(formatted, limit=3900)

    assert len(chunks) > 1
    assert all(len(chunk) <= 3900 for chunk in chunks)
    assert "…" not in formatted
    assert all(chunk.count("<b>") == chunk.count("</b>") for chunk in chunks)
    combined = "\n".join(chunks)
    assert all(marker in combined for marker in markers)
