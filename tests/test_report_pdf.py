from __future__ import annotations

import io
from datetime import UTC, datetime

from PIL import Image, ImageDraw
from pypdf import PdfReader

from miocard.models import ECGReport
from miocard.report_pdf import build_ecg_report_pdf, format_annotation_caption


def _sample_report() -> ECGReport:
    unavailable = {
        "value": "Не оценивается",
        "confidence": "low",
        "evidence": "Недостаточно данных",
    }
    return ECGReport.model_validate(
        {
            "image_quality": {
                "grade": "limited",
                "issues": ["Небольшой наклон снимка"],
                "visible_leads": ["I", "II", "III", "aVR", "aVL", "aVF"],
                "usable_leads": ["I", "II", "III"],
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
                    "value": "Синусовый ритм",
                    "confidence": "medium",
                    "evidence": "Перед комплексами QRS различимы зубцы P",
                },
                "heart_rate": {
                    "value": "Около 72 в минуту",
                    "confidence": "medium",
                    "evidence": "Оценка по расстояниям R-R",
                },
                "pr_interval": dict(unavailable),
                "qrs_complex": {
                    "value": "Без явного расширения",
                    "confidence": "medium",
                    "evidence": "Комплексы визуально узкие",
                },
                "electrical_axis": dict(unavailable),
                "q_waves": dict(unavailable),
                "r_progression": dict(unavailable),
                "st_t": dict(unavailable),
                "qt_qtc": dict(unavailable),
            },
            "possible_interpretations": [
                {
                    "label": "Неспецифические изменения реполяризации",
                    "confidence": "low",
                    "supporting_evidence": ["Форма T неоднородна"],
                    "contradicting_or_missing_evidence": ["Нет цифровых измерений"],
                }
            ],
            "urgent_findings": [],
            "limitations": ["Фотография не заменяет цифровой сигнал"],
            "recommended_checks": ["Сопоставить с симптомами и прошлой ЭКГ"],
            "overall_confidence": "medium",
            "concise_impression": "Синусовый ритм без очевидных острых изменений.",
            "localized_findings": [
                {
                    "finding": "Участок для контрольной оценки",
                    "related_observation": "st_t",
                    "box": {"top": 250, "left": 600, "bottom": 600, "right": 780},
                }
            ],
        }
    )


def _sample_ecg() -> bytes:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    for coordinate in range(0, 1600, 20):
        draw.line((coordinate, 0, coordinate, 900), fill=(240, 220, 220))
    for coordinate in range(0, 900, 20):
        draw.line((0, coordinate, 1600, coordinate), fill=(240, 220, 220))
    points = []
    for x in range(40, 1560):
        phase = x % 180
        y = 450
        if 70 <= phase < 76:
            y -= (phase - 70) * 38
        elif 76 <= phase < 82:
            y += (82 - phase) * 38
        points.append((x, y))
    draw.line(points, fill="black", width=3)
    draw.rectangle((960, 220, 1248, 540), outline=(220, 60, 40), width=7)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_pdf_report_contains_image_and_structured_sections() -> None:
    pdf = build_ecg_report_pdf(
        _sample_report(),
        _sample_ecg(),
        created_at=datetime(2026, 8, 27, 11, 45, tzinfo=UTC),
        user_context="50 лет, боль в груди",
        image_is_annotated=True,
    )

    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 20_000
    reader = PdfReader(io.BytesIO(pdf))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) >= 2
    assert "Предварительный разбор ЭКГ" in extracted
    assert "Контекст пользователя" in extracted
    assert "Модель" not in extracted
    assert "Sol" not in extracted
    assert "Синусовый ритм без очевидных острых изменений" in extracted
    assert "Параметры" in extracted
    assert "Ограничения" in extracted
    assert "Расшифровка отметок" in extracted
    assert "[1]" in extracted
    assert "Участок для контрольной оценки" in extracted
    assert "ST-T: Не оценивается" in extracted
    assert extracted.index("Расшифровка отметок") < extracted.index("Технические данные")


def test_telegram_annotation_caption_links_marker_to_observation() -> None:
    caption = format_annotation_caption(_sample_report())

    assert caption.startswith("Отметки на изображении:")
    assert "[1] Участок для контрольной оценки" in caption
    assert "ST-T: Не оценивается" in caption
    assert len(caption) <= 1024
