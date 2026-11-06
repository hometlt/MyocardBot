from __future__ import annotations

import io
import os
import re
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from miocard.models import ECGReport, Interpretation, Observation

PAGE_WIDTH, PAGE_HEIGHT = A4
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2D6B8A")
PALE_BLUE = colors.HexColor("#EAF3F7")
PALE_RED = colors.HexColor("#FDECEC")
PALE_GREEN = colors.HexColor("#EAF6EF")
TEXT = colors.HexColor("#243442")
MUTED = colors.HexColor("#62717D")
LINE = colors.HexColor("#D7E0E5")
CONFIDENCE = {"high": "высокая", "medium": "средняя", "low": "низкая"}
QUALITY = {
    "diagnostic": "приемлемое",
    "limited": "ограниченное",
    "non_diagnostic": "непригодное для надежного разбора",
}
OBSERVATION_FIELDS = (
    ("Ритм", "rhythm"),
    ("ЧСС", "heart_rate"),
    ("PQ/PR", "pr_interval"),
    ("QRS", "qrs_complex"),
    ("ЭОС", "electrical_axis"),
    ("Q", "q_waves"),
    ("Прогрессия R", "r_progression"),
    ("ST-T", "st_t"),
    ("QT/QTc", "qt_qtc"),
)
OBSERVATION_LABEL_BY_FIELD = {field_name: label for label, field_name in OBSERVATION_FIELDS}
RHYTHM_CONCEPT_MARKERS = (
    "синусов",
    "экстрасистол",
    "аритми",
    "фибрилляц",
    "трепет",
    "тахикард",
    "брадикард",
    "эктоп",
)


def _font_candidates(filename: str) -> tuple[Path, ...]:
    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    dejavu_filename = (
        "DejaVuSans-Bold.ttf" if filename == "arialbd.ttf" else "DejaVuSans.ttf"
    )
    return (
        windows_dir / "Fonts" / filename,
        Path("/usr/share/fonts/truetype/dejavu") / dejavu_filename,
    )


@lru_cache(maxsize=1)
def _register_fonts() -> tuple[str, str]:
    regular_candidates = _font_candidates("arial.ttf")
    bold_candidates = _font_candidates("arialbd.ttf")
    regular = next((path for path in regular_candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), None)
    if regular is None or bold is None:
        raise RuntimeError("Не найден Unicode-шрифт для формирования PDF-отчета.")
    pdfmetrics.registerFont(TTFont("MiocardSans", str(regular)))
    pdfmetrics.registerFont(TTFont("MiocardSans-Bold", str(bold)))
    return "MiocardSans", "MiocardSans-Bold"


def _clean(value: str) -> str:
    text = re.sub(
        r"\bvisual[-\s]+(?:preflight|проход\w*)\b",
        "проверка изображения",
        value,
        flags=re.IGNORECASE,
    )
    text = text.replace("LOCKED_IMAGE_FACTS", "данные проверки изображения")
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return " ".join(text.split())


def _safe(value: str) -> str:
    return escape(_clean(value), quote=False)


def _clip(value: str, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def localized_finding_legend(report: ECGReport) -> list[tuple[int, str, str]]:
    """Map every image marker to its finding and related observation value."""
    entries: list[tuple[int, str, str]] = []
    for marker, finding in enumerate(report.localized_findings, start=1):
        field_name = finding.related_observation
        if field_name == "other":
            observation = "Другое наблюдение"
        else:
            label = OBSERVATION_LABEL_BY_FIELD.get(field_name, field_name)
            item: Observation = getattr(report.observations, field_name)
            observation = f"{label}: {item.value}"
        entries.append((marker, _clean(finding.finding), _clean(observation)))
    return entries


def format_annotation_caption(report: ECGReport, limit: int = 1024) -> str:
    lines = ["Отметки на изображении:"]
    for marker, finding, observation in localized_finding_legend(report):
        lines.append(f"[{marker}] {finding}\n{_clip(observation, 180)}")
    caption = "\n\n".join(lines)
    if len(caption) <= limit:
        return caption
    return caption[: limit - 1].rstrip() + "…"


def _timestamp(value: datetime) -> str:
    local_time = value.astimezone()
    offset = local_time.strftime("%z")
    zone = f"UTC{offset[:3]}:{offset[3:]}" if len(offset) == 5 else "UTC"
    return f"{local_time:%d.%m.%Y %H:%M:%S} {zone}"


def _calibration_text(report: ECGReport) -> str:
    calibration = report.calibration
    speed = (
        f"{calibration.speed_mm_s:g} мм/с"
        if calibration.speed_mm_s is not None
        else "скорость не читается"
    )
    if calibration.gain_mm_mv is not None:
        gain = f"{calibration.gain_mm_mv:g} мм/мВ"
    elif calibration.gain_values_mm_mv:
        values = "/".join(f"{value:g}" for value in calibration.gain_values_mm_mv)
        gain = f"{values} мм/мВ"
    else:
        gain = "усиление не читается"
    return f"{speed}, {gain}; уверенность - {CONFIDENCE[calibration.confidence]}"


def _is_duplicate_rhythm_interpretation(item: Interpretation, rhythm: str) -> bool:
    label_lower = item.label.lower()
    rhythm_lower = rhythm.lower()
    return any(
        marker in label_lower and marker in rhythm_lower
        for marker in RHYTHM_CONCEPT_MARKERS
    )


def _footer(canvas, document, regular_font: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, PAGE_WIDTH - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(regular_font, 7.5)
    canvas.drawString(
        18 * mm,
        8.5 * mm,
        "Миокард - предварительный автоматизированный разбор, не окончательный диагноз",
    )
    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        8.5 * mm,
        f"Страница {document.page}",
    )
    canvas.restoreState()


def _list_story(
    heading: str,
    items: list[str],
    *,
    heading_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> list[object]:
    story: list[object] = [Paragraph(heading, heading_style)]
    if not items:
        story.append(Paragraph("- нет", body_style))
    else:
        story.extend(Paragraph("- " + _safe(item), body_style) for item in items)
    story.append(Spacer(1, 3 * mm))
    return story


def build_ecg_report_pdf(
    report: ECGReport,
    ecg_image: bytes | None,
    *,
    created_at: datetime,
    user_context: str | None,
    image_is_annotated: bool,
) -> bytes:
    """Create a polished, self-contained PDF from the structured ECG report."""
    regular_font, bold_font = _register_fonts()
    stylesheet = getSampleStyleSheet()
    title = ParagraphStyle(
        "MiocardTitle",
        parent=stylesheet["Title"],
        fontName=bold_font,
        fontSize=19,
        leading=23,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=4 * mm,
    )
    subtitle = ParagraphStyle(
        "MiocardSubtitle",
        parent=stylesheet["Normal"],
        fontName=regular_font,
        fontSize=9,
        leading=12,
        textColor=MUTED,
        spaceAfter=4 * mm,
    )
    heading = ParagraphStyle(
        "MiocardHeading",
        parent=stylesheet["Heading2"],
        fontName=bold_font,
        fontSize=12,
        leading=15,
        textColor=NAVY,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "MiocardBody",
        parent=stylesheet["BodyText"],
        fontName=regular_font,
        fontSize=9.2,
        leading=13,
        textColor=TEXT,
        alignment=TA_LEFT,
        spaceAfter=1.5 * mm,
    )
    small = ParagraphStyle(
        "MiocardSmall",
        parent=body,
        fontSize=8,
        leading=10.5,
        textColor=MUTED,
    )
    callout = ParagraphStyle(
        "MiocardCallout",
        parent=body,
        fontName=bold_font,
        fontSize=11,
        leading=15,
        textColor=NAVY,
    )
    table_text = ParagraphStyle(
        "MiocardTable",
        parent=body,
        fontSize=7.8,
        leading=10,
        spaceAfter=0,
    )
    table_header = ParagraphStyle(
        "MiocardTableHeader",
        parent=table_text,
        fontName=bold_font,
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="Предварительный разбор ЭКГ",
        author="Миокард",
    )
    story: list[object] = [
        Paragraph("Предварительный разбор ЭКГ", title),
        Paragraph(
            "Система поддержки врача по изображению. Документ не заменяет очную оценку и измерения по цифровому сигналу.",
            subtitle,
        ),
    ]

    metadata = Table(
        [
            [Paragraph("Дата и время", table_text), Paragraph(_timestamp(created_at), table_text)],
            [
                Paragraph("Качество", table_text),
                Paragraph(
                    f"{QUALITY[report.image_quality.grade]}; общая уверенность - {CONFIDENCE[report.overall_confidence]}",
                    table_text,
                ),
            ],
            [Paragraph("Калибровка", table_text), Paragraph(_calibration_text(report), table_text)],
        ],
        colWidths=(34 * mm, document.width - 34 * mm),
        hAlign="LEFT",
    )
    metadata.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
                ("FONTNAME", (0, 0), (0, -1), bold_font),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metadata, Spacer(1, 4 * mm)])

    if user_context and user_context.strip():
        context_table = Table(
            [[Paragraph("Контекст пользователя", heading)], [Paragraph(_safe(user_context), body)]],
            colWidths=(document.width,),
        )
        context_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FA")),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([context_table, Spacer(1, 4 * mm)])

    summary_table = Table(
        [[Paragraph("Краткий итог", heading)], [Paragraph(_safe(report.concise_impression), callout)]],
        colWidths=(document.width,),
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 4 * mm)])

    urgent_items = [
        f"{item.finding} ({CONFIDENCE[item.confidence]}): {item.rationale}. {item.recommended_action}"
        for item in report.urgent_findings
    ]
    urgent_body = urgent_items or ["Потенциально неотложных признаков на снимке не выявлено."]
    urgent_table = Table(
        [
            [Paragraph("Срочные признаки", heading)],
            *[[Paragraph("- " + _safe(item), body)] for item in urgent_body],
        ],
        colWidths=(document.width,),
    )
    urgent_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    PALE_RED if report.urgent_findings else PALE_GREEN,
                ),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([urgent_table, PageBreak()])

    if ecg_image:
        image = Image(io.BytesIO(ecg_image))
        max_width = document.width
        max_height = 150 * mm
        scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        image_block: list[object] = [
            Paragraph("Изображение ЭКГ", heading),
            image,
            Spacer(1, 2 * mm),
        ]
        legend = localized_finding_legend(report) if image_is_annotated else []
        if legend:
            legend_rows: list[list[Paragraph]] = [
                [
                    Paragraph("Метка", table_header),
                    Paragraph("Находка", table_header),
                    Paragraph("Связанное наблюдение", table_header),
                ]
            ]
            legend_rows.extend(
                [
                    Paragraph(f"[{marker}]", table_text),
                    Paragraph(_safe(finding), table_text),
                    Paragraph(_safe(_clip(observation, 320)), table_text),
                ]
                for marker, finding, observation in legend
            )
            legend_table = Table(
                legend_rows,
                colWidths=(14 * mm, 58 * mm, document.width - 72 * mm),
                hAlign="LEFT",
            )
            legend_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            image_block.extend(
                [
                    Paragraph("Расшифровка отметок", heading),
                    legend_table,
                ]
            )
        else:
            image_block.append(
                Paragraph(
                    "Обработанная ЭКГ. Автоматическая разметка на изображении отсутствует.",
                    small,
                )
            )
        image_block.append(Spacer(1, 4 * mm))
        story.append(KeepTogether(image_block))
        story.append(PageBreak())

    visible_leads = ", ".join(report.image_quality.visible_leads) or "не определены"
    story.extend(
        [
            Paragraph("Технические данные", heading),
            Paragraph(f"<b>Видимые отведения:</b> {_safe(visible_leads)}", body),
            Paragraph(
                f"<b>Исходная строка калибровки:</b> {_safe(report.calibration.raw_text or 'не читается')}",
                body,
            ),
            Spacer(1, 2 * mm),
        ]
    )

    finding_markers: dict[str, list[int]] = {}
    for marker, finding in enumerate(report.localized_findings, start=1):
        finding_markers.setdefault(finding.related_observation, []).append(marker)

    observation_rows: list[list[Paragraph]] = [
        [
            Paragraph("Параметр", table_header),
            Paragraph("Наблюдение", table_header),
            Paragraph("Уверенность", table_header),
            Paragraph("Основание", table_header),
        ]
    ]
    for label, field_name in OBSERVATION_FIELDS:
        item: Observation = getattr(report.observations, field_name)
        marker_text = "".join(f" [{marker}]" for marker in finding_markers.get(field_name, []))
        observation_rows.append(
            [
                Paragraph(_safe(label + marker_text), table_text),
                Paragraph(_safe(item.value), table_text),
                Paragraph(CONFIDENCE[item.confidence], table_text),
                Paragraph(_safe(item.evidence), table_text),
            ]
        )
    observations = LongTable(
        observation_rows,
        colWidths=(24 * mm, 50 * mm, 24 * mm, document.width - 98 * mm),
        repeatRows=1,
        hAlign="LEFT",
    )
    observations.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([Paragraph("Параметры", heading), observations, PageBreak()])

    interpretations = [
        item
        for item in report.possible_interpretations
        if not _is_duplicate_rhythm_interpretation(item, report.observations.rhythm.value)
    ]
    story.append(Paragraph("Возможные интерпретации", heading))
    if not interpretations:
        story.append(Paragraph("- нет", body))
    for item in interpretations:
        supporting = "; ".join(item.supporting_evidence) or "нет"
        missing = "; ".join(item.contradicting_or_missing_evidence) or "нет"
        story.append(
            KeepTogether(
                [
                    Paragraph(
                        f"<b>{_safe(item.label)}</b> - уверенность {CONFIDENCE[item.confidence]}",
                        body,
                    ),
                    Paragraph(f"За: {_safe(supporting)}", small),
                    Paragraph(f"Ограничения: {_safe(missing)}", small),
                    Spacer(1, 2 * mm),
                ]
            )
        )

    limitations = list(
        dict.fromkeys(
            (
                report.image_quality.issues
                if report.image_quality.grade != "diagnostic"
                else []
            )
            + report.limitations
        )
    )
    story.extend(
        _list_story(
            "Ограничения",
            limitations,
            heading_style=heading,
            body_style=body,
        )
    )
    story.extend(
        _list_story(
            "Рекомендуется проверить",
            report.recommended_checks,
            heading_style=heading,
            body_style=body,
        )
    )

    document.build(
        story,
        onFirstPage=lambda canvas, doc: _footer(canvas, doc, regular_font),
        onLaterPages=lambda canvas, doc: _footer(canvas, doc, regular_font),
    )
    return output.getvalue()
