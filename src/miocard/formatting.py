from __future__ import annotations

import re
from html import escape, unescape

from miocard.models import NON_ECG_IMPRESSION, ECGReport

CONFIDENCE = {
    "high": "🟢 высокая",
    "medium": "🟡 средняя",
    "low": "⚪ низкая",
}
CONFIDENCE_ICON = {"high": "🟢", "medium": "🟡", "low": "⚪"}
CONFIDENCE_LEGEND = (
    "Обозначения уверенности: 🟢 высокая, 🟡 средняя, ⚪ низкая. "
    "Значок показывает уверенность в конкретном выводе, а не степень опасности."
)
QUALITY = {
    "diagnostic": "приемлемое",
    "limited": "ограниченное",
    "non_diagnostic": "непригодное для надежного разбора",
}
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


def _escape_text(value: str) -> str:
    user_facing = re.sub(
        r"\bvisual[-\s]+(?:preflight|проход\w*)\b",
        "проверка изображения",
        value,
        flags=re.IGNORECASE,
    )
    user_facing = user_facing.replace(
        "LOCKED_IMAGE_FACTS",
        "данные проверки изображения",
    )
    return escape(" ".join(user_facing.split()))


def _is_unavailable(value: str) -> bool:
    return value.strip().lower().startswith("не оценивается")


def _compact_list(items: list[str]) -> str:
    return "\n".join(f"• {_escape_text(item)}" for item in items) or "• нет"


def _repeats_rhythm_observation(label: str, rhythm_value: str) -> bool:
    label_lower = label.lower()
    rhythm_lower = rhythm_value.lower()
    return any(
        marker in label_lower and marker in rhythm_lower for marker in RHYTHM_CONCEPT_MARKERS
    )


NON_ECG_MESSAGE = (
    "⚕️ <b>Предварительный разбор ЭКГ</b>\n\n"
    f"<b>Кратко:</b> {NON_ECG_IMPRESSION}"
)
BLOCKQUOTE_OPEN = "<blockquote expandable>"
BLOCKQUOTE_CLOSE = "</blockquote>"


def _expandable(body: str) -> str:
    """Collapse a long block so the report stays scannable in the chat."""
    return BLOCKQUOTE_OPEN + body + BLOCKQUOTE_CLOSE


def _report_sections(report: ECGReport) -> tuple[list[str], list[str]]:
    """Split the report into the sections a reader needs first and the rest.

    The lead carries the impression and the urgent findings so they survive in a
    notification preview; the supporting detail follows in its own message.
    """
    calibration = report.calibration
    speed = f"{calibration.speed_mm_s:g} мм/с" if calibration.speed_mm_s is not None else "не читается"
    if calibration.gain_mm_mv is not None:
        gain = f"{calibration.gain_mm_mv:g} мм/мВ"
    elif calibration.gain_values_mm_mv:
        values = "/".join(f"{value:g}" for value in calibration.gain_values_mm_mv)
        gain = f"{values} мм/мВ (двойное усиление)"
    else:
        gain = "не читается"

    if report.urgent_findings:
        urgent_text = "\n".join(
            f"• <b>{_escape_text(item.finding)}</b> {CONFIDENCE_ICON[item.confidence]} — "
            f"{_escape_text(item.rationale)}; {_escape_text(item.recommended_action)}"
            for item in report.urgent_findings
        )
    else:
        urgent_text = "• Потенциально неотложных признаков на снимке не выявлено."

    lead = [
        "⚕️ <b>Предварительный разбор ЭКГ</b>",
        f"<b>Кратко:</b> {_escape_text(report.concise_impression)}",
        "🚨 <b>Срочные признаки:</b>\n" + urgent_text,
        (
            f"<b>Качество снимка:</b> {QUALITY[report.image_quality.grade]}; "
            f"общая уверенность — {CONFIDENCE[report.overall_confidence]}."
        ),
    ]

    sections = [
        (
            f"<b>Калибровка:</b> {speed}, {gain} "
            f"(уверенность: {CONFIDENCE[calibration.confidence]}).\n"
            f"Исходная подпись: {_escape_text(calibration.raw_text or 'не читается')}."
        ),
    ]

    if report.image_quality.visible_leads:
        sections.append(
            "<b>Видимые отведения:</b> "
            + _escape_text(", ".join(report.image_quality.visible_leads))
        )

    finding_markers: dict[str, list[int]] = {}
    for marker, finding in enumerate(report.localized_findings, start=1):
        finding_markers.setdefault(finding.related_observation, []).append(marker)

    observation_fields = [
        ("rhythm", "Ритм", report.observations.rhythm),
        ("heart_rate", "ЧСС", report.observations.heart_rate),
        ("pr_interval", "PQ/PR", report.observations.pr_interval),
        ("qrs_complex", "QRS", report.observations.qrs_complex),
        ("electrical_axis", "ЭОС", report.observations.electrical_axis),
        ("q_waves", "Q", report.observations.q_waves),
        ("r_progression", "Прогрессия R", report.observations.r_progression),
        ("st_t", "ST–T", report.observations.st_t),
        ("qt_qtc", "QT/QTc", report.observations.qt_qtc),
    ]
    observations = [
        f"• <b>{label}:</b>"
        + "".join(f" [{marker}]" for marker in finding_markers.get(field_name, []))
        + f" {_escape_text(item.value)} "
        f"{CONFIDENCE_ICON[item.confidence]} — {_escape_text(item.evidence)}"
        for field_name, label, item in observation_fields
        if not _is_unavailable(item.value)
    ]
    sections.append(
        "<b>Параметры:</b>\n"
        + _expandable(
            "\n".join(observations)
            or "• Недостаточно данных для надёжной оценки параметров."
        )
    )

    interpretations = []
    for item in report.possible_interpretations:
        if _repeats_rhythm_observation(item.label, report.observations.rhythm.value):
            continue
        evidence = "; ".join(item.supporting_evidence) or "нет"
        missing = "; ".join(item.contradicting_or_missing_evidence) or "нет"
        interpretations.append(
            f"• <b>{_escape_text(item.label)}</b> {CONFIDENCE_ICON[item.confidence]}\n"
            f"  За: {_escape_text(evidence)}; ограничение: {_escape_text(missing)}"
        )
    sections.append(
        "<b>Предположения:</b>\n" + ("\n".join(interpretations) or "• нет")
    )

    if report.localized_findings:
        localized = [
            f"• <b>[{marker}] {_escape_text(item.finding)}</b>"
            for marker, item in enumerate(report.localized_findings, start=1)
        ]
        sections.append("<b>Отметки на снимке:</b>\n" + "\n".join(localized))

    quality_issues = (
        report.image_quality.issues if report.image_quality.grade != "diagnostic" else []
    )
    limitations = list(dict.fromkeys(quality_issues + report.limitations))
    sections.extend(
        [
            "<b>Ограничения:</b>\n" + _expandable(_compact_list(limitations)),
            "<b>Проверить:</b>\n" + _expandable(_compact_list(report.recommended_checks)),
        ]
    )
    return lead, sections


def format_report(report: ECGReport) -> str:
    if report.concise_impression.strip() == NON_ECG_IMPRESSION:
        return NON_ECG_MESSAGE
    lead, detail = _report_sections(report)
    return "\n\n".join(lead + detail)


def format_report_messages(report: ECGReport) -> list[str]:
    """Render the report as the Telegram messages to send, in order."""
    if report.concise_impression.strip() == NON_ECG_IMPRESSION:
        return [NON_ECG_MESSAGE]
    lead, detail = _report_sections(report)
    return [
        chunk
        for part in ("\n\n".join(lead), "\n\n".join(detail))
        if part
        for chunk in split_html_message(part)
    ]


def split_html_message(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]

    # Every report section contains balanced tags. Prefer section boundaries so the
    # formatting survives in every Telegram message.
    chunks: list[str] = []
    current = ""
    for section in text.split("\n\n"):
        candidate = section if not current else current + "\n\n" + section
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(section) <= limit:
            current = section
        else:
            chunks.extend(_split_oversized_html_section(section, limit))
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_html_section(text: str, limit: int) -> list[str]:
    # An expandable block must not be cut in half. Split what is inside it, wrap every
    # piece separately, and repeat the section heading so each part stays identifiable.
    start = text.find(BLOCKQUOTE_OPEN)
    if start >= 0 and text.endswith(BLOCKQUOTE_CLOSE):
        heading = text[:start]
        inner = text[start + len(BLOCKQUOTE_OPEN) : -len(BLOCKQUOTE_CLOSE)]
        budget = limit - len(BLOCKQUOTE_OPEN) - len(BLOCKQUOTE_CLOSE) - len(heading)
        if budget > 0:
            return [
                heading + _expandable(chunk)
                for chunk in _split_oversized_html_section(inner, budget)
            ]

    # Lines generated by the formatter have balanced tags, so prefer their boundaries.
    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = line if not current else current + "\n" + line
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(line) <= limit:
            current = line
        else:
            chunks.extend(_split_plain_html(line, limit))
    if current:
        chunks.append(current)
    return chunks


def _split_plain_html(text: str, limit: int) -> list[str]:
    # A single pathological line may still exceed the limit. Strip its formatting and
    # split escaped plain text without cutting HTML entities.
    plain = unescape(re.sub(r"</?(?:b|i|blockquote(?: expandable)?)>", "", text))
    chunks: list[str] = []
    remaining = plain
    while remaining:
        current: list[str] = []
        escaped_length = 0
        for character in remaining:
            encoded = escape(character)
            if current and escaped_length + len(encoded) > limit:
                break
            current.append(character)
            escaped_length += len(encoded)
        cut = len(current)
        if cut < len(remaining):
            preferred = max(
                remaining.rfind("\n\n", 0, cut),
                remaining.rfind("\n", 0, cut),
                remaining.rfind(" ", 0, cut),
            )
            if preferred >= cut // 2:
                cut = preferred + 1
        chunk = remaining[:cut].rstrip()
        if chunk:
            chunks.append(escape(chunk))
        remaining = remaining[cut:].lstrip()
    return chunks
