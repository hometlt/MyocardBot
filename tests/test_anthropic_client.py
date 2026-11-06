from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from miocard.anthropic_client import (
    STANDARD_LEADS,
    AnthropicClient,
    AnthropicServiceError,
    TokenBalance,
)
from miocard.config import Settings
from miocard.imaging import ProcessedImage
from miocard.models import (
    ANALYSIS_JSON_SCHEMA,
    NON_ECG_IMPRESSION,
    ECGImageFacts,
    ECGReport,
)


def test_retryable_gateway_error_is_retried() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(502, json={"error": {"message": "temporary"}})
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="secret",
            anthropic_model="claude-sonnet-5",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        try:
            with patch("miocard.anthropic_client.asyncio.sleep", new=AsyncMock()):
                assert await client._request_text({"messages": []}, "test") == "ok"
        finally:
            await client.close()

    asyncio.run(scenario())
    assert calls == 2


def test_tokenator_balance_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://api.tokenator.cloud/anthropic/v1/tokens"
        assert request.headers["x-api-key"] == "sk-fp-test-key"
        return httpx.Response(
            200,
            json={"used": 1250, "limit": 8000000, "remaining": 7998750},
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="sk-fp-test-key",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        try:
            balance = await client.get_token_balance()
            assert balance == TokenBalance(
                used=1250,
                limit=8000000,
                remaining=7998750,
            )
        finally:
            await client.close()

    asyncio.run(scenario())


def test_string_rate_limit_error_is_retried_and_reported_cleanly() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json="Too many requests")

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="secret",
            anthropic_model="claude-sonnet-5",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        sleep = AsyncMock()
        try:
            with patch("miocard.anthropic_client.asyncio.sleep", new=sleep):
                try:
                    await client._request_text({"messages": []}, "основной анализ")
                except AnthropicServiceError as exc:
                    assert "временно ограничил частоту запросов" in str(exc)
                    assert "Too many requests" in str(exc)
                else:
                    raise AssertionError("Ожидалась ошибка ограничения частоты")
        finally:
            await client.close()

        assert [call.args[0] for call in sleep.await_args_list] == [5.0, 10.0]

    asyncio.run(scenario())
    assert calls == 3


def test_rate_limit_retry_respects_retry_after_header() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "7"},
                json={"error": "rate limited"},
            )
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="secret",
            anthropic_model="claude-sonnet-5",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        sleep = AsyncMock()
        try:
            with patch("miocard.anthropic_client.asyncio.sleep", new=sleep):
                assert await client._request_text({"messages": []}, "test") == "ok"
        finally:
            await client.close()
        sleep.assert_awaited_once_with(7.0)

    asyncio.run(scenario())
    assert calls == 2


def test_exhausted_output_limit_is_not_retried() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            json={
                "error": "Output limit exceeded",
                "usage": {
                    "output_limit": 150000,
                    "output_remaining": -764,
                    "output_tokens_used": 150764,
                },
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="secret",
            anthropic_model="claude-sonnet-5",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        sleep = AsyncMock()
        try:
            with patch("miocard.anthropic_client.asyncio.sleep", new=sleep):
                try:
                    await client._request_text({"messages": []}, "инвентаризация изображения")
                except AnthropicServiceError as exc:
                    assert "Лимит выходных токенов API-ключа исчерпан" in str(exc)
                else:
                    raise AssertionError("Ожидалась ошибка исчерпанного лимита")
        finally:
            await client.close()
        sleep.assert_not_awaited()

    asyncio.run(scenario())
    assert calls == 1


def test_prompt_cache_can_be_disabled() -> None:
    settings = Settings(
        telegram_bot_token="123456:telegram-secret",
        anthropic_api_key="secret",
        anthropic_model="claude-sonnet-5",
        prompt_cache_enabled=False,
        _env_file=None,
    )
    client = AnthropicClient(settings)
    try:
        assert isinstance(client._system_field(), str)
    finally:
        asyncio.run(client.close())


def test_schema_sent_to_the_model_carries_no_generated_titles() -> None:
    serialized = json.dumps(ANALYSIS_JSON_SCHEMA, ensure_ascii=False)

    assert '"title"' not in serialized
    # Dropping titles must not drop the structure the model has to follow.
    assert set(ANALYSIS_JSON_SCHEMA["properties"]) == {"image_facts", "report"}
    assert "ECGReport" in ANALYSIS_JSON_SCHEMA["$defs"]


def test_wave_not_visible_is_not_mistaken_for_unreadable_lead() -> None:
    readable = {"I", "II", "V1", "V2"}
    assert not AnthropicClient._claims_locked_lead_is_unavailable(
        "Патологических Q не видно в отведениях I, II, V1 и V2.",
        readable,
    )
    assert AnthropicClient._claims_locked_lead_is_unavailable(
        "Верхние отведения I и II плохо читаемы.",
        readable,
    )


def observation_block() -> dict[str, dict[str, str]]:
    item = {"value": "Не оценивается", "confidence": "low", "evidence": "Тест"}
    return {
        "rhythm": dict(item),
        "heart_rate": dict(item),
        "pr_interval": dict(item),
        "qrs_complex": dict(item),
        "electrical_axis": dict(item),
        "q_waves": dict(item),
        "r_progression": dict(item),
        "st_t": dict(item),
        "qt_qtc": dict(item),
    }


def test_clinical_consistency_downgrades_ectopy_and_removes_unsupported_hypertrophy() -> None:
    observations = observation_block()
    observations["rhythm"] = {
        "value": "Синусовый ритм с возможными наджелудочковыми экстрасистолами",
        "confidence": "medium",
        "evidence": "Измененный P перед ранними QRS различим недостаточно четко.",
    }
    observations["qrs_complex"] = {
        "value": "Высокий грудной вольтаж, возможен гипертрофический паттерн",
        "confidence": "medium",
        "evidence": "Высокая амплитуда V3–V6; точные амплитуды вручную не измерены.",
    }
    observations["st_t"] = {
        "value": "На доступном изображении нет убедительного острого ишемического паттерна",
        "confidence": "medium",
        "evidence": "Нет убедительных территориальных изменений.",
    }
    report = ECGReport.model_validate(
        {
            "image_quality": {
                "grade": "limited",
                "issues": [],
                "visible_leads": list(STANDARD_LEADS),
                "usable_leads": list(STANDARD_LEADS),
                "unreadable_leads": [],
            },
            "calibration": {
                "raw_text": "25 mm/s 10 mm/mV",
                "speed_mm_s": 25,
                "gain_mm_mv": 10,
                "gain_values_mm_mv": [10],
                "gain_notes": "Стандартная калибровка",
                "confidence": "high",
                "source": "Подпись",
            },
            "observations": observations,
            "possible_interpretations": [
                {
                    "label": "Возможные наджелудочковые экстрасистолы",
                    "confidence": "medium",
                    "supporting_evidence": ["Повторяются укороченные R–R"],
                    "contradicting_or_missing_evidence": [
                        "P перед ранними комплексами недостаточно четкий"
                    ],
                },
                {
                    "label": "Высоковольтный грудной паттерн",
                    "confidence": "medium",
                    "supporting_evidence": ["Высокая амплитуда QRS в V3–V6"],
                    "contradicting_or_missing_evidence": [
                        "Нельзя исключить гипертрофический паттерн"
                    ],
                },
            ],
            "urgent_findings": [],
            "limitations": [
                "Очень низкая амплитуда aVL и V1 ограничивает оценку их морфологии"
            ],
            "recommended_checks": [
                "Проверить амплитудные критерии гипертрофии на оригинале"
            ],
            "overall_confidence": "medium",
            "concise_impression": (
                "Синусовый нерегулярный ритм. Высокий вольтаж допускает гипертрофию."
            ),
        }
    )

    AnthropicClient._apply_clinical_consistency_rules(report)

    assert report.observations.rhythm.confidence == "low"
    assert report.possible_interpretations[0].confidence == "low"
    serialized = json.dumps(report.model_dump(), ensure_ascii=False)
    serialized_lower = serialized.lower()
    assert "гипертроф" not in serialized_lower
    assert "Высокий вольтаж грудных комплексов" in serialized
    assert "нет убедительного острого ишемического паттерна" in serialized_lower
    assert "Очень низкая амплитуда aVL и V1" in serialized

    report.observations.rhythm.value = "Синусовый, преимущественно регулярный ритм"
    report.observations.rhythm.confidence = "low"
    report.observations.rhythm.evidence = "Сходные P предшествуют узким QRS в I и V4."
    AnthropicClient._apply_clinical_consistency_rules(report)
    assert report.observations.rhythm.confidence == "medium"


def test_anthropic_request_and_structured_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "secret"
        body = json.loads(request.content)
        assert body["model"] == "claude-sonnet-5"
        assert body["stream"] is False
        assert body["max_tokens"] == 3584
        assert body["output_config"]["effort"] == "medium"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert set(body["output_config"]["format"]["schema"]["properties"]) == {
            "image_facts",
            "report",
        }
        # Constant instructions belong to the cacheable system prefix, so only the
        # per-image metadata and the user's context travel in the message itself.
        system = body["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert "низкая амплитуда" in system[0]["text"]
        assert "не является насыщением" in system[0]["text"]
        assert body["messages"][0]["content"][0]["type"] == "image"
        prompt = body["messages"][0]["content"][1]["text"]
        assert "низкая амплитуда" not in prompt
        assert "без ФИО" in prompt
        facts = {
            "is_ecg_image": True,
            "non_ecg_reason": "",
            "visible_lead_labels": ["II"],
            "inferred_leads": [
                {
                    "lead": "I",
                    "confidence": "high",
                    "evidence": "Верхняя строка перед II, метка обрезана.",
                }
            ],
            "missing_standard_leads": [],
            "readable_waveform_leads": ["II"],
            "unreadable_waveform_leads": [],
            "calibration_text": "25 mm/s 10 mm/mV",
            "speed_mm_s": 25,
            "gain_values_mm_mv": [10],
        }
        report = {
            "image_quality": {
                "grade": "Limited",
                "issues": ["Небольшой наклон"],
                "visible_leads": ["I"],
                "usable_leads": ["I", "II"],
                "unreadable_leads": [],
            },
            "calibration": {
                "raw_text": "ошибочная строка модели",
                "speed_mm_s": 25,
                "gain_mm_mv": 10,
                "gain_values_mm_mv": [10],
                "gain_notes": "Одинаковое усиление",
                "confidence": "Medium",
                "source": "подпись",
            },
            "observations": observation_block(),
            "possible_interpretations": [],
            "urgent_findings": [],
            "limitations": ["Только фотография"],
            "recommended_checks": [
                "Сверить с оригиналом",
                "Повторить запись с включенным отведением I",
            ],
            "overall_confidence": "Low",
            "concise_impression": "Предварительный отчет.",
        }
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"image_facts": facts, "report": report}),
                    }
                ],
                "stop_reason": "end_turn",
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="secret",
            anthropic_model="claude-sonnet-5",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        image = ProcessedImage(
            data=b"png",
            media_type="image/png",
            width=1000,
            height=800,
            estimated_visual_tokens=1036,
            blur_score=100,
            brightness=220,
            contrast=40,
        )
        try:
            report = await client.analyze(image, "без ФИО")
            assert report.image_quality.grade == "limited"
            assert report.overall_confidence == "low"
            assert report.image_quality.visible_leads == ["I", "II"]
            assert report.image_quality.usable_leads == ["I", "II"]
            assert report.image_quality.unreadable_leads == []
            assert report.calibration.raw_text == "25 mm/s 10 mm/mV"
            assert report.recommended_checks == ["Сверить с оригиналом"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_tokenator_uses_prompt_schema_and_extracts_json() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert str(request.url) == "https://api.tokenator.cloud/anthropic/v1/messages"
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.6-terra"
        assert body["stream"] is False
        assert "output_config" not in body
        assert body["max_tokens"] == 3584
        assert body["reasoning"] == {"enabled": False}
        # The gateway is not sent cache_control blocks, so its system stays a plain string.
        assert isinstance(body["system"], str)
        assert "JSON Schema" in body["system"]
        assert '"title"' not in body["system"]
        facts = {
            "is_ecg_image": True,
            "non_ecg_reason": "",
            "visible_lead_labels": ["I", "II"],
            "inferred_leads": [],
            "missing_standard_leads": [],
            "readable_waveform_leads": ["I", "II"],
            "unreadable_waveform_leads": [],
            "calibration_text": "10/5 mm/mV 25 mm/s",
            "speed_mm_s": 25,
            "gain_values_mm_mv": [10, 5],
        }
        report = {
            "image_quality": {
                "grade": "limited",
                "issues": ["Тестовый снимок", "Верхние отведения обрезаны"],
                "visible_leads": ["I"],
                "usable_leads": ["I"],
                "unreadable_leads": ["II"],
            },
            "calibration": {
                "raw_text": "не читается",
                "speed_mm_s": None,
                "gain_mm_mv": None,
                "gain_values_mm_mv": [],
                "gain_notes": "нет данных",
                "confidence": "low",
                "source": "не читается",
            },
            "observations": {
                **observation_block(),
                "electrical_axis": {
                    "value": "Не оценивается",
                    "confidence": "low",
                    "evidence": "I/II/III плохо читаемы и верхние отведения обрезаны",
                },
            },
            "possible_interpretations": [],
            "urgent_findings": [],
            "limitations": ["Синтетические данные", "Отведения I/II не читаются"],
            "recommended_checks": [
                "Проверить прототип",
                "Повторить съемку, так как отведения I/II не читаются",
            ],
            "overall_confidence": "low",
            "concise_impression": "Верхние отведения I/II плохо читаемы. Тестовый отчет.",
        }
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": "Результат:\n```json\n"
                        + json.dumps({"image_facts": facts, "report": report})
                        + "\n```",
                    }
                ],
                "stop_reason": "end_turn",
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="sk-fp-test-key",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        image = ProcessedImage(
            data=b"png",
            media_type="image/png",
            width=1000,
            height=800,
            estimated_visual_tokens=1036,
            blur_score=100,
            brightness=220,
            contrast=40,
        )
        try:
            report = await client.analyze(image, model="gpt-5.6-terra")
            assert report.concise_impression == "Тестовый отчет."
            assert report.calibration.gain_values_mm_mv == [10, 5]
            assert report.calibration.gain_mm_mv is None
            assert report.calibration.source == "Автоматическая проверка изображения"
            assert "соответствие группам" in report.calibration.gain_notes
            assert report.image_quality.issues == ["Тестовый снимок"]
            assert report.image_quality.usable_leads == ["I", "II"]
            assert report.image_quality.unreadable_leads == []
            assert report.observations.electrical_axis.value == "Не оценивается"
            assert "visual preflight" not in report.observations.electrical_axis.evidence
            assert report.observations.electrical_axis.evidence == (
                "Оценка выполнена по доступным кривым."
            )
            assert report.limitations == ["Синтетические данные"]
            assert report.recommended_checks == ["Проверить прототип"]
        finally:
            await client.close()

    asyncio.run(scenario())
    assert calls == 1


def test_non_ecg_image_stops_after_single_pass() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["max_tokens"] == 3584
        assert body["reasoning"] == {"enabled": False}
        facts = {
            "is_ecg_image": False,
            "non_ecg_reason": "Обычная фотография без ЭКГ-кривых.",
            "visible_lead_labels": ["I"],
            "inferred_leads": [
                {
                    "lead": "V4",
                    "confidence": "high",
                    "evidence": "Ложное предположение для проверки очистки.",
                }
            ],
            "missing_standard_leads": [],
            "readable_waveform_leads": ["I"],
            "unreadable_waveform_leads": [],
            "calibration_text": "ложная калибровка",
            "speed_mm_s": 25,
            "gain_values_mm_mv": [10],
        }
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"image_facts": facts, "report": None}),
                    }
                ],
                "stop_reason": "end_turn",
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="sk-fp-test-key",
            _env_file=None,
        )
        client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        image = ProcessedImage(
            data=b"png",
            media_type="image/png",
            width=1000,
            height=800,
            estimated_visual_tokens=1036,
            blur_score=100,
            brightness=220,
            contrast=40,
        )
        try:
            report = await client.analyze(image)
            assert report.image_quality.grade == "non_diagnostic"
            assert report.image_quality.visible_leads == []
            assert report.calibration.speed_mm_s is None
            assert report.concise_impression == NON_ECG_IMPRESSION
        finally:
            await client.close()

    asyncio.run(scenario())
    assert calls == 1


def test_identical_analysis_is_loaded_from_persistent_cache(tmp_path: Path) -> None:
    calls = 0
    facts = {
        "is_ecg_image": True,
        "non_ecg_reason": "",
        "visible_lead_labels": ["I", "II"],
        "inferred_leads": [],
        "missing_standard_leads": [],
        "readable_waveform_leads": ["I", "II"],
        "unreadable_waveform_leads": [],
        "calibration_text": "25 mm/s 10 mm/mV",
        "speed_mm_s": 25,
        "gain_values_mm_mv": [10],
    }
    report_data = {
        "image_quality": {
            "grade": "limited",
            "issues": [],
            "visible_leads": ["I", "II"],
            "usable_leads": ["I", "II"],
            "unreadable_leads": [],
        },
        "calibration": {
            "raw_text": "25 mm/s 10 mm/mV",
            "speed_mm_s": 25,
            "gain_mm_mv": 10,
            "gain_values_mm_mv": [10],
            "gain_notes": "Одно усиление",
            "confidence": "high",
            "source": "Подпись",
        },
        "observations": observation_block(),
        "possible_interpretations": [],
        "urgent_findings": [],
        "limitations": [],
        "recommended_checks": [],
        "overall_confidence": "low",
        "concise_impression": "Кэшируемый тестовый отчет.",
    }

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"image_facts": facts, "report": report_data}
                        ),
                    }
                ],
                "stop_reason": "end_turn",
            },
        )

    async def scenario() -> None:
        settings = Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="sk-fp-test-key",
            analysis_cache_enabled=True,
            analysis_cache_dir=tmp_path,
            _env_file=None,
        )
        image = ProcessedImage(
            data=b"same-image",
            media_type="image/png",
            width=1000,
            height=800,
            estimated_visual_tokens=1036,
            blur_score=100,
            brightness=220,
            contrast=40,
        )
        first_client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        try:
            first = await first_client.analyze(image, "одинаковый контекст")
        finally:
            await first_client.close()

        second_client = AnthropicClient(settings, transport=httpx.MockTransport(handler))
        try:
            second = await second_client.analyze(image, "одинаковый контекст")
        finally:
            await second_client.close()

        assert first == second
        assert second.concise_impression == "Кэшируемый тестовый отчет."

    asyncio.run(scenario())
    assert calls == 1


def test_infers_any_cropped_lead_from_existing_waveform_row() -> None:
    labeled = ["II", "III", "aVR", "aVF", "V1", "V2", "V3", "V5", "V6"]
    facts = ECGImageFacts.model_validate(
        {
            "is_ecg_image": True,
            "non_ecg_reason": "",
            "visible_lead_labels": labeled,
            "inferred_leads": [
                {
                    "lead": "I",
                    "confidence": "high",
                    "evidence": "Отдельная верхняя строка непосредственно перед II.",
                },
                {
                    "lead": "aVL",
                    "confidence": "medium",
                    "evidence": "Строка между aVR и aVF с неразборчивой меткой.",
                },
                {
                    "lead": "V4",
                    "confidence": "medium",
                    "evidence": "Строка между V3 и V5 с обрезанной меткой.",
                },
            ],
            "missing_standard_leads": ["I", "aVL", "V4"],
            "readable_waveform_leads": labeled,
            "unreadable_waveform_leads": [],
            "calibration_text": "25 mm/s 10 mm/mV",
            "speed_mm_s": 25,
            "gain_values_mm_mv": [10],
        }
    )

    normalized = AnthropicClient._normalize_image_facts(facts)

    assert normalized.visible_lead_labels == list(STANDARD_LEADS)
    assert normalized.readable_waveform_leads == normalized.visible_lead_labels
    assert normalized.missing_standard_leads == []
    assert [item.lead for item in normalized.inferred_leads] == ["I", "aVL", "V4"]
