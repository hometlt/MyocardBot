import pytest

from miocard.anthropic_client import ModelUsage, TokenBalance, UsageStats
from miocard.bot import (
    HELP,
    MODEL_CHOICES,
    PROGRESS_UPDATE_INTERVAL_SECONDS,
    WELCOME,
    analysis_status_text,
    can_select_model,
    format_usage_notification,
    format_usage_stats,
    greeting_for_user,
    model_keyboard,
    user_has_access,
)


@pytest.mark.parametrize(
    ("user_id", "first_name", "username", "expected"),
    [
        (153241307, "Любое имя", "moderator", "Здравствуй, Модератор"),
        (2053404449, "Любое имя", "masha", "Привет, Маша ❤️"),
        (100, "Алексей", "alex", "Здравствуйте, Алексей"),
        (101, "   ", "telegram_user", "Здравствуйте, @telegram_user"),
        (102, None, None, "Здравствуйте"),
    ],
)
def test_greeting_for_user(
    user_id: int,
    first_name: str | None,
    username: str | None,
    expected: str,
) -> None:
    moderators = frozenset({153241307})
    assert greeting_for_user(user_id, first_name, username, moderators) == expected


@pytest.mark.parametrize(
    ("elapsed", "expected_stage", "expected_time"),
    [
        (0, "Подготавливаю изображение", "0:00"),
        (12, "Проверяю отведения и калибровку", "0:12"),
        (50, "Выполняю структурированный разбор", "0:50"),
        (95, "Формирую итоговый отчёт", "1:35"),
        (130, "больше обычного", "2:10"),
    ],
)
def test_analysis_status_text(elapsed: float, expected_stage: str, expected_time: str) -> None:
    text = analysis_status_text(elapsed, frame_index=3)
    assert expected_stage in text
    assert expected_time in text
    assert "Модель:" not in text
    assert "%" not in text


def test_model_keyboard_contains_all_choices() -> None:
    keyboard = model_keyboard()
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    assert len(buttons) == len(MODEL_CHOICES)
    assert [button.callback_data for button in buttons] == [
        f"model:{model_id}" for _, model_id in MODEL_CHOICES
    ]
    assert [button.text for button in buttons] == [label for label, _ in MODEL_CHOICES]


def test_only_moderators_can_select_model() -> None:
    moderators = frozenset({153241307, 987654321})
    assert can_select_model(153241307, moderators)
    assert can_select_model(987654321, moderators)
    assert not can_select_model(2053404449, moderators)


def test_moderators_always_have_service_access() -> None:
    moderators = frozenset({153241307})
    allowed = frozenset({2053404449})

    assert user_has_access(153241307, allowed, moderators)
    assert user_has_access(2053404449, allowed, moderators)
    assert not user_has_access(123456789, allowed, moderators)
    assert user_has_access(123456789, frozenset(), moderators)


def test_progress_updates_every_second() -> None:
    assert PROGRESS_UPDATE_INTERVAL_SECONDS == 1.0


def test_confidence_legend_is_shown_in_start_and_help() -> None:
    for text in (WELCOME, HELP):
        assert "🟢 высокая" in text
        assert "🟡 средняя" in text
        assert "⚪ низкая" in text
        assert "не степень опасности" in text


def test_context_hint_asks_for_history_instead_of_recording_time() -> None:
    for text in (WELCOME, HELP):
        assert "время записи" not in text
        assert "предыдущих ЭКГ" in text
        assert "динамик" in text


def test_annotation_legend_is_shown_in_start_and_help() -> None:
    for text in (WELCOME, HELP):
        assert "Метки [1], [2], [3]" in text
        assert "размеченн" in text


def test_usage_notification_shows_raw_and_package_tokens() -> None:
    text = format_usage_notification(
        "gpt-5.6-sol",
        [{"input_tokens": 7517, "output_tokens": 3325}],
        TokenBalance(used=800000, limit=8000000, remaining=7200000),
        TokenBalance(used=819516, limit=8000000, remaining=7180484),
    )

    assert "Модель: Sol" in text
    assert "Исходящие токены (ответ модели): 3 325" in text
    assert "Входные токены" not in text
    assert "10 842" not in text
    assert "Списано с пакета за время анализа: 19 516" in text
    assert "Осталось токенов: 7 180 484" in text


def test_cached_usage_notification_reports_zero_spend() -> None:
    balance = TokenBalance(used=900000, limit=8000000, remaining=7100000)
    text = format_usage_notification("gpt-5.6-sol", [], balance, balance)

    assert "ответ получен из кэша" in text
    assert "Списано с пакета за время анализа: 0" in text


def test_usage_stats_report_covers_package_and_directions() -> None:
    stats = UsageStats(
        analyses=3,
        cache_hits=1,
        api_errors=1,
        by_model={
            "gpt-5.6-sol": ModelUsage(requests=4, input_tokens=120000, output_tokens=3500),
            "claude-opus-4-8": ModelUsage(requests=1, input_tokens=9000, output_tokens=800),
        },
    )

    text = format_usage_stats(
        "gpt-5.6-sol",
        stats,
        TokenBalance(used=803754, limit=8000000, remaining=7196246),
    )

    assert "Текущая модель: Sol" in text
    assert "Запас: 8 000 000" in text
    assert "Использовано: 803 754 (10.0%)" in text
    assert "Осталось: 7 196 246" in text
    assert "Разборов: 3 (из кэша: 1)" in text
    assert "API-запросов: 5 (с ошибкой: 1)" in text
    assert "Входящие токены (запрос, включая изображение): 129 000" in text
    assert "Исходящие токены (ответ модели): 4 300" in text
    assert "Всего токенов: 133 300" in text
    assert "Sol: 4 · 120 000 · 3 500" in text
    assert "Opus: 1 · 9 000 · 800" in text


def test_usage_stats_report_without_gateway_balance() -> None:
    text = format_usage_stats("claude-sonnet-5", UsageStats(), None)

    assert "остаток недоступен" in text
    assert "Всего токенов: 0" in text
