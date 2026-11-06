import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from miocard.anthropic_client import ModelUsage, TokenBalance, UsageStats
from miocard.bot import (
    HELP,
    MODEL_CHOICES,
    ONLINE_NOTIFICATION,
    PROGRESS_UPDATE_INTERVAL_SECONDS,
    RELEASE_HISTORY_TEXT,
    WELCOME,
    ECGTelegramRouter,
    analysis_status_text,
    analysis_support_keyboard,
    can_select_model,
    can_view_prompt,
    format_admin_audit_summary,
    format_admin_upload_caption,
    format_usage_notification,
    format_usage_stats,
    greeting_for_user,
    is_moderator,
    model_keyboard,
    notify_online,
    support_menu_keyboard,
    user_has_access,
)
from miocard.release import CURRENT_RELEASE, ReleaseNotificationStore
from miocard.support import PendingSupport, SupportRoute, SupportStore


def make_router(tmp_path, *, moderator_user_ids, allowed_user_ids=frozenset()):
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=moderator_user_ids,
        allowed_user_ids=allowed_user_ids,
    )
    router.support = SupportStore(tmp_path / "support.sqlite3")
    router.shutdown_event = asyncio.Event()
    return router


def make_message(user_id: int, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        text=text,
        answer=AsyncMock(),
    )


def test_online_notification_contains_current_release() -> None:
    assert f"v{CURRENT_RELEASE.version}" in ONLINE_NOTIFICATION
    assert "Что изменилось:" in ONLINE_NOTIFICATION
    assert all(change in ONLINE_NOTIFICATION for change in CURRENT_RELEASE.changes)


@pytest.mark.asyncio
async def test_notify_online_sends_release_once_to_every_allowed_user(tmp_path) -> None:
    bot = AsyncMock()
    settings = SimpleNamespace(
        moderator_user_ids=frozenset({100, 200}),
        allowed_user_ids=frozenset({200, 300}),
    )

    store = ReleaseNotificationStore(tmp_path / "release-state.json")
    await notify_online(bot, settings, store)  # type: ignore[arg-type]
    await notify_online(bot, settings, store)  # type: ignore[arg-type]

    assert bot.send_message.await_args_list == [
        call(100, ONLINE_NOTIFICATION),
        call(200, ONLINE_NOTIFICATION),
        call(300, ONLINE_NOTIFICATION),
    ]


def test_release_history_text_contains_current_and_previous_versions() -> None:
    assert "История версий" in RELEASE_HISTORY_TEXT
    assert f"v{CURRENT_RELEASE.version}" in RELEASE_HISTORY_TEXT
    assert "v0.5.0" in RELEASE_HISTORY_TEXT


@pytest.mark.asyncio
async def test_updates_command_sends_release_history_to_allowed_user() -> None:
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        allowed_user_ids=frozenset(),
        moderator_user_ids=frozenset(),
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=123),
        answer=AsyncMock(),
    )

    await router.updates_command(message)

    sent_text = "\n".join(call.args[0] for call in message.answer.await_args_list)
    assert "История версий" in sent_text
    assert f"v{CURRENT_RELEASE.version}" in sent_text


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


def test_support_keyboards_bind_reports_to_analysis() -> None:
    case_id = "a" * 32
    analysis_buttons = analysis_support_keyboard(case_id).inline_keyboard[0]

    assert [button.callback_data for button in analysis_buttons] == [
        f"support:technical:{case_id}",
        f"support:result:{case_id}",
    ]
    menu_buttons = [button for row in support_menu_keyboard(case_id).inline_keyboard for button in row]
    assert menu_buttons[-1].callback_data == "support:other"


@pytest.mark.asyncio
async def test_open_support_ticket_sends_to_all_moderators(tmp_path) -> None:
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(moderator_user_ids=frozenset({700, 800}))
    router.support = SupportStore(tmp_path / "support.sqlite3")
    router.support.set_pending(100, "other", None)
    message = SimpleNamespace(
        from_user=SimpleNamespace(
            id=100,
            first_name="Анна",
            last_name=None,
            username="anna",
        ),
        chat=SimpleNamespace(id=100),
        message_id=10,
        answer=AsyncMock(),
    )
    bot = AsyncMock()
    bot.send_message.side_effect = [
        SimpleNamespace(message_id=701),
        SimpleNamespace(message_id=801),
    ]
    bot.forward_message.side_effect = [
        SimpleNamespace(message_id=702),
        SimpleNamespace(message_id=802),
    ]

    await router._open_support_ticket(message, bot, PendingSupport("other", None))

    assert bot.forward_message.await_count == 2
    assert bot.forward_message.await_args_list[0].kwargs["chat_id"] == 700
    assert bot.forward_message.await_args_list[1].kwargs["chat_id"] == 800
    route_700 = router.support.get_route(700, 702)
    route_800 = router.support.get_route(800, 802)
    assert route_700 is not None and route_700.moderator_id == 700
    assert route_800 is not None and route_800.moderator_id == 800
    assert router.support.get_pending(100) is None
    assert "передано модераторам" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_user_reply_returns_only_to_the_moderator_who_answered(tmp_path) -> None:
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=frozenset({700, 800}),
        allowed_user_ids=frozenset({100}),
    )
    router.support = SupportStore(tmp_path / "support.sqlite3")
    ticket_id = router.support.create_ticket(100, "other", None)
    route = SupportRoute(ticket_id=ticket_id, user_id=100, moderator_id=700)
    router.support.add_route(
        chat_id=100,
        message_id=50,
        ticket_id=ticket_id,
        user_id=100,
        moderator_id=700,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=100),
        chat=SimpleNamespace(id=100),
        message_id=51,
        reply_to_message=SimpleNamespace(message_id=50),
        answer=AsyncMock(),
    )
    bot = AsyncMock()
    bot.send_message.return_value = SimpleNamespace(message_id=703)
    bot.forward_message.return_value = SimpleNamespace(message_id=704)

    await router.handle_support_message(message, bot)

    assert bot.send_message.await_args.args[0] == route.moderator_id
    assert bot.forward_message.await_args.kwargs["chat_id"] == route.moderator_id
    assert 800 not in [item.args[0] for item in bot.send_message.await_args_list]
    assert "выбранному модератору" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_unsupported_message_does_not_go_to_support() -> None:
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=frozenset({700}),
        allowed_user_ids=frozenset({100}),
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=100),
        answer=AsyncMock(),
    )

    await router.unsupported(message)

    assert "/support" in message.answer.await_args.args[0]


def test_only_moderators_can_select_model() -> None:
    moderators = frozenset({153241307, 987654321})
    assert can_select_model(153241307, moderators)
    assert can_select_model(987654321, moderators)
    assert not can_select_model(2053404449, moderators)


def test_moderators_and_maria_can_view_prompt() -> None:
    moderators = frozenset({153241307, 987654321})

    assert can_view_prompt(153241307, moderators)
    assert can_view_prompt(987654321, moderators)
    assert can_view_prompt(2053404449, moderators)
    assert not can_view_prompt(123456789, moderators)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [153241307, 2053404449])
async def test_prompt_command_sends_exact_working_prompt(user_id: int) -> None:
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=frozenset({153241307}),
        allowed_user_ids=frozenset({2053404449}),
    )
    router.anthropic = SimpleNamespace(
        analysis_system_prompt=lambda: "Точный системный промпт",
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
        answer_document=AsyncMock(),
    )

    await router.prompt_command(message)

    document = message.answer_document.await_args.args[0]
    assert document.data == "Точный системный промпт".encode()
    assert document.filename == "miocard-ai-system-prompt.txt"
    message.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_command_rejects_regular_user() -> None:
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=frozenset({153241307}),
        allowed_user_ids=frozenset({100}),
    )
    router.anthropic = SimpleNamespace(analysis_system_prompt=lambda: "секрет")
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=100),
        answer=AsyncMock(),
        answer_document=AsyncMock(),
    )

    await router.prompt_command(message)

    assert "только модераторам и Марии" in message.answer.await_args.args[0]
    message.answer_document.assert_not_awaited()


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
    assert "API-запросов: 1" in text
    assert "Входящие токены (инструкции и изображение): 7 517" in text
    assert "Исходящие токены (ответ модели): 3 325" in text
    assert "Списано с общего пакета за время анализа: 19 516" in text
    assert "Осталось токенов: 7 180 484" in text
    assert "Баланс пакета общий" in text


def test_cached_usage_notification_reports_zero_spend() -> None:
    balance = TokenBalance(used=900000, limit=8000000, remaining=7100000)
    text = format_usage_notification("gpt-5.6-sol", [], balance, balance)

    assert "ответ получен из кэша" in text
    assert "Списано с общего пакета за время анализа: 0" in text


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
    assert "Opus 4.8: 1 · 9 000 · 800" in text


def test_usage_stats_report_without_gateway_balance() -> None:
    text = format_usage_stats("claude-sonnet-5", UsageStats(), None)

    assert "остаток недоступен" in text
    assert "Всего токенов: 0" in text


def test_admin_audit_summary_contains_user_time_and_per_request_usage() -> None:
    text = format_admin_audit_summary(
        user_id=123456789,
        first_name="Анна",
        last_name="Тестова",
        username="anna_test",
        received_at=datetime(2026, 8, 27, 9, 30, tzinfo=UTC),
        model="gpt-5.6-sol",
        outcome="Готово",
        usages=[{"input_tokens": 7517, "output_tokens": 3325}],
        balance_before=TokenBalance(used=800000, limit=8000000, remaining=7200000),
        balance_after=TokenBalance(used=819516, limit=8000000, remaining=7180484),
        include_content=False,
        cache_hit=False,
    )

    assert "Анна Тестова (@anna_test)" in text
    assert "Telegram ID: 123456789" in text
    assert "27.08.2026" in text
    time_line = next(line for line in text.splitlines() if line.startswith("Время:"))
    assert " UTC+" in time_line or " UTC-" in time_line or time_line.endswith(" UTC")
    assert "?" not in time_line
    assert "Модель: Sol" in text
    assert "Входящие токены: 7 517" in text
    assert "Исходящие токены: 3 325" in text
    assert "Списано с общего пакета: 19 516" in text
    assert "Содержимое: отключено настройкой приватности" in text


def test_admin_upload_caption_combines_summary_and_context_within_limit() -> None:
    caption = format_admin_upload_caption(
        "Журнал анализа",
        "возраст 50 лет, боль в груди",
        limit=60,
    )

    assert caption == "Журнал анализа\nКонтекст: возраст 50 лет, боль в груди"
    assert len(format_admin_upload_caption("Журнал", "я" * 100, limit=40)) == 40


@pytest.mark.asyncio
async def test_metadata_audit_does_not_send_ecg_context_or_answer() -> None:
    bot = AsyncMock()
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=frozenset({100}),
        admin_audit_include_content=False,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(
            id=200,
            first_name="Иван",
            last_name=None,
            username="ivan",
        ),
        caption="секретный контекст",
    )

    await router._send_admin_audit(
        bot,
        message,
        received_at=datetime(2026, 8, 27, tzinfo=UTC),
        model="gpt-5.6-sol",
        outcome="Готово",
        usages=[],
        balance_before=None,
        balance_after=None,
        raw_image=b"private-ecg",
        original_filename="private.png",
        answer_parts=["секретный ответ"],
        answer_parse_mode=None,
        report_document=b"private-pdf",
        report_filename="private.pdf",
    )

    assert bot.send_message.await_count == 1
    sent_summary = bot.send_message.await_args.args[1]
    assert "Пользователь: Иван (@ivan)" in sent_summary
    assert "секретный контекст" not in sent_summary
    assert "секретный ответ" not in sent_summary
    bot.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_audit_sends_pdf_with_journal_in_one_message() -> None:
    bot = AsyncMock()
    router = object.__new__(ECGTelegramRouter)
    router.settings = SimpleNamespace(
        moderator_user_ids=frozenset({100}),
        admin_audit_include_content=True,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(
            id=200,
            first_name="Иван",
            last_name=None,
            username="ivan",
        ),
        caption="контекст пользователя",
    )

    await router._send_admin_audit(
        bot,
        message,
        received_at=datetime(2026, 8, 27, tzinfo=UTC),
        model="gpt-5.6-sol",
        outcome="Готово",
        usages=[],
        balance_before=None,
        balance_after=None,
        raw_image=b"ecg-bytes",
        original_filename="folder\\ecg.png",
        answer_parts=["точный ответ"],
        answer_parse_mode=None,
        report_document=b"%PDF-report",
        report_filename="miocard-report.pdf",
    )

    assert bot.send_document.await_count == 1
    report_call = bot.send_document.await_args
    attached_report = report_call.args[1]
    assert attached_report.data == b"%PDF-report"
    assert attached_report.filename == "miocard-report.pdf"
    assert "Журнал анализа ЭКГ" in report_call.kwargs["caption"]
    assert "Контекст: контекст пользователя" in report_call.kwargs["caption"]
    bot.send_message.assert_not_awaited()
    bot.send_photo.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_gets_pdf_and_marked_image_linked_to_result() -> None:
    message = SimpleNamespace(
        answer_document=AsyncMock(),
        answer_photo=AsyncMock(),
    )
    status = SimpleNamespace(message_id=321)

    await ECGTelegramRouter._send_user_artifacts(
        message,
        status,
        report_document=b"%PDF-report",
        report_filename="miocard-report.pdf",
        annotated_image=b"marked-png",
        annotation_caption=(
            "Отметки на изображении:\n\n"
            "[1] Преждевременный комплекс\nРитм: единичная экстрасистола"
        ),
    )

    pdf_call = message.answer_document.await_args
    assert pdf_call.args[0].data == b"%PDF-report"
    assert pdf_call.kwargs["reply_parameters"].message_id == 321
    image_call = message.answer_photo.await_args
    assert image_call.args[0].data == b"marked-png"
    assert image_call.kwargs["reply_parameters"].message_id == 321
    assert "[1] Преждевременный комплекс" in image_call.kwargs["caption"]
    assert "Ритм: единичная экстрасистола" in image_call.kwargs["caption"]


def test_only_moderators_pass_is_moderator() -> None:
    moderators = frozenset({153241307})
    assert is_moderator(153241307, moderators)
    assert not is_moderator(2053404449, moderators)


@pytest.mark.asyncio
async def test_allowed_interaction_is_recorded_as_a_known_user(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset(), allowed_user_ids=frozenset({100}))
    message = make_message(100)

    assert router._is_allowed(message) is True
    assert router.support.list_known_user_ids() == [100]


@pytest.mark.asyncio
async def test_announce_rejects_regular_user_without_broadcasting(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}), allowed_user_ids=frozenset({100}))
    bot = AsyncMock()
    message = make_message(100, "/announce привет")
    command = SimpleNamespace(args="привет")

    await router.announce_command(message, command, bot)

    assert "только модераторам" in message.answer.await_args.args[0]
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_announce_without_text_shows_usage(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}))
    bot = AsyncMock()
    message = make_message(900, "/announce")
    command = SimpleNamespace(args=None)

    await router.announce_command(message, command, bot)

    assert "Использование: /announce" in message.answer.await_args.args[0]
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_announce_broadcasts_to_known_users_excluding_sender(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}), allowed_user_ids=frozenset())
    # Simulate prior interactions that populated the known-user list.
    router.support.record_user_seen(100)
    router.support.record_user_seen(200)
    router.support.record_user_seen(900)
    bot = AsyncMock()
    message = make_message(900, "/announce Плановые работы завтра")
    command = SimpleNamespace(args="Плановые работы завтра")

    await router.announce_command(message, command, bot)

    sent_ids = sorted(item.args[0] for item in bot.send_message.await_args_list)
    assert sent_ids == [100, 200]
    for item in bot.send_message.await_args_list:
        assert item.args[1] == "📢 Плановые работы завтра"
    assert "отправлена 2 из 2" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_announce_reports_delivery_failures(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}))
    router.support.record_user_seen(100)
    router.support.record_user_seen(200)
    bot = AsyncMock()

    async def send_message(target_id: int, _text: str) -> None:
        if target_id == 200:
            raise RuntimeError("blocked the bot")

    bot.send_message.side_effect = send_message
    message = make_message(900, "/announce привет")
    command = SimpleNamespace(args="привет")

    await router.announce_command(message, command, bot)

    assert "отправлена 1 из 2" in message.answer.await_args.args[0]
    assert "Не удалось доставить 1" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_announce_with_no_known_users_does_not_error(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}))
    bot = AsyncMock()
    message = make_message(900, "/announce привет")
    command = SimpleNamespace(args="привет")

    await router.announce_command(message, command, bot)

    assert "нет известных пользователей" in message.answer.await_args.args[0]
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_rejects_regular_user(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}), allowed_user_ids=frozenset({100}))
    bot = AsyncMock()
    message = make_message(100)

    await router.shutdown_command(message, bot)

    assert "только модераторам" in message.answer.await_args.args[0]
    bot.send_message.assert_not_awaited()
    assert not router.shutdown_event.is_set()


@pytest.mark.asyncio
async def test_shutdown_notifies_known_users_and_sets_the_event(tmp_path) -> None:
    router = make_router(tmp_path, moderator_user_ids=frozenset({900}))
    router.support.record_user_seen(100)
    router.support.record_user_seen(200)
    bot = AsyncMock()
    message = make_message(900)

    await router.shutdown_command(message, bot)

    sent_ids = sorted(item.args[0] for item in bot.send_message.await_args_list)
    assert sent_ids == [100, 200]
    for item in bot.send_message.await_args_list:
        assert "временно остановлен" in item.args[1]
    assert router.shutdown_event.is_set()
