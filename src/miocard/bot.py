from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import defaultdict
from datetime import datetime

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart, Filter
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyParameters,
)

from miocard.anthropic_client import (
    AnthropicClient,
    AnthropicServiceError,
    OutputLimitExceededError,
    TokenBalance,
    UsageStats,
)
from miocard.config import Settings
from miocard.formatting import (
    CONFIDENCE_LEGEND,
    format_report_messages,
    split_html_message,
)
from miocard.imaging import ImageProcessingError, annotate_findings, process_ecg_image
from miocard.release import (
    CURRENT_RELEASE,
    ReleaseNotificationStore,
    format_release_history,
    format_startup_notification,
)
from miocard.report_pdf import build_ecg_report_pdf, format_annotation_caption
from miocard.support import AnalysisCase, PendingSupport, SupportRoute, SupportStore

logger = logging.getLogger(__name__)

MASHA_USER_ID = 2053404449
PROGRESS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
PROGRESS_UPDATE_INTERVAL_SECONDS = 1.0
MODEL_CHOICES = (
    ("Sol", "gpt-5.6-sol"),
    ("Terra", "gpt-5.6-terra"),
    ("Opus", "claude-opus-5"),
    ("Opus 4.8", "claude-opus-4-8"),
    ("Sonnet", "claude-sonnet-5"),
    ("Flash", "gemini-3.5-flash"),
    ("Fable", "claude-fable-5"),
)
MODEL_LABEL_BY_ID = {model_id: label for label, model_id in MODEL_CHOICES}
MODEL_CALLBACK_PREFIX = "model:"
SUPPORT_CALLBACK_PREFIX = "support:"
SUPPORT_CATEGORY_LABELS = {
    "technical": "Ошибка анализа",
    "result": "Неверный результат",
    "other": "Другая причина",
}

WELCOME = f"""Отправьте фотографию или PNG/JPEG/WebP-файл с 12-канальной ЭКГ.

Можно добавить в подписи краткий контекст без ФИО и других прямых идентификаторов: возраст, пол, симптомы и препараты. Полезно указать данные прошлых осмотров и выводы предыдущих ЭКГ — тогда бот сможет отметить динамику. Бот выполнит предварительный разбор для врача, но не поставит окончательный диагноз.

После анализа бот пришлёт PDF-отчёт со строгим оформлением и копией ЭКГ. Если модель найдёт локализуемое отклонение, в PDF появится размеченный снимок. Метки [1], [2], [3] соответствуют сноскам в отчёте.

{CONFIDENCE_LEGEND}

Не отправляйте данные без законного основания и согласия пациента. При острых симптомах не ждите ответа бота — нужна неотложная очная помощь."""

ONLINE_NOTIFICATION = format_startup_notification()
RELEASE_HISTORY_TEXT = format_release_history()

HELP = f"""Пришлите фотографию ЭКГ или изображение как файл PNG/JPEG/WebP. Лучше отправлять оригинал как файл, чтобы избежать сжатия.

В подписи можно добавить обезличенный контекст: возраст, пол, симптомы и препараты. Если известны прошлые осмотры и заключения предыдущих ЭКГ, коротко укажите их — бот сопоставит текущий снимок с описанной динамикой. Не добавляйте ФИО и другие прямые идентификаторы.

PDF-отчёт содержит полный структурированный разбор и копию ЭКГ. Метки [1], [2], [3] в отчёте соответствуют пронумерованным рамкам на размеченном снимке.

{CONFIDENCE_LEGEND}"""


def analysis_support_keyboard(case_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⚠️ Ошибка анализа",
                    callback_data=f"{SUPPORT_CALLBACK_PREFIX}technical:{case_id}",
                ),
                InlineKeyboardButton(
                    text="🩺 Неверный результат",
                    callback_data=f"{SUPPORT_CALLBACK_PREFIX}result:{case_id}",
                ),
            ]
        ]
    )


def support_menu_keyboard(case_id: str | None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if case_id:
        rows.extend(analysis_support_keyboard(case_id).inline_keyboard)
    rows.append(
        [
            InlineKeyboardButton(
                text="💬 Другая причина",
                callback_data=f"{SUPPORT_CALLBACK_PREFIX}other",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


class SupportMessageFilter(Filter):
    def __init__(self, store: SupportStore, moderator_user_ids: frozenset[int]) -> None:
        self.store = store
        self.moderator_user_ids = moderator_user_ids

    async def __call__(self, message: Message) -> bool:
        user = message.from_user
        if user is None:
            return False
        replied = message.reply_to_message
        if replied is not None:
            route = self.store.get_route(message.chat.id, replied.message_id)
            if route is not None:
                return True
        return (
            user.id not in self.moderator_user_ids
            and self.store.get_pending(user.id) is not None
        )


def greeting_for_user(
    user_id: int,
    first_name: str | None,
    username: str | None,
    moderator_user_ids: frozenset[int],
) -> str:
    if user_id in moderator_user_ids:
        return "Здравствуй, Модератор"
    if user_id == MASHA_USER_ID:
        return "Привет, Маша ❤️"

    profile_name = first_name.strip() if first_name else ""
    if profile_name:
        return f"Здравствуйте, {profile_name}"
    if username and username.strip().lstrip("@"):
        return f"Здравствуйте, @{username.strip().lstrip('@')}"
    return "Здравствуйте"


def can_select_model(user_id: int, moderator_user_ids: frozenset[int]) -> bool:
    return user_id in moderator_user_ids


def is_moderator(user_id: int, moderator_user_ids: frozenset[int]) -> bool:
    """Gate for admin-only actions that are not about picking the analysis model."""
    return user_id in moderator_user_ids


def can_view_prompt(user_id: int, moderator_user_ids: frozenset[int]) -> bool:
    return user_id == MASHA_USER_ID or user_id in moderator_user_ids


def user_has_access(
    user_id: int,
    allowed_user_ids: frozenset[int],
    moderator_user_ids: frozenset[int],
) -> bool:
    return (
        user_id in moderator_user_ids
        or not allowed_user_ids
        or user_id in allowed_user_ids
    )


def model_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=label,
            callback_data=MODEL_CALLBACK_PREFIX + model_id,
        )
        for label, model_id in MODEL_CHOICES
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[buttons[:2], buttons[2:4], buttons[4:]],
    )


def analysis_status_text(
    elapsed_seconds: float,
    frame_index: int = 0,
) -> str:
    elapsed = max(0, int(elapsed_seconds))
    if elapsed < 8:
        stage = "Подготавливаю изображение"
    elif elapsed < 35:
        stage = "Проверяю отведения и калибровку"
    elif elapsed < 75:
        stage = "Выполняю структурированный разбор"
    else:
        stage = "Формирую итоговый отчёт"

    minutes, seconds = divmod(elapsed, 60)
    timing = (
        "Обычно анализ занимает около 1–2 минут"
        if elapsed < 120
        else "Ответ занимает немного больше обычного"
    )
    frame = PROGRESS_FRAMES[frame_index % len(PROGRESS_FRAMES)]
    return f"{frame} {stage}…\nПрошло {minutes}:{seconds:02d}. {timing}."


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _format_token_count(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def format_usage_notification(
    model: str,
    usages: list[dict[str, object]],
    balance_before: TokenBalance | None,
    balance_after: TokenBalance | None,
) -> str:
    input_tokens = sum(
        _token_count(item.get("input_tokens", item.get("prompt_tokens")))
        for item in usages
    )
    output_tokens = sum(_token_count(item.get("output_tokens")) for item in usages)
    label = MODEL_LABEL_BY_ID.get(model, model)
    lines = ["📊 Использование API", f"Модель: {label}"]
    if usages:
        lines.extend(
            [
                "API-запросов: " + _format_token_count(len(usages)),
                "Входящие токены (инструкции и изображение): "
                + _format_token_count(input_tokens),
                "Исходящие токены (ответ модели): "
                + _format_token_count(output_tokens),
            ]
        )
    else:
        lines.append("API-вызов не выполнялся: ответ получен из кэша.")

    if balance_after is not None:
        if balance_before is not None:
            spent = max(0, balance_after.used - balance_before.used)
            lines.append(
                "Списано с общего пакета за время анализа: "
                + _format_token_count(spent)
            )
        lines.extend(
            [
                "Использовано в пакете: "
                f"{_format_token_count(balance_after.used)} / "
                f"{_format_token_count(balance_after.limit)}",
                "Осталось токенов: " + _format_token_count(balance_after.remaining),
            ]
        )
        lines.append("Баланс пакета общий для всех запросов с этим API-ключом.")
    else:
        lines.append("Остаток пакета получить не удалось.")
    return "\n".join(lines)


def format_usage_stats(
    active_model: str,
    stats: UsageStats,
    balance: TokenBalance | None,
) -> str:
    lines = ["📈 Статистика использования", f"Текущая модель: {MODEL_LABEL_BY_ID.get(active_model, active_model)}"]

    lines.append("")
    lines.append("Пакет токенов (общий счетчик шлюза):")
    if balance is None:
        lines.append(
            "остаток недоступен: шлюз не отвечает или ключ работает напрямую с Anthropic."
        )
    else:
        share = f" ({balance.used * 100 / balance.limit:.1f}%)" if balance.limit else ""
        lines.extend(
            [
                f"Запас: {_format_token_count(balance.limit)}",
                f"Использовано: {_format_token_count(balance.used)}{share}",
                f"Осталось: {_format_token_count(balance.remaining)}",
            ]
        )

    lines.append("")
    lines.append("С момента запуска бота:")
    lines.extend(
        [
            f"Разборов: {stats.analyses} (из кэша: {stats.cache_hits})",
            f"API-запросов: {stats.requests} (с ошибкой: {stats.api_errors})",
            "Входящие токены (запрос, включая изображение): "
            + _format_token_count(stats.input_tokens),
            "Исходящие токены (ответ модели): " + _format_token_count(stats.output_tokens),
        ]
    )
    if stats.cache_read_tokens or stats.cache_write_tokens:
        lines.append(
            "Кэш промпта: чтение "
            + _format_token_count(stats.cache_read_tokens)
            + ", запись "
            + _format_token_count(stats.cache_write_tokens)
        )
    lines.append("Всего токенов: " + _format_token_count(stats.total_tokens))
    if stats.analyses:
        lines.append(
            "В среднем на разбор: "
            + _format_token_count(round(stats.total_tokens / stats.analyses))
        )

    if stats.by_model:
        lines.append("")
        lines.append("По моделям (запросы · вход · выход):")
        for model, usage in sorted(
            stats.by_model.items(),
            key=lambda item: item[1].total_tokens,
            reverse=True,
        ):
            lines.append(
                f"{MODEL_LABEL_BY_ID.get(model, model)}: {usage.requests} · "
                f"{_format_token_count(usage.input_tokens)} · "
                f"{_format_token_count(usage.output_tokens)}"
            )
    return "\n".join(lines)


def format_admin_audit_summary(
    *,
    user_id: int,
    first_name: str | None,
    last_name: str | None,
    username: str | None,
    received_at: datetime,
    model: str,
    outcome: str,
    usages: list[dict[str, object]],
    balance_before: TokenBalance | None,
    balance_after: TokenBalance | None,
    include_content: bool,
    cache_hit: bool,
) -> str:
    """Build one plain-text audit card without exposing request content."""
    name_parts = (
        " ".join((first_name or "").split()),
        " ".join((last_name or "").split()),
    )
    clean_name = " ".join(part for part in name_parts if part)
    clean_username = "".join((username or "").split()).lstrip("@")
    identity = clean_name or "имя не указано"
    if clean_username:
        identity += f" (@{clean_username})"

    local_time = received_at.astimezone()
    numeric_offset = local_time.strftime("%z")
    utc_label = (
        f"UTC{numeric_offset[:3]}:{numeric_offset[3:]}"
        if len(numeric_offset) == 5
        else "UTC"
    )
    input_tokens = sum(
        _token_count(item.get("input_tokens", item.get("prompt_tokens")))
        for item in usages
    )
    output_tokens = sum(
        _token_count(item.get("output_tokens", item.get("completion_tokens")))
        for item in usages
    )
    lines = [
        "🧪 Журнал анализа ЭКГ",
        f"Пользователь: {identity}",
        f"Telegram ID: {user_id}",
        f"Время: {local_time:%d.%m.%Y %H:%M:%S} {utc_label}",
        f"Модель: {MODEL_LABEL_BY_ID.get(model, model)}",
        f"Результат: {outcome}",
        f"API-запросов: {len(usages)}",
        "Входящие токены: " + _format_token_count(input_tokens),
        "Исходящие токены: " + _format_token_count(output_tokens),
    ]
    if cache_hit:
        lines.append("Источник ответа: кэш готовых разборов (расход API: 0).")
    if balance_before is not None and balance_after is not None:
        spent = max(0, balance_after.used - balance_before.used)
        lines.append("Списано с общего пакета: " + _format_token_count(spent))
        lines.append("Осталось в пакете: " + _format_token_count(balance_after.remaining))
    else:
        lines.append("Расход общего пакета: недоступен.")
    lines.append(
        "Содержимое: ЭКГ и ответ приложены."
        if include_content
        else "Содержимое: отключено настройкой приватности."
    )
    return "\n".join(lines)


def format_admin_upload_caption(
    summary: str,
    user_context: str | None,
    limit: int = 950,
) -> str:
    caption = summary
    context = " ".join((user_context or "").split())
    if context:
        caption += "\nКонтекст: " + context
    if len(caption) <= limit:
        return caption
    return caption[: limit - 1].rstrip() + "…"


async def notify_online(
    bot: Bot,
    settings: Settings,
    store: ReleaseNotificationStore | None = None,
) -> None:
    notification_store = store or ReleaseNotificationStore(
        settings.release_notification_state_file
    )
    user_ids = settings.moderator_user_ids | settings.allowed_user_ids
    for user_id in sorted(user_ids):
        if notification_store.was_notified(user_id, CURRENT_RELEASE.version):
            continue
        try:
            await bot.send_message(user_id, ONLINE_NOTIFICATION)
        except Exception:
            logger.exception("Could not send online notification to %s", user_id)
        else:
            try:
                notification_store.mark_notified(user_id, CURRENT_RELEASE.version)
            except OSError:
                logger.exception(
                    "Could not persist release notification for %s",
                    user_id,
                )


class ECGTelegramRouter:
    def __init__(self, settings: Settings, anthropic: AnthropicClient) -> None:
        self.settings = settings
        self.anthropic = anthropic
        self.router = Router(name="ecg")
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_analyses)
        self._metering_lock = asyncio.Lock()
        self._last_request: dict[int, float] = defaultdict(float)
        self._active_model = settings.anthropic_model
        self.support = SupportStore(settings.support_db_file)
        # Set by /shutdown once every user has been notified; main() waits on this
        # alongside polling to stop the process gracefully.
        self.shutdown_event = asyncio.Event()
        self.router.message.register(self.start, CommandStart())
        self.router.message.register(self.help, Command("help"))
        self.router.message.register(self.updates_command, Command("updates"))
        self.router.message.register(self.support_command, Command("support"))
        self.router.message.register(self.cancel_support, Command("cancel"))
        self.router.message.register(self.prompt_command, Command("prompt"))
        self.router.message.register(self.model_command, Command("model"))
        self.router.message.register(self.usage_command, Command("usage"))
        self.router.message.register(self.announce_command, Command("announce"))
        self.router.message.register(self.shutdown_command, Command("shutdown"))
        self.router.callback_query.register(
            self.select_model,
            F.data.startswith(MODEL_CALLBACK_PREFIX),
        )
        self.router.callback_query.register(
            self.select_support_category,
            F.data.startswith(SUPPORT_CALLBACK_PREFIX),
        )
        self.router.message.register(
            self.handle_support_message,
            SupportMessageFilter(self.support, settings.moderator_user_ids),
        )
        self.router.message.register(self.handle_photo, F.photo)
        self.router.message.register(
            self.handle_document,
            F.document.mime_type.in_({"image/jpeg", "image/png", "image/webp", "image/gif"}),
        )
        self.router.message.register(self.unsupported)

    def _is_allowed(self, message: Message) -> bool:
        user = message.from_user
        user_id = user.id if user else 0
        allowed = self._user_is_allowed(user_id)
        if allowed and user is not None:
            self._record_user_seen(user_id)
        return allowed

    def _user_is_allowed(self, user_id: int) -> bool:
        return user_has_access(
            user_id,
            self.settings.allowed_user_ids,
            self.settings.moderator_user_ids,
        )

    def _record_user_seen(self, user_id: int) -> None:
        """Track every user the bot has actually talked to.

        `ALLOWED_USER_IDS` is often left empty for open access, so it cannot be
        used as the recipient list for /announce and /shutdown by itself.
        """
        try:
            self.support.record_user_seen(user_id)
        except Exception:
            logger.exception("Could not record known user %s", user_id)

    def _broadcast_targets(self, *, exclude: int | None = None) -> list[int]:
        target_ids = set(self.settings.moderator_user_ids) | set(
            self.settings.allowed_user_ids
        )
        try:
            target_ids |= set(self.support.list_known_user_ids())
        except Exception:
            logger.exception("Could not read known users for broadcast")
        if exclude is not None:
            target_ids.discard(exclude)
        return sorted(target_ids)

    @staticmethod
    async def _broadcast(bot: Bot, user_ids: list[int], text: str) -> tuple[int, int]:
        sent = 0
        for target_id in user_ids:
            try:
                await bot.send_message(target_id, text)
                sent += 1
            except Exception:
                logger.exception("Could not deliver broadcast message to %s", target_id)
        return sent, len(user_ids) - sent

    def _selected_model(self) -> str:
        return self._active_model

    @staticmethod
    def _model_selection_text() -> str:
        return (
            "Выберите модель анализа для всех пользователей. "
            "Изменение применяется к следующим запросам и действует до перезапуска бота."
        )

    async def start(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user = message.from_user
        greeting = greeting_for_user(
            user.id if user else 0,
            user.first_name if user else None,
            user.username if user else None,
            self.settings.moderator_user_ids,
        )
        await message.answer(f"{greeting}\n\n{WELCOME}")

    async def help(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        moderator_help = (
            "\n\n/model — выбрать модель анализа для всех пользователей."
            "\n/usage — остаток пакета и статистика использования токенов."
            if can_select_model(user_id, self.settings.moderator_user_ids)
            else ""
        )
        prompt_help = (
            "\n/prompt — скачать текущий рабочий промпт анализа."
            if can_view_prompt(user_id, self.settings.moderator_user_ids)
            else ""
        )
        admin_help = (
            "\n/announce — разослать сообщение всем пользователям бота."
            "\n/shutdown — остановить бота, уведомив всех пользователей."
            if is_moderator(user_id, self.settings.moderator_user_ids)
            else ""
        )
        await message.answer(
            HELP
            + "\n\n/support — обратиться к модераторам."
            + "\n\n/updates — посмотреть историю версий бота."
            + moderator_help
            + prompt_help
            + admin_help
        )

    async def updates_command(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        for part in split_html_message(RELEASE_HISTORY_TEXT):
            await message.answer(part)

    async def support_command(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        if user_id in self.settings.moderator_user_ids:
            await message.answer(
                "Чтобы ответить пользователю, используйте функцию «Ответить» "
                "на карточке или сообщении обращения."
            )
            return
        latest = self.support.latest_analysis_case(user_id)
        await message.answer(
            "Выберите причину обращения. После выбора отправьте или перешлите "
            "одно сообщение с описанием, фотографией либо документом.",
            reply_markup=support_menu_keyboard(latest.case_id if latest else None),
        )

    async def cancel_support(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        pending = self.support.pop_pending(user_id)
        if pending is None:
            await message.answer("Сейчас нет незавершённого обращения.")
        else:
            await message.answer("Создание обращения отменено.")

    async def prompt_command(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        if not can_view_prompt(user_id, self.settings.moderator_user_ids):
            await message.answer("Просмотр промпта доступен только модераторам и Марии.")
            return
        prompt_text = self.anthropic.analysis_system_prompt()
        await message.answer_document(
            BufferedInputFile(
                prompt_text.encode("utf-8"),
                filename="miocard-ai-system-prompt.txt",
            ),
            caption=(
                "Текущий системный промпт основного агента анализа ЭКГ. "
                "Файл сформирован из реально используемой конфигурации без сокращений."
            ),
        )

    async def select_support_category(self, callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        if not self._user_is_allowed(user_id):
            await callback.answer("У вас нет доступа к этому боту.", show_alert=True)
            return
        if user_id in self.settings.moderator_user_ids:
            await callback.answer(
                "Модератор отвечает пользователю через функцию «Ответить».",
                show_alert=True,
            )
            return

        self._record_user_seen(user_id)
        parts = (callback.data or "").split(":", 2)
        if len(parts) < 2 or parts[0] != SUPPORT_CALLBACK_PREFIX.rstrip(":"):
            await callback.answer("Неизвестная категория.", show_alert=True)
            return
        category = parts[1]
        if category not in SUPPORT_CATEGORY_LABELS:
            await callback.answer("Неизвестная категория.", show_alert=True)
            return

        analysis_case_id = parts[2] if len(parts) == 3 and parts[2] else None
        if category in {"technical", "result"}:
            case = (
                self.support.get_analysis_case(analysis_case_id)
                if analysis_case_id
                else None
            )
            if case is None or case.user_id != user_id:
                await callback.answer(
                    "Не удалось найти этот анализ. Откройте /support ещё раз.",
                    show_alert=True,
                )
                return
        else:
            analysis_case_id = None

        self.support.set_pending(user_id, category, analysis_case_id)
        await callback.answer("Категория выбрана")
        if callback.message is not None:
            await callback.message.answer(
                f"Категория: {SUPPORT_CATEGORY_LABELS[category]}.\n"
                "Теперь отправьте или перешлите одно сообщение. "
                "Для отмены используйте /cancel."
            )

    async def model_command(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        if not can_select_model(user_id, self.settings.moderator_user_ids):
            await message.answer("Выбор модели доступен только модераторам.")
            return
        await message.answer(
            self._model_selection_text(),
            reply_markup=model_keyboard(),
        )

    async def usage_command(self, message: Message) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        if not can_select_model(user_id, self.settings.moderator_user_ids):
            await message.answer("Статистика доступна только модераторам.")
            return
        balance = await self._safe_token_balance()
        await message.answer(
            format_usage_stats(
                self._selected_model(),
                self.anthropic.usage_stats,
                balance,
            )
        )

    async def announce_command(
        self,
        message: Message,
        command: CommandObject,
        bot: Bot,
    ) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        if not is_moderator(user_id, self.settings.moderator_user_ids):
            await message.answer("Рассылка сообщений доступна только модераторам.")
            return
        text = (command.args or "").strip()
        if not text:
            await message.answer(
                "Использование: /announce текст сообщения.\n"
                "Сообщение будет отправлено всем известным пользователям бота."
            )
            return
        targets = self._broadcast_targets(exclude=user_id)
        if not targets:
            await message.answer("Пока нет известных пользователей для рассылки.")
            return
        sent, failed = await self._broadcast(bot, targets, f"📢 {text}")
        report = f"Рассылка отправлена {sent} из {len(targets)} пользователям."
        if failed:
            report += f" Не удалось доставить {failed}."
        await message.answer(report)

    async def shutdown_command(self, message: Message, bot: Bot) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        if not is_moderator(user_id, self.settings.moderator_user_ids):
            await message.answer("Остановка бота доступна только модераторам.")
            return
        await message.answer("Уведомляю пользователей и останавливаю бота…")
        targets = self._broadcast_targets(exclude=user_id)
        sent, failed = await self._broadcast(
            bot,
            targets,
            "🔴 Бот «Миокард» временно остановлен модератором и сейчас не отвечает "
            "на сообщения. Пожалуйста, попробуйте позже.",
        )
        logger.warning(
            "Shutdown requested by moderator %s: notified %d/%d users",
            user_id,
            sent,
            len(targets),
        )
        self.shutdown_event.set()

    async def select_model(self, callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        if not self._user_is_allowed(user_id):
            await callback.answer("У вас нет доступа к этому боту.", show_alert=True)
            return
        if not can_select_model(user_id, self.settings.moderator_user_ids):
            await callback.answer(
                "Выбор модели доступен только модераторам.",
                show_alert=True,
            )
            return

        self._record_user_seen(user_id)
        data = callback.data or ""
        selected = data.removeprefix(MODEL_CALLBACK_PREFIX)
        if selected not in MODEL_LABEL_BY_ID:
            await callback.answer("Неизвестная модель.", show_alert=True)
            return

        self._active_model = selected
        label = MODEL_LABEL_BY_ID[selected]
        await callback.answer(f"Выбрана модель: {label}")
        if callback.message:
            await callback.message.edit_text(
                self._model_selection_text(),
                reply_markup=model_keyboard(),
            )

    @staticmethod
    def _support_user_label(message: Message) -> str:
        user = message.from_user
        if user is None:
            return "имя не указано"
        name = " ".join(
            part.strip()
            for part in (user.first_name or "", user.last_name or "")
            if part.strip()
        )
        username = (user.username or "").strip().lstrip("@")
        label = name or "имя не указано"
        if username:
            label += f" (@{username})"
        return label

    @staticmethod
    def _message_id(sent: object) -> int:
        message_id = getattr(sent, "message_id", None)
        if not isinstance(message_id, int):
            raise TypeError("Telegram did not return a message identifier")
        return message_id

    @staticmethod
    async def _forward_support_content(bot: Bot, target_id: int, message: Message) -> int:
        try:
            sent = await bot.forward_message(
                chat_id=target_id,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
            )
        except Exception:
            sent = await bot.copy_message(
                chat_id=target_id,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
            )
        return ECGTelegramRouter._message_id(sent)

    @staticmethod
    async def _copy_support_content(bot: Bot, target_id: int, message: Message) -> int:
        try:
            sent = await bot.copy_message(
                chat_id=target_id,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
            )
        except Exception:
            sent = await bot.forward_message(
                chat_id=target_id,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
            )
        return ECGTelegramRouter._message_id(sent)

    def _support_header(
        self,
        message: Message,
        ticket_id: str,
        pending: PendingSupport,
    ) -> str:
        user_id = message.from_user.id if message.from_user else 0
        lines = [
            f"🆘 Обращение #{ticket_id[:8]}",
            f"Пользователь: {self._support_user_label(message)}",
            f"Telegram ID: {user_id}",
            f"Причина: {SUPPORT_CATEGORY_LABELS.get(pending.category, pending.category)}",
        ]
        if pending.analysis_case_id:
            case = self.support.get_analysis_case(pending.analysis_case_id)
            if case is not None:
                local_time = case.created_at.astimezone()
                lines.extend(
                    [
                        f"Анализ: #{case.case_id[:8]} от {local_time:%d.%m.%Y %H:%M}",
                        f"Результат обработки: {case.outcome}",
                    ]
                )
        lines.append("Ответьте на эту карточку или следующее сообщение пользователя.")
        return "\n".join(lines)

    def _add_support_route(
        self,
        *,
        chat_id: int,
        message_id: int,
        route: SupportRoute,
    ) -> None:
        self.support.add_route(
            chat_id=chat_id,
            message_id=message_id,
            ticket_id=route.ticket_id,
            user_id=route.user_id,
            moderator_id=route.moderator_id,
        )

    async def _open_support_ticket(
        self,
        message: Message,
        bot: Bot,
        pending: PendingSupport,
    ) -> None:
        user_id = message.from_user.id if message.from_user else 0
        if not self.settings.moderator_user_ids:
            await message.answer("Сейчас ни один модератор не подключён. Попробуйте позже.")
            return

        ticket_id = self.support.create_ticket(
            user_id,
            pending.category,
            pending.analysis_case_id,
        )
        header = self._support_header(message, ticket_id, pending)
        delivered = 0
        for moderator_id in sorted(self.settings.moderator_user_ids):
            route = SupportRoute(ticket_id, user_id, moderator_id)
            try:
                header_message = await bot.send_message(moderator_id, header)
                header_id = self._message_id(header_message)
                self._add_support_route(
                    chat_id=moderator_id,
                    message_id=header_id,
                    route=route,
                )
                content_id = await self._forward_support_content(bot, moderator_id, message)
                self._add_support_route(
                    chat_id=moderator_id,
                    message_id=content_id,
                    route=route,
                )
                delivered += 1
            except Exception:
                logger.exception(
                    "Could not deliver support ticket %s to moderator %s",
                    ticket_id,
                    moderator_id,
                )

        if delivered:
            self.support.pop_pending(user_id)
            await message.answer(
                f"Обращение #{ticket_id[:8]} передано модераторам. "
                "Ответ придёт в этот чат."
            )
        else:
            await message.answer(
                "Не удалось передать обращение модераторам. Попробуйте отправить его ещё раз позже."
            )

    async def _reply_to_user(
        self,
        message: Message,
        bot: Bot,
        route: SupportRoute,
    ) -> None:
        moderator_id = message.from_user.id if message.from_user else 0
        user_route = SupportRoute(route.ticket_id, route.user_id, moderator_id)
        try:
            header = await bot.send_message(
                route.user_id,
                f"💬 Ответ поддержки по обращению #{route.ticket_id[:8]}. "
                "Чтобы продолжить разговор с этим модератором, ответьте на это сообщение.",
            )
            self._add_support_route(
                chat_id=route.user_id,
                message_id=self._message_id(header),
                route=user_route,
            )
            content_id = await self._copy_support_content(bot, route.user_id, message)
            self._add_support_route(
                chat_id=route.user_id,
                message_id=content_id,
                route=user_route,
            )
            await message.answer("Ответ отправлен пользователю.")
        except Exception:
            logger.exception("Could not reply to support user %s", route.user_id)
            await message.answer("Не удалось отправить ответ пользователю.")

    async def _reply_to_moderator(
        self,
        message: Message,
        bot: Bot,
        route: SupportRoute,
    ) -> None:
        moderator_id = route.moderator_id
        if moderator_id is None:
            await message.answer("Не удалось определить модератора для ответа.")
            return
        moderator_route = SupportRoute(route.ticket_id, route.user_id, moderator_id)
        try:
            header = await bot.send_message(
                moderator_id,
                f"↩️ Ответ пользователя по обращению #{route.ticket_id[:8]}. "
                "Ответьте на эту карточку или следующее сообщение.",
            )
            self._add_support_route(
                chat_id=moderator_id,
                message_id=self._message_id(header),
                route=moderator_route,
            )
            content_id = await self._forward_support_content(bot, moderator_id, message)
            self._add_support_route(
                chat_id=moderator_id,
                message_id=content_id,
                route=moderator_route,
            )
            await message.answer("Ответ отправлен выбранному модератору.")
        except Exception:
            logger.exception("Could not deliver support reply to moderator %s", moderator_id)
            await message.answer("Не удалось отправить ответ модератору.")

    async def handle_support_message(self, message: Message, bot: Bot) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        replied = message.reply_to_message
        route = (
            self.support.get_route(message.chat.id, replied.message_id)
            if replied is not None
            else None
        )
        if route is not None:
            if user_id in self.settings.moderator_user_ids:
                await self._reply_to_user(message, bot, route)
            elif user_id == route.user_id and route.moderator_id is not None:
                await self._reply_to_moderator(message, bot, route)
            else:
                await message.answer("Это обращение принадлежит другому пользователю.")
            return

        if user_id in self.settings.moderator_user_ids:
            await message.answer(
                "Используйте функцию «Ответить» на сообщении конкретного обращения."
            )
            return
        pending = self.support.get_pending(user_id)
        if pending is None:
            await message.answer("Чтобы обратиться к модераторам, используйте /support.")
            return
        await self._open_support_ticket(message, bot, pending)

    async def handle_photo(self, message: Message, bot: Bot) -> None:
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > self.settings.max_image_bytes:
            await message.answer("Файл слишком большой.")
            return
        await self._handle_image(
            message,
            bot,
            photo.file_id,
            f"telegram-photo-{photo.file_unique_id}.jpg",
        )

    async def handle_document(self, message: Message, bot: Bot) -> None:
        document = message.document
        if document is None:
            return
        if document.file_size and document.file_size > self.settings.max_image_bytes:
            await message.answer("Файл слишком большой.")
            return
        await self._handle_image(
            message,
            bot,
            document.file_id,
            document.file_name or f"telegram-document-{document.file_unique_id}",
        )

    @staticmethod
    async def _animate_analysis(status: Message) -> None:
        started = time.monotonic()
        frame_index = 1
        while True:
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL_SECONDS)
            try:
                await status.edit_text(
                    analysis_status_text(time.monotonic() - started, frame_index)
                )
            except TelegramRetryAfter as exc:
                await asyncio.sleep(max(float(exc.retry_after), 1.0))
                continue
            except Exception:
                logger.debug("Could not update analysis animation", exc_info=True)
                return
            frame_index += 1

    async def _safe_token_balance(self) -> TokenBalance | None:
        try:
            return await self.anthropic.get_token_balance()
        except AnthropicServiceError:
            logger.warning("Could not read Tokenator token balance", exc_info=True)
            return None

    async def _send_admin_audit(
        self,
        bot: Bot,
        message: Message,
        *,
        received_at: datetime,
        model: str,
        outcome: str,
        usages: list[dict[str, object]],
        balance_before: TokenBalance | None,
        balance_after: TokenBalance | None,
        raw_image: bytes | None,
        original_filename: str,
        answer_parts: list[str],
        answer_parse_mode: ParseMode | None,
        report_document: bytes | None,
        report_filename: str,
    ) -> None:
        include_content = self.settings.admin_audit_include_content
        user = message.from_user
        user_id = user.id if user else 0
        summary = format_admin_audit_summary(
            user_id=user_id,
            first_name=user.first_name if user else None,
            last_name=user.last_name if user else None,
            username=user.username if user else None,
            received_at=received_at,
            model=model,
            outcome=outcome,
            usages=usages,
            balance_before=balance_before,
            balance_after=balance_after,
            include_content=include_content,
            cache_hit=outcome == "Готово" and not usages,
        )
        safe_filename = (
            original_filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
            or "ecg-original.jpg"
        )[:120]
        for moderator_id in sorted(self.settings.moderator_user_ids):
            try:
                if not include_content:
                    await bot.send_message(moderator_id, summary)
                    continue
                if report_document is not None:
                    await bot.send_document(
                        moderator_id,
                        BufferedInputFile(report_document, filename=report_filename),
                        caption=format_admin_upload_caption(summary, message.caption),
                    )
                    continue
                if raw_image is not None:
                    await bot.send_document(
                        moderator_id,
                        BufferedInputFile(raw_image, filename=safe_filename),
                        caption=format_admin_upload_caption(summary, message.caption),
                    )
                else:
                    await bot.send_message(moderator_id, summary)
                await bot.send_message(moderator_id, "Ответ бота:")
                for part in answer_parts:
                    await bot.send_message(
                        moderator_id,
                        part,
                        parse_mode=answer_parse_mode,
                    )
            except Exception:
                logger.exception(
                    "Could not send ECG audit entry to moderator %s",
                    moderator_id,
                )

    @staticmethod
    async def _send_user_artifacts(
        message: Message,
        status: Message,
        *,
        report_document: bytes | None,
        report_filename: str,
        annotated_image: bytes | None,
        annotation_caption: str | None,
    ) -> None:
        if report_document is not None:
            try:
                await message.answer_document(
                    BufferedInputFile(report_document, filename=report_filename),
                    caption="PDF-отчёт с ЭКГ и полным разбором.",
                    reply_parameters=ReplyParameters(message_id=status.message_id),
                )
            except Exception:
                logger.exception("Could not send ECG PDF report")
        if annotated_image is not None:
            try:
                await message.answer_photo(
                    BufferedInputFile(annotated_image, filename="ecg-marked.png"),
                    caption=annotation_caption or "Размеченная ЭКГ.",
                    reply_parameters=ReplyParameters(message_id=status.message_id),
                )
            except Exception:
                logger.exception("Could not send annotated ECG image")

    async def _handle_image(
        self,
        message: Message,
        bot: Bot,
        file_id: str,
        original_filename: str = "ecg-original.jpg",
    ) -> None:
        if not self._is_allowed(message):
            await message.answer("У вас нет доступа к этому боту.")
            return
        user_id = message.from_user.id if message.from_user else 0
        now = time.monotonic()
        remaining = self.settings.user_cooldown_seconds - (now - self._last_request[user_id])
        if remaining > 0:
            await message.answer(f"Подождите еще {remaining:.0f} сек. перед следующим анализом.")
            return
        self._last_request[user_id] = now
        selected_model = self._selected_model()
        received_at = message.date or datetime.now().astimezone()
        analysis_case: AnalysisCase | None = None
        try:
            analysis_case = self.support.create_analysis_case(user_id, received_at)
        except Exception:
            logger.exception("Could not persist analysis case for support")

        status = await message.answer(analysis_status_text(0))
        animation = asyncio.create_task(self._animate_analysis(status))
        final_parts: list[str]
        final_parse_mode: ParseMode | None = None
        annotated_image: bytes | None = None
        annotation_caption: str | None = None
        report_document: bytes | None = None
        report_filename = "miocard-report.pdf"
        raw: bytes | None = None
        request_usages: list[dict[str, object]] = []
        balance_before: TokenBalance | None = None
        balance_after: TokenBalance | None = None
        outcome = "Готово"
        try:
            await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
            buffer = io.BytesIO()
            await bot.download(file_id, destination=buffer)
            raw = buffer.getvalue()
            processed = await asyncio.to_thread(
                process_ecg_image,
                raw,
                selected_model,
                self.settings.max_image_bytes,
            )
            async with self._semaphore, self._metering_lock:
                balance_before = await self._safe_token_balance()
                usage_start = len(self.anthropic.response_usage)
                try:
                    report = await self.anthropic.analyze_multiscale(
                        processed,
                        message.caption,
                        model=selected_model,
                    )
                finally:
                    request_usages = self.anthropic.response_usage[usage_start:]
                    balance_after = await self._safe_token_balance()
            final_parts = format_report_messages(report)
            final_parse_mode = ParseMode.HTML
            if report.localized_findings:
                try:
                    annotated_image = await asyncio.to_thread(
                        annotate_findings,
                        processed,
                        report.localized_findings,
                    )
                    annotation_caption = format_annotation_caption(report)
                except Exception:
                    # A failed overlay must not discard an otherwise valid report.
                    logger.exception("Could not draw findings on the ECG image")
                    annotated_image = None
            try:
                report_document = await asyncio.to_thread(
                    build_ecg_report_pdf,
                    report,
                    annotated_image or processed.data,
                    created_at=received_at,
                    user_context=message.caption,
                    image_is_annotated=annotated_image is not None,
                )
                local_time = received_at.astimezone()
                report_filename = f"miocard-report-{local_time:%Y%m%d-%H%M%S}.pdf"
            except Exception:
                # PDF is an additional representation; its failure must not discard
                # the normal Telegram report.
                logger.exception("Could not build ECG PDF report")
                report_document = None
        except ImageProcessingError as exc:
            outcome = "Ошибка обработки изображения"
            final_parts = [f"Не удалось обработать снимок: {exc}"]
        except OutputLimitExceededError as exc:
            outcome = "Превышен лимит ответа модели"
            logger.warning("API output limit exhausted: %s", exc)
            final_parts = [str(exc)]
        except AnthropicServiceError as exc:
            outcome = "Ошибка сервиса анализа"
            logger.warning("Anthropic analysis failed: %s", exc)
            final_parts = [
                "Сервис анализа временно недоступен. Попробуйте позже или передайте снимок врачу "
                "обычным способом. Техническая причина: " + str(exc)
            ]
        except Exception as exc:
            outcome = "Внутренняя ошибка"
            logger.exception("Unexpected ECG processing error")
            detail = f"{type(exc).__name__}: {exc}".strip().rstrip(":")
            final_parts = [
                "Произошла внутренняя ошибка. Попробуйте еще раз позже.\n"
                f"Техническая причина: {detail}"
            ]
        finally:
            animation.cancel()
            await asyncio.gather(animation, return_exceptions=True)

        if analysis_case is not None:
            try:
                self.support.update_analysis_outcome(analysis_case.case_id, outcome)
            except Exception:
                logger.exception("Could not update analysis case outcome")
        await status.edit_text(
            final_parts[0],
            parse_mode=final_parse_mode,
            reply_markup=(
                analysis_support_keyboard(analysis_case.case_id)
                if analysis_case is not None
                else None
            ),
        )
        for continuation in final_parts[1:]:
            await message.answer(continuation, parse_mode=final_parse_mode)
        await self._send_user_artifacts(
            message,
            status,
            report_document=report_document,
            report_filename=report_filename,
            annotated_image=annotated_image,
            annotation_caption=annotation_caption,
        )
        await self._send_admin_audit(
            bot,
            message,
            received_at=received_at,
            model=selected_model,
            outcome=outcome,
            usages=request_usages,
            balance_before=balance_before,
            balance_after=balance_after,
            raw_image=raw,
            original_filename=original_filename,
            answer_parts=final_parts,
            answer_parse_mode=final_parse_mode,
            report_document=report_document,
            report_filename=report_filename,
        )

    async def unsupported(self, message: Message) -> None:
        if self._is_allowed(message):
            await message.answer(
                "Пришлите фотографию ЭКГ или изображение как файл (PNG/JPEG/WebP). "
                "Чтобы обратиться к модераторам, используйте /support."
            )
