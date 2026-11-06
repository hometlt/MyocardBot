from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import defaultdict

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from miocard.anthropic_client import (
    AnthropicClient,
    AnthropicServiceError,
    OutputLimitExceededError,
    TokenBalance,
    UsageStats,
)
from miocard.config import Settings
from miocard.formatting import CONFIDENCE_LEGEND, format_report_messages
from miocard.imaging import ImageProcessingError, annotate_findings, process_ecg_image
from miocard.overlay import annotate_digitisation
from miocard.pipeline import to_json

logger = logging.getLogger(__name__)

MASHA_USER_ID = 2053404449
PROGRESS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
PROGRESS_UPDATE_INTERVAL_SECONDS = 1.0
MODEL_CHOICES = (
    ("Sol", "gpt-5.6-sol"),
    ("Terra", "gpt-5.6-terra"),
    ("Opus", "claude-opus-4-8"),
    ("Sonnet", "claude-sonnet-5"),
    ("Flash", "gemini-3.5-flash"),
    ("Fable", "claude-fable-5"),
)
MODEL_LABEL_BY_ID = {model_id: label for label, model_id in MODEL_CHOICES}
MODEL_CALLBACK_PREFIX = "model:"

RECORD_CAPTION = """Цифровой код записи.

Сетка (px/мм), найденные отведения и их положение на снимке, отсчёты каждого отведения в мВ, усреднённый комплекс, отметки R-зубцов, распознанные текстовые объекты и оценка качества с перечнем ограничений.

Это результат оцифровки фотографии, а не выгрузка из аппарата: доверять значениям стоит ровно настолько, насколько зелёная линия на размеченном снимке легла на напечатанную кривую."""

DIGITISATION_CAPTION = """Как бот прочитал снимок.

Зелёным показано положение пера, измеренное по отпечатку; оранжевым — участки, где читаемых чернил не нашлось и линия достроена; синим — комплексы, посчитанные за сердечные сокращения. Слева от каждой линии — отведение, к которому бот её отнёс.

Это проверка распознавания, а не заключение: если зелёная линия не лежит на напечатанной кривой или отведение подписано неверно, измерения в отчёте по этой строке читать не нужно."""

WELCOME = f"""Отправьте фотографию или PNG/JPEG/WebP-файл с 12-канальной ЭКГ.

Можно добавить в подписи краткий контекст без ФИО и других прямых идентификаторов: возраст, пол, симптомы и препараты. Полезно указать данные прошлых осмотров и выводы предыдущих ЭКГ — тогда бот сможет отметить динамику. Бот выполнит предварительный разбор для врача, но не поставит окончательный диагноз.

К отчёту бот прикладывает снимок с нанесённой оцифровкой: какие отведения он нашёл и где именно прочитал линию. По нему видно, каким строкам отчёта можно верить. Рядом приходит файл с цифровым кодом записи — отсчёты каждого отведения в мВ и всё измеренное по снимку.

Если модель найдёт локализуемое отклонение, бот пришлёт размеченную копию снимка. Метки [1], [2], [3] соответствуют сноскам в отчёте.

{CONFIDENCE_LEGEND}

Не отправляйте данные без законного основания и согласия пациента. При острых симптомах не ждите ответа бота — нужна неотложная очная помощь."""

HELP = f"""Пришлите фотографию ЭКГ или изображение как файл PNG/JPEG/WebP. Лучше отправлять оригинал как файл, чтобы избежать сжатия.

В подписи можно добавить обезличенный контекст: возраст, пол, симптомы и препараты. Если известны прошлые осмотры и заключения предыдущих ЭКГ, коротко укажите их — бот сопоставит текущий снимок с описанной динамикой. Не добавляйте ФИО и другие прямые идентификаторы.

Вместе с отчётом приходит снимок с нанесённой оцифровкой: зелёным — прочитанная линия, оранжевым — достроенные участки, синим — посчитанные комплексы. Если линия не легла на напечатанную кривую, измерения по этому отведению недостоверны.

Файлом `ecg-record.json` приходит цифровой код записи: сетка, отведения и их положение на снимке, отсчёты в мВ, усреднённый комплекс, отметки R-зубцов, текстовые объекты и оценка качества.

Метки [1], [2], [3] в отчёте соответствуют пронумерованным рамкам на размеченной копии снимка.

{CONFIDENCE_LEGEND}"""


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
    output_tokens = sum(_token_count(item.get("output_tokens")) for item in usages)
    label = MODEL_LABEL_BY_ID.get(model, model)
    lines = ["📊 Использование API", f"Модель: {label}"]
    if usages:
        lines.append(
            "Исходящие токены (ответ модели): "
            + _format_token_count(output_tokens)
        )
    else:
        lines.append("API-вызов не выполнялся: ответ получен из кэша.")

    if balance_after is not None:
        if balance_before is not None:
            spent = max(0, balance_after.used - balance_before.used)
            lines.append(
                "Списано с пакета за время анализа: "
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


class ECGTelegramRouter:
    def __init__(self, settings: Settings, anthropic: AnthropicClient) -> None:
        self.settings = settings
        self.anthropic = anthropic
        self.router = Router(name="ecg")
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_analyses)
        self._metering_lock = asyncio.Lock()
        self._last_request: dict[int, float] = defaultdict(float)
        self._active_model = settings.anthropic_model
        self.router.message.register(self.start, CommandStart())
        self.router.message.register(self.help, Command("help"))
        self.router.message.register(self.model_command, Command("model"))
        self.router.message.register(self.usage_command, Command("usage"))
        self.router.callback_query.register(
            self.select_model,
            F.data.startswith(MODEL_CALLBACK_PREFIX),
        )
        self.router.message.register(self.handle_photo, F.photo)
        self.router.message.register(
            self.handle_document,
            F.document.mime_type.in_({"image/jpeg", "image/png", "image/webp", "image/gif"}),
        )
        self.router.message.register(self.unsupported)

    def _is_allowed(self, message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else 0
        return self._user_is_allowed(user_id)

    def _user_is_allowed(self, user_id: int) -> bool:
        return user_has_access(
            user_id,
            self.settings.allowed_user_ids,
            self.settings.moderator_user_ids,
        )

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
        await message.answer(HELP + moderator_help)

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

    async def handle_photo(self, message: Message, bot: Bot) -> None:
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > self.settings.max_image_bytes:
            await message.answer("Файл слишком большой.")
            return
        await self._handle_image(message, bot, photo.file_id)

    async def handle_document(self, message: Message, bot: Bot) -> None:
        document = message.document
        if document is None:
            return
        if document.file_size and document.file_size > self.settings.max_image_bytes:
            await message.answer("Файл слишком большой.")
            return
        await self._handle_image(message, bot, document.file_id)

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

    async def _send_usage_to_moderators(
        self,
        bot: Bot,
        text: str | None,
    ) -> None:
        if not text:
            return
        for moderator_id in sorted(self.settings.moderator_user_ids):
            try:
                await bot.send_message(moderator_id, text)
            except Exception:
                logger.exception(
                    "Could not send API usage to moderator %s",
                    moderator_id,
                )

    async def _handle_image(self, message: Message, bot: Bot, file_id: str) -> None:
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

        status = await message.answer(analysis_status_text(0))
        animation = asyncio.create_task(self._animate_analysis(status))
        final_parts: list[str]
        final_parse_mode: ParseMode | None = None
        annotated_image: bytes | None = None
        digitised_image: bytes | None = None
        record: bytes | None = None
        annotation_count = 0
        usage_notification: str | None = None
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
            try:
                digitised_image = await asyncio.to_thread(annotate_digitisation, processed)
            except Exception:
                # The reading still stands even if it cannot be drawn.
                logger.exception("Could not draw the digitisation overlay")
            if processed.pipeline is not None and processed.pipeline.lead_signals:
                try:
                    record = (
                        await asyncio.to_thread(to_json, processed.pipeline, 1)
                    ).encode("utf-8")
                except Exception:
                    logger.exception("Could not package the digitised record")
            async with self._semaphore, self._metering_lock:
                balance_before = await self._safe_token_balance()
                usage_start = len(self.anthropic.response_usage)
                try:
                    report = await self.anthropic.analyze(
                        processed,
                        message.caption,
                        model=selected_model,
                    )
                finally:
                    request_usages = self.anthropic.response_usage[usage_start:]
                    balance_after = await self._safe_token_balance()
                    usage_notification = format_usage_notification(
                        selected_model,
                        request_usages,
                        balance_before,
                        balance_after,
                    )
            final_parts = format_report_messages(report)
            final_parse_mode = ParseMode.HTML
            if report.localized_findings:
                try:
                    annotated_image = await asyncio.to_thread(
                        annotate_findings,
                        processed,
                        report.localized_findings,
                    )
                    annotation_count = len(report.localized_findings)
                except Exception:
                    # A failed overlay must not discard an otherwise valid report.
                    logger.exception("Could not draw findings on the ECG image")
                    annotated_image = None
        except ImageProcessingError as exc:
            final_parts = [f"Не удалось обработать снимок: {exc}"]
        except OutputLimitExceededError as exc:
            logger.warning("API output limit exhausted: %s", exc)
            final_parts = [str(exc)]
        except AnthropicServiceError as exc:
            logger.warning("Anthropic analysis failed: %s", exc)
            final_parts = [
                "Сервис анализа временно недоступен. Попробуйте позже или передайте снимок врачу "
                "обычным способом. Техническая причина: " + str(exc)
            ]
        except Exception as exc:
            logger.exception("Unexpected ECG processing error")
            detail = f"{type(exc).__name__}: {exc}".strip().rstrip(":")
            final_parts = [
                "Произошла внутренняя ошибка. Попробуйте еще раз позже.\n"
                f"Техническая причина: {detail}"
            ]
        finally:
            animation.cancel()
            await asyncio.gather(animation, return_exceptions=True)

        await status.edit_text(final_parts[0], parse_mode=final_parse_mode)
        for continuation in final_parts[1:]:
            await message.answer(continuation, parse_mode=final_parse_mode)
        if digitised_image is not None:
            try:
                await message.answer_photo(
                    BufferedInputFile(digitised_image, filename="ecg-digitised.png"),
                    caption=DIGITISATION_CAPTION,
                )
            except Exception:
                logger.exception("Could not send the digitisation overlay")
        if record is not None:
            try:
                await message.answer_document(
                    BufferedInputFile(record, filename="ecg-record.json"),
                    caption=RECORD_CAPTION,
                )
            except Exception:
                logger.exception("Could not send the digitised record")
        if annotated_image is not None:
            try:
                marker_range = (
                    "[1]" if annotation_count == 1 else f"[1]–[{annotation_count}]"
                )
                await message.answer_photo(
                    BufferedInputFile(annotated_image, filename="ecg-marked.png"),
                    caption=(
                        f"Метки {marker_range} соответствуют разделу "
                        "«Отметки на снимке»."
                    ),
                )
            except Exception:
                logger.exception("Could not send annotated ECG image")
        await self._send_usage_to_moderators(bot, usage_notification)

    async def unsupported(self, message: Message) -> None:
        if self._is_allowed(message):
            await message.answer("Пришлите фотографию ЭКГ или изображение как файл (PNG/JPEG/WebP).")
