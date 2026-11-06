from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, BotCommandScopeChat

from miocard.anthropic_client import AnthropicClient
from miocard.bot import MASHA_USER_ID, ECGTelegramRouter, notify_online
from miocard.config import get_settings
from miocard.single_instance import LocalBotInstance


async def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    telegram_token = settings.telegram_bot_token.get_secret_value()
    instance = LocalBotInstance(telegram_token)
    took_over = await instance.acquire_or_take_over()
    if took_over:
        logging.getLogger(__name__).info("Previous local bot instance was stopped")
    try:
        bot = Bot(token=telegram_token)
        anthropic = AnthropicClient(settings)
        dispatcher = Dispatcher()
        ecg_router = ECGTelegramRouter(settings, anthropic)
        dispatcher.include_router(ecg_router.router)
        try:
            await bot.delete_webhook(drop_pending_updates=False)
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description="Начать работу"),
                    BotCommand(command="support", description="Обратиться в поддержку"),
                    BotCommand(command="updates", description="История версий"),
                    BotCommand(command="help", description="Показать справку"),
                ]
            )
            for moderator_id in settings.moderator_user_ids:
                await bot.set_my_commands(
                    [
                        BotCommand(command="start", description="Начать работу"),
                        BotCommand(command="support", description="Инструкция модератору"),
                        BotCommand(command="prompt", description="Текущий промпт анализа"),
                        BotCommand(command="model", description="Выбрать модель анализа"),
                        BotCommand(command="usage", description="Токены и статистика"),
                        BotCommand(command="announce", description="Разослать сообщение всем"),
                        BotCommand(command="shutdown", description="Остановить бота"),
                        BotCommand(command="updates", description="История версий"),
                        BotCommand(command="help", description="Показать справку"),
                    ],
                    scope=BotCommandScopeChat(chat_id=moderator_id),
                )
            if MASHA_USER_ID not in settings.moderator_user_ids:
                await bot.set_my_commands(
                    [
                        BotCommand(command="start", description="Начать работу"),
                        BotCommand(command="support", description="Обратиться в поддержку"),
                        BotCommand(command="prompt", description="Текущий промпт анализа"),
                        BotCommand(command="updates", description="История версий"),
                        BotCommand(command="help", description="Показать справку"),
                    ],
                    scope=BotCommandScopeChat(chat_id=MASHA_USER_ID),
                )
            await notify_online(bot, settings)
            polling = asyncio.create_task(
                dispatcher.start_polling(
                    bot,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    close_bot_session=False,
                )
            )
            takeover = asyncio.create_task(instance.wait_for_takeover())
            shutdown_requested = asyncio.create_task(ecg_router.shutdown_event.wait())
            try:
                completed, _pending = await asyncio.wait(
                    {polling, takeover, shutdown_requested},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if takeover in completed and not polling.done():
                    logging.getLogger(__name__).warning(
                        "A new local bot instance requested takeover"
                    )
                    await dispatcher.stop_polling()
                if shutdown_requested in completed and not polling.done():
                    logging.getLogger(__name__).warning(
                        "Bot stopped by a moderator via /shutdown"
                    )
                    await dispatcher.stop_polling()
                await polling
            finally:
                takeover.cancel()
                shutdown_requested.cancel()
                await asyncio.gather(takeover, shutdown_requested, return_exceptions=True)
                # aiogram stops fetching updates before already-created handler tasks
                # finish. Keep both HTTP clients alive until those analyses drain.
                pending_updates = tuple(dispatcher._handle_update_tasks)
                if pending_updates:
                    logging.getLogger(__name__).info(
                        "Waiting for %d active update(s) before shutdown",
                        len(pending_updates),
                    )
                    await asyncio.gather(*pending_updates, return_exceptions=True)
        finally:
            await anthropic.close()
            await bot.session.close()
    finally:
        instance.release()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
