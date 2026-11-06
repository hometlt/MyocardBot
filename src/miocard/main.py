from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, BotCommandScopeChat

from miocard.anthropic_client import AnthropicClient
from miocard.bot import ECGTelegramRouter
from miocard.config import get_settings


async def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    anthropic = AnthropicClient(settings)
    dispatcher = Dispatcher()
    dispatcher.include_router(ECGTelegramRouter(settings, anthropic).router)
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Начать работу"),
                BotCommand(command="help", description="Показать справку"),
            ]
        )
        for moderator_id in settings.moderator_user_ids:
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description="Начать работу"),
                    BotCommand(command="model", description="Выбрать модель анализа"),
                    BotCommand(command="usage", description="Токены и статистика"),
                    BotCommand(command="help", description="Показать справку"),
                ],
                scope=BotCommandScopeChat(chat_id=moderator_id),
            )
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        await anthropic.close()
        await bot.session.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
