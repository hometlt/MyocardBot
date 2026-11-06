import asyncio

import pytest

from miocard.single_instance import LocalBotInstance


@pytest.mark.asyncio
async def test_new_local_instance_takes_over_from_previous_one(tmp_path) -> None:
    first = LocalBotInstance("same-telegram-token", tmp_path)
    second = LocalBotInstance("same-telegram-token", tmp_path)
    assert await first.acquire_or_take_over() is False

    second_acquisition = asyncio.create_task(
        second.acquire_or_take_over(timeout_seconds=2, poll_interval=0.01)
    )
    await first.wait_for_takeover(poll_interval=0.01)
    first.release()

    assert await second_acquisition is True
    second.release()


@pytest.mark.asyncio
async def test_different_tokens_use_independent_local_locks(tmp_path) -> None:
    first = LocalBotInstance("first-token", tmp_path)
    second = LocalBotInstance("second-token", tmp_path)

    assert await first.acquire_or_take_over() is False
    assert await second.acquire_or_take_over() is False

    first.release()
    second.release()
