from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO


class InstanceTakeoverError(RuntimeError):
    pass


def _try_lock(handle: BinaryIO) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LocalBotInstance:
    """Coordinate one polling process per Telegram token on this computer."""

    def __init__(
        self,
        token: str,
        directory: Path | None = None,
    ) -> None:
        instance_key = hashlib.sha256(token.encode("utf-8")).hexdigest()[:20]
        base_dir = directory or Path(tempfile.gettempdir())
        self.lock_path = base_dir / f"miocard-{instance_key}.lock"
        self.takeover_path = base_dir / f"miocard-{instance_key}.takeover"
        self._handle: BinaryIO | None = None

    def _acquire_now(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        if not _try_lock(handle):
            handle.close()
            return False
        self._handle = handle
        return True

    def _request_takeover(self) -> None:
        request_id = uuid.uuid4().hex
        temporary_path = self.takeover_path.with_suffix(f".{request_id}.tmp")
        temporary_path.write_text(request_id, encoding="ascii")
        os.replace(temporary_path, self.takeover_path)

    def _clear_takeover_request(self) -> None:
        self.takeover_path.unlink(missing_ok=True)

    async def acquire_or_take_over(
        self,
        *,
        timeout_seconds: float = 600.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """Acquire the local lock, asking the previous compatible process to exit."""
        if self._acquire_now():
            self._clear_takeover_request()
            return False

        self._request_takeover()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            await asyncio.sleep(poll_interval)
            if self._acquire_now():
                self._clear_takeover_request()
                return True
        raise InstanceTakeoverError(
            "Другой локальный экземпляр бота не завершился за отведённое время."
        )

    async def wait_for_takeover(self, *, poll_interval: float = 0.25) -> None:
        if self._handle is None:
            raise RuntimeError("Local bot instance lock is not acquired")
        while True:
            await asyncio.sleep(poll_interval)
            try:
                request_id = self.takeover_path.read_text(encoding="ascii").strip()
            except FileNotFoundError:
                continue
            if request_id:
                return

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            _unlock(handle)
        finally:
            handle.close()
