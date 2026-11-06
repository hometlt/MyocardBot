from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

RELEASE_NOTES_FILE = Path(__file__).with_name("BOT_UPDATES.md")
_VERSION_PATTERN = re.compile(
    r"^Текущая версия:\s*(?P<version>\d+\.\d+\.\d+)\s*$",
    re.MULTILINE,
)
_SECTION_PATTERN = re.compile(r"^##\s+(?P<version>\d+\.\d+\.\d+)\s*$", re.MULTILINE)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BotRelease:
    version: str
    changes: tuple[str, ...]


class ReleaseNotificationStore:
    """Persist the last release notification successfully sent to each user."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._versions = self._load()

    def _load(self) -> dict[str, str]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.warning("Could not read release notification state", exc_info=True)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(user_id): version
            for user_id, version in payload.items()
            if str(user_id).isdigit()
            and isinstance(version, str)
            and re.fullmatch(r"\d+\.\d+\.\d+", version)
        }

    def was_notified(self, user_id: int, version: str) -> bool:
        return self._versions.get(str(user_id)) == version

    def mark_notified(self, user_id: int, version: str) -> None:
        self._versions[str(user_id)] = version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._versions, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def load_release_history(path: Path = RELEASE_NOTES_FILE) -> tuple[BotRelease, ...]:
    text = path.read_text(encoding="utf-8")
    releases: list[BotRelease] = []
    for match in _SECTION_PATTERN.finditer(text):
        section_start = match.end()
        next_section = re.search(r"^##\s+", text[section_start:], re.MULTILINE)
        section_end = (
            section_start + next_section.start()
            if next_section is not None
            else len(text)
        )
        changes = tuple(
            line.removeprefix("- ").strip()
            for line in text[section_start:section_end].splitlines()
            if line.startswith("- ") and line.removeprefix("- ").strip()
        )
        if changes:
            releases.append(
                BotRelease(version=match.group("version"), changes=changes)
            )
    return tuple(releases)


def load_current_release(path: Path = RELEASE_NOTES_FILE) -> BotRelease:
    text = path.read_text(encoding="utf-8")
    version_match = _VERSION_PATTERN.search(text)
    if version_match is None:
        raise ValueError(f"Не найдена текущая версия в {path}")

    version = version_match.group("version")
    section_pattern = re.compile(rf"^##\s+{re.escape(version)}\s*$", re.MULTILINE)
    section_match = section_pattern.search(text)
    if section_match is None:
        raise ValueError(f"Не найден раздел версии {version} в {path}")

    section_start = section_match.end()
    next_section = re.search(r"^##\s+", text[section_start:], re.MULTILINE)
    section_end = (
        section_start + next_section.start() if next_section is not None else len(text)
    )
    changes = tuple(
        line.removeprefix("- ").strip()
        for line in text[section_start:section_end].splitlines()
        if line.startswith("- ") and line.removeprefix("- ").strip()
    )
    if not changes:
        raise ValueError(f"Для версии {version} не описаны изменения в {path}")

    return BotRelease(version=version, changes=changes)


def _parse_version(version: str) -> tuple[int, int, int]:
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise ValueError(f"Некорректная версия: {version}")
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def next_major_version(version: str) -> str:
    major, _minor, _patch = _parse_version(version)
    return f"{major + 1}.0.0"


def next_minor_version(version: str) -> str:
    major, minor, _patch = _parse_version(version)
    return f"{major}.{minor + 1}.0"


def next_patch_version(version: str) -> str:
    major, minor, patch = _parse_version(version)
    return f"{major}.{minor}.{patch + 1}"


def format_startup_notification(release: BotRelease | None = None) -> str:
    current = release or CURRENT_RELEASE
    changes = "\n".join(f"• {change}" for change in current.changes)
    return (
        f"🟢 Бот «Миокард» v{current.version} в сети и готов принимать ЭКГ.\n\n"
        f"Что изменилось:\n{changes}"
    )


def format_release_history(releases: tuple[BotRelease, ...] | None = None) -> str:
    history = releases if releases is not None else RELEASE_HISTORY
    lines = ["📋 История версий бота «Миокард»"]
    for release in history:
        lines.extend(
            [
                "",
                f"v{release.version}",
                *(f"• {change}" for change in release.changes),
            ]
        )
    return "\n".join(lines)


CURRENT_RELEASE = load_current_release()
RELEASE_HISTORY = load_release_history()
