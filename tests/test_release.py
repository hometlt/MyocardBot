import tomllib
from pathlib import Path

import pytest

from miocard.release import (
    CURRENT_RELEASE,
    BotRelease,
    ReleaseNotificationStore,
    format_release_history,
    format_startup_notification,
    load_current_release,
    load_release_history,
    next_major_version,
    next_minor_version,
    next_patch_version,
)


def test_release_version_matches_project_version() -> None:
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))

    assert CURRENT_RELEASE.version == project["project"]["version"]


def test_current_release_has_short_user_facing_changes() -> None:
    assert CURRENT_RELEASE.changes
    assert all(len(change) <= 180 for change in CURRENT_RELEASE.changes)


def test_startup_notification_contains_version_and_changes() -> None:
    release = BotRelease(
        version="1.4.0",
        changes=("Бот лучше замечает нарушения ритма.", "Ответ стал понятнее."),
    )

    notification = format_startup_notification(release)

    assert "v1.4.0" in notification
    assert "Что изменилось:" in notification
    assert "• Бот лучше замечает нарушения ритма." in notification
    assert "• Ответ стал понятнее." in notification


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.2.0", "0.3.0"), ("1.9.7", "1.10.0")],
)
def test_next_minor_version(current: str, expected: str) -> None:
    assert next_minor_version(current) == expected


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.2.5", "1.0.0"), ("2.9.7", "3.0.0")],
)
def test_next_major_version(current: str, expected: str) -> None:
    assert next_major_version(current) == expected


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.2.0", "0.2.1"), ("1.9.7", "1.9.8")],
)
def test_next_patch_version(current: str, expected: str) -> None:
    assert next_patch_version(current) == expected


def test_release_notes_parser_reads_current_section(tmp_path: Path) -> None:
    release_file = tmp_path / "updates.md"
    release_file.write_text(
        """# Обновления

Текущая версия: 2.3.0

## 2.3.0

- Первое изменение.
- Второе изменение.

## 2.2.0

- Старое изменение.
""",
        encoding="utf-8",
    )

    assert load_current_release(release_file) == BotRelease(
        version="2.3.0",
        changes=("Первое изменение.", "Второе изменение."),
    )


def test_release_notes_parser_rejects_missing_changes(tmp_path: Path) -> None:
    release_file = tmp_path / "updates.md"
    release_file.write_text(
        "Текущая версия: 2.3.0\n\n## 2.3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="не описаны изменения"):
        load_current_release(release_file)


def test_release_history_parser_ignores_non_version_sections(tmp_path: Path) -> None:
    release_file = tmp_path / "updates.md"
    release_file.write_text(
        """# Обновления

Текущая версия: 1.1.0

## 1.1.0

- Новое изменение.

## 1.0.0

- Первое изменение.

## Правило версий

- Этот пункт не является изменением версии.
""",
        encoding="utf-8",
    )

    releases = load_release_history(release_file)

    assert releases == (
        BotRelease(version="1.1.0", changes=("Новое изменение.",)),
        BotRelease(version="1.0.0", changes=("Первое изменение.",)),
    )
    history = format_release_history(releases)
    assert "v1.1.0" in history
    assert "v1.0.0" in history
    assert "Этот пункт" not in history


def test_release_notification_store_persists_last_version_per_user(tmp_path: Path) -> None:
    state_file = tmp_path / "release-state.json"
    store = ReleaseNotificationStore(state_file)

    assert not store.was_notified(123, "1.2.3")
    store.mark_notified(123, "1.2.3")

    reloaded = ReleaseNotificationStore(state_file)
    assert reloaded.was_notified(123, "1.2.3")
    assert not reloaded.was_notified(123, "1.2.4")
