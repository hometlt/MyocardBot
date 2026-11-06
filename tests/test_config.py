import pytest
from pydantic import ValidationError

from miocard.config import PROJECT_ENV_FILE, PROJECT_ROOT, Settings


def test_project_env_file_is_resolved_independently_of_working_directory(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert PROJECT_ENV_FILE == PROJECT_ROOT / ".env"
    assert PROJECT_ENV_FILE.is_absolute()
    assert Settings.model_config["env_file"] == (PROJECT_ENV_FILE, ".env")


def test_comma_separated_allowed_user_ids(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_USER_IDS", "153241307, 2053404449")
    settings = Settings(
        telegram_bot_token="123456:telegram-secret",
        anthropic_api_key="secret",
        anthropic_model="claude-sonnet-5",
        _env_file=None,
    )
    assert settings.allowed_user_ids == frozenset({153241307, 2053404449})


def test_empty_allowed_user_ids(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_USER_IDS", "")
    settings = Settings(
        telegram_bot_token="123456:telegram-secret",
        anthropic_api_key="secret",
        anthropic_model="claude-sonnet-5",
        _env_file=None,
    )
    assert settings.allowed_user_ids == frozenset()


def test_comma_separated_moderator_user_ids(monkeypatch) -> None:
    monkeypatch.setenv("MODERATOR_USER_IDS", "153241307, 987654321")
    settings = Settings(
        telegram_bot_token="123456:telegram-secret",
        anthropic_api_key="secret",
        anthropic_model="claude-sonnet-5",
        _env_file=None,
    )
    assert settings.moderator_user_ids == frozenset({153241307, 987654321})


def test_tokenator_key_selects_compatible_endpoint() -> None:
    settings = Settings(
        telegram_bot_token="123456:telegram-secret",
        anthropic_api_key="sk-fp-test-key",
        _env_file=None,
    )
    assert settings.is_tokenator is True
    assert settings.anthropic_model == "gpt-5.6-sol"
    assert settings.effective_api_url == "https://api.tokenator.cloud/anthropic/v1/messages"


def test_tokenator_accepts_benchmarked_vision_model() -> None:
    settings = Settings(
        telegram_bot_token="123456:telegram-secret",
        anthropic_api_key="sk-fp-test-key",
        anthropic_model="gpt-5.6-sol",
        _env_file=None,
    )
    assert settings.anthropic_model == "gpt-5.6-sol"


def test_tokenator_only_model_is_rejected_for_direct_anthropic() -> None:
    with pytest.raises(ValidationError, match="supported only through the Tokenator"):
        Settings(
            telegram_bot_token="123456:telegram-secret",
            anthropic_api_key="sk-ant-test-key",
            anthropic_model="gpt-5.6-sol",
            _env_file=None,
        )
