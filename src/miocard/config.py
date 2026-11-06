from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ANTHROPIC_MODELS = {
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-fable-5",
}
# claude-opus-4-6 is the only supported model without the xhigh effort level.
NO_XHIGH_MODELS = {"claude-opus-4-6"}
TOKENATOR_VISION_MODELS = {"gpt-5.6-sol", "gpt-5.6-terra", "gemini-3.5-flash"}
SUPPORTED_MODELS = ANTHROPIC_MODELS | TOKENATOR_VISION_MODELS
SUPPORTED_EFFORTS = {"low", "medium", "high", "max", "xhigh"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load the project configuration even when the console script is launched
        # from .venv/Scripts. A .env in the current directory, when present, wins.
        env_file=(PROJECT_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    telegram_bot_token: SecretStr
    anthropic_api_key: SecretStr
    anthropic_model: str = "gpt-5.6-sol"
    anthropic_effort: str = "high"
    # The enlarged-fragment pass answers one narrow question, so it does not need the
    # deliberation budget of the full-sheet report.
    anthropic_scout_effort: str = "low"
    # The specification asks for a multi-pass beat inventory, a reverse scan and a final
    # consistency check. Without reasoning that work has nowhere to happen except inside
    # the compact JSON the same specification asks to keep short.
    anthropic_thinking_enabled: bool = True
    # Reasoning tokens are billed as output, so the visible report needs headroom on top
    # of them. The benchmark averages ~2 860 tokens of report by itself.
    anthropic_max_tokens: int = Field(default=16_000, ge=1024, le=64_000)
    anthropic_scout_max_tokens: int = Field(default=4096, ge=512, le=16_000)
    anthropic_api_url: str | None = None
    anthropic_timeout_seconds: float = Field(default=300.0, ge=10, le=600)
    prompt_cache_enabled: bool = True

    allowed_user_ids: Annotated[frozenset[int], NoDecode] = frozenset()
    moderator_user_ids: Annotated[frozenset[int], NoDecode] = frozenset()
    admin_audit_include_content: bool = True
    max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    max_concurrent_analyses: int = Field(default=1, ge=1, le=20)
    user_cooldown_seconds: float = Field(default=10.0, ge=0, le=3600)
    analysis_cache_enabled: bool = False
    analysis_cache_dir: Path = Path(".cache/miocard")
    release_notification_state_file: Path = (
        PROJECT_ROOT / ".cache/miocard/release-notifications.json"
    )
    support_db_file: Path = PROJECT_ROOT / ".cache/miocard/support.sqlite3"

    @field_validator("allowed_user_ids", "moderator_user_ids", mode="before")
    @classmethod
    def parse_user_ids(cls, value: object) -> object:
        if value is None or value == "":
            return frozenset()
        if isinstance(value, str):
            return frozenset(int(part.strip()) for part in value.split(",") if part.strip())
        return value

    @field_validator("anthropic_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in SUPPORTED_MODELS:
            choices = ", ".join(sorted(SUPPORTED_MODELS))
            raise ValueError(f"ANTHROPIC_MODEL must be one of: {choices}")
        return value

    @field_validator("anthropic_effort", "anthropic_scout_effort")
    @classmethod
    def validate_effort(cls, value: str, info: ValidationInfo) -> str:
        if value not in SUPPORTED_EFFORTS:
            choices = ", ".join(sorted(SUPPORTED_EFFORTS))
            raise ValueError(f"{info.field_name.upper()} must be one of: {choices}")
        return value

    @model_validator(mode="after")
    def validate_model_effort_pair(self) -> Settings:
        if self.anthropic_model in TOKENATOR_VISION_MODELS and not self.is_tokenator:
            raise ValueError(
                f"{self.anthropic_model} is supported only through the Tokenator gateway"
            )
        if self.anthropic_model in NO_XHIGH_MODELS and "xhigh" in {
            self.anthropic_effort,
            self.anthropic_scout_effort,
        }:
            raise ValueError(f"xhigh effort is not supported by {self.anthropic_model}")
        return self

    @property
    def is_tokenator(self) -> bool:
        key = self.anthropic_api_key.get_secret_value()
        configured_host = urlparse(self.anthropic_api_url).hostname if self.anthropic_api_url else None
        return key.startswith("sk-fp-") or configured_host in {
            "api.tokenator.cloud",
            "dashboard.tokenator.cloud",
        }

    @property
    def effective_api_url(self) -> str:
        if self.anthropic_api_url:
            return self.anthropic_api_url
        if self.is_tokenator:
            return "https://api.tokenator.cloud/anthropic/v1/messages"
        return "https://api.anthropic.com/v1/messages"



@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
