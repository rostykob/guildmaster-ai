from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuildSettings(BaseSettings):
    """Central configuration for the Guildmaster-AI framework.

    Framework-level settings use the ``GUILD_`` prefix.
    Provider API keys are read from their canonical env var names
    (e.g. ``OPENROUTER_API_KEY``, ``ANTHROPIC_API_KEY``) so that existing
    credentials are picked up automatically without renaming variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="GUILD_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Provider selection ────────────────────────────────────────────────
    llm_provider: str = "openrouter"
    llm_default_model: str = "anthropic/claude-sonnet-4-20250514"
    llm_base_url: str = ""  # empty → each provider uses its own default
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # ── Provider API keys — official env var names (no GUILD_ prefix) ─────
    # OpenRouter
    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "GUILD_LLM_API_KEY"),
    )

    # Anthropic
    anthropic_api_key: str = Field(
        default="",
        validation_alias="ANTHROPIC_API_KEY",
    )

    # OpenAI
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
    )

    # Google / Gemini — support both canonical names
    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    # Azure OpenAI
    azure_openai_api_key: str = Field(
        default="",
        validation_alias="AZURE_OPENAI_API_KEY",
    )
    azure_openai_endpoint: str = Field(
        default="",
        validation_alias="AZURE_OPENAI_ENDPOINT",
    )
    # e.g. "2024-08-01-preview" — uses GUILD_ prefix
    azure_openai_api_version: str = "2024-08-01-preview"

    # AWS / Bedrock — read from the standard AWS credential env vars
    aws_access_key_id: str = Field(
        default="",
        validation_alias="AWS_ACCESS_KEY_ID",
    )
    aws_secret_access_key: str = Field(
        default="",
        validation_alias="AWS_SECRET_ACCESS_KEY",
    )
    aws_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("AWS_DEFAULT_REGION", "AWS_REGION"),
    )

    # ── Storage ──────────────────────────────────────────────────────────
    db_path: str = "guildmaster.db"

    # ── Quest behaviour ──────────────────────────────────────────────────
    max_clarification_rounds: int = 3
    max_quest_retries: int = 3

    # ── Observability ────────────────────────────────────────────────────
    log_level: str = "INFO"
