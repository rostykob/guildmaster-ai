from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class GuildSettings(BaseSettings):
    """Central configuration for the Guildmaster-AI framework."""

    model_config = SettingsConfigDict(env_prefix="GUILD_", env_file=".env", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_default_model: str = "anthropic/claude-sonnet-4-20250514"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # ── Storage ──────────────────────────────────────────────────────────
    db_path: str = "guildmaster.db"

    # ── Quest behaviour ──────────────────────────────────────────────────
    max_clarification_rounds: int = 3
    max_quest_retries: int = 3

    # ── Observability ────────────────────────────────────────────────────
    log_level: str = "INFO"
