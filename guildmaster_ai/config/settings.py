from __future__ import annotations

from pathlib import Path

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
    llm_default_model: str = "anthropic/claude-sonnet-4"
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
    guild_home: Path = Path("./.guildmaster")

    @property
    def db_path(self) -> Path:
        return self.guild_home / "guild.db"

    @property
    def chroma_path(self) -> Path:
        return self.guild_home / "chroma"

    # ── Quest behaviour ──────────────────────────────────────────────────
    # Refine each request via the receptionist (title + acceptance criteria,
    # one LLM call). Disable for a faster lane: raw request goes straight to
    # triage, and without criteria the verification call is skipped too.
    refine_requests: bool = True
    max_clarification_rounds: int = 3
    max_quest_retries: int = 3
    # How many quests may execute concurrently in the background worker.
    max_concurrent_quests: int = 3
    # Owner-provided domain context, appended to the guildmaster's and
    # receptionist's system prompts (triage, planning, verification, intake).
    # Use it to scope a guild to one domain (e.g. "an English exam tutoring
    # guild"). Never replaces the built-in prompts.
    guild_charter: str = ""
    # Minimum quest rank at which the guildmaster's LLM result verification
    # runs. "F" (default) verifies every rank; "D" skips the extra LLM call
    # for trivial F/E quests. The guard, when enabled, always runs.
    verify_min_rank: str = "F"

    # ── Librarian / memory ───────────────────────────────────────────────
    # Librarian observations (LLM analysis of finished quests) are disconnected
    # from the main quest flow by default: they add an LLM call + embedding per
    # quest without feeding back into planning yet. Quests, results, findings,
    # and conversations are always persisted to SQLite regardless.
    enable_observations: bool = False
    # Persist librarian observations to the Chroma vector store for semantic
    # recall of past quests (only used when enable_observations is on).
    enable_vector_store: bool = True
    # Run quest archival (LLM analysis + persistence) off the request path.
    background_archival: bool = True

    # ── Observability ────────────────────────────────────────────────────
    log_level: str = "INFO"
