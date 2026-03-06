from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.config.settings import GuildSettings
from guildmaster_ai.llm.base_provider import BaseLLMProvider
from guildmaster_ai.llm.openrouter import OpenRouterProvider
from guildmaster_ai.sdk.guild import Guild


class GuildBuilder:
    """Fluent builder for constructing a Guild instance."""

    def __init__(self) -> None:
        self._provider: BaseLLMProvider | None = None
        self._adventurers: list[tuple[type[BaseAdventurer] | BaseAdventurer, int]] = []
        self._settings = GuildSettings()
        self._enable_guard: bool = False

    def with_llm_provider(
        self,
        provider: str | BaseLLMProvider = "openrouter",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> GuildBuilder:
        """Configure the LLM provider."""
        if isinstance(provider, BaseLLMProvider):
            self._provider = provider
        elif provider == "openrouter":
            self._provider = OpenRouterProvider(
                api_key=api_key or self._settings.llm_api_key,
                base_url=base_url or self._settings.llm_base_url,
                default_model=model or self._settings.llm_default_model,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        return self

    def register_adventurer(
        self,
        adventurer: type[BaseAdventurer] | BaseAdventurer,
        count: int = 1,
    ) -> GuildBuilder:
        """Register an adventurer class or instance to be added to the guild."""
        self._adventurers.append((adventurer, count))
        return self

    def with_guard(self) -> GuildBuilder:
        """Enable the guard agent for safety verification."""
        self._enable_guard = True
        return self

    def with_settings(self, **kwargs) -> GuildBuilder:
        """Override individual settings by keyword argument."""
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        return self

    def build(self) -> Guild:
        """Build and return the configured Guild instance."""
        if self._provider is None:
            raise ValueError("LLM provider is required. Call with_llm_provider() first.")

        guild = Guild(llm_provider=self._provider, settings=self._settings)

        for adventurer_or_cls, count in self._adventurers:
            for _i in range(count):
                if isinstance(adventurer_or_cls, type):
                    adv = adventurer_or_cls()
                else:
                    adv = adventurer_or_cls
                guild.register_adventurer(adv)

        if self._enable_guard:
            guild.enable_guard()

        return guild
