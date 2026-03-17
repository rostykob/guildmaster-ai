from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.config.settings import GuildSettings
from guildmaster_ai.llm.base_provider import create_chat_model
from guildmaster_ai.llm.types import GuildLLM
from guildmaster_ai.sdk.guild import Guild


class GuildBuilder:
    """Fluent builder for constructing a :class:`Guild` instance.

    Pass a provider string (e.g. ``"openrouter"``) or a pre-configured
    ``GuildLLM`` (any LangChain ``BaseChatModel``) to
    :meth:`with_llm_provider`.
    """

    def __init__(self) -> None:
        self._llm: GuildLLM | None = None
        self._adventurers: list[tuple[type[BaseAdventurer] | BaseAdventurer, int]] = []
        self._settings = GuildSettings()
        self._enable_guard: bool = False

    def with_llm_provider(
        self,
        provider: str | GuildLLM = "openrouter",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **provider_kwargs: object,
    ) -> GuildBuilder:
        """Configure the LLM provider.

        Pass a pre-configured ``GuildLLM`` instance directly, or a
        provider name (``"openrouter"`` or ``"openai"``) to create one from
        settings.
        """
        if isinstance(provider, GuildLLM):
            self._llm = provider
        elif isinstance(provider, str):
            kwargs = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            self._llm = create_chat_model(
                provider,
                api_key=api_key or self._settings.openrouter_api_key,
                model=model or self._settings.llm_default_model,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                **kwargs,
            )
        else:
            raise TypeError(f"Expected str or GuildLLM, got {type(provider)}")
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

    def with_settings(self, **kwargs: object) -> GuildBuilder:
        """Override individual settings by keyword argument."""
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        return self

    def build(self) -> Guild:
        """Build and return the configured :class:`Guild` instance."""
        if self._llm is None:
            raise ValueError("LLM provider is required. Call with_llm_provider() first.")

        guild = Guild(llm=self._llm, settings=self._settings)

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
