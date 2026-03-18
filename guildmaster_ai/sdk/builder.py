from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.adventurers.base_guard import BaseGuard
from guildmaster_ai.config.settings import GuildSettings
from guildmaster_ai.llm.base_provider import create_chat_model
from guildmaster_ai.llm.types import GuildLLM
from guildmaster_ai.sdk.guild import Guild


class GuildBuilder:
    """Fluent builder for constructing a :class:`Guild` instance.

    Supported provider strings for :meth:`with_llm_provider`:

    +-----------------+-------------------------------+----------------------------+
    | String          | Class                         | Extra                      |
    +=================+===============================+============================+
    | ``openrouter``  | ``ChatOpenRouter``            | ``openrouter``             |
    +-----------------+-------------------------------+----------------------------+
    | ``anthropic``   | ``ChatAnthropic``             | ``anthropic``              |
    +-----------------+-------------------------------+----------------------------+
    | ``openai``      | ``ChatOpenAIProvider``        | (core dep)                 |
    +-----------------+-------------------------------+----------------------------+
    | ``google``      | ``ChatGoogle``                | ``google``                 |
    | ``gemini``      |                               |                            |
    +-----------------+-------------------------------+----------------------------+
    | ``azure``       | ``ChatAzureOpenAI``           | (core dep)                 |
    +-----------------+-------------------------------+----------------------------+
    | ``bedrock``     | ``ChatBedrock``               | ``bedrock``                |
    +-----------------+-------------------------------+----------------------------+
    """

    def __init__(self) -> None:
        self._llm: GuildLLM | None = None
        self._adventurers: list[tuple[type[BaseAdventurer] | BaseAdventurer, int]] = []
        self._settings = GuildSettings()
        self._guard: BaseGuard | bool = False

    def _resolve_api_key(self, provider: str, api_key: str | None) -> str:
        """Return the API key for *provider*, falling back to settings."""
        if api_key:
            return api_key
        s = self._settings
        key_map: dict[str, str] = {
            "openrouter": s.openrouter_api_key,
            "anthropic": s.anthropic_api_key,
            "openai": s.openai_api_key,
            "google": s.google_api_key,
            "gemini": s.google_api_key,
            "azure": s.azure_openai_api_key,
        }
        return key_map.get(provider, "")

    def with_llm_provider(
        self,
        provider: str | GuildLLM = "openrouter",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **provider_kwargs: object,
    ) -> GuildBuilder:
        """Configure the LLM provider.

        Pass a pre-configured ``GuildLLM`` instance (any LangChain
        ``BaseChatModel``) directly, or a provider name string to create
        one from settings.
        """
        if isinstance(provider, GuildLLM):
            self._llm = provider
        elif isinstance(provider, str):
            s = self._settings
            resolved_key = self._resolve_api_key(provider, api_key)
            resolved_model = model or s.llm_default_model

            extra_kwargs: dict[str, object] = dict(provider_kwargs)

            # Azure-specific defaults from settings
            if provider == "azure":
                extra_kwargs.setdefault(
                    "azure_endpoint",
                    s.azure_openai_endpoint,
                )
                extra_kwargs.setdefault(
                    "api_version",
                    s.azure_openai_api_version,
                )

            # Bedrock-specific defaults from settings
            if provider == "bedrock":
                extra_kwargs.setdefault("region_name", s.aws_region)
                if s.aws_access_key_id:
                    extra_kwargs.setdefault("aws_access_key_id", s.aws_access_key_id)
                if s.aws_secret_access_key:
                    extra_kwargs.setdefault("aws_secret_access_key", s.aws_secret_access_key)

            self._llm = create_chat_model(
                provider,
                api_key=resolved_key,
                model=resolved_model,
                base_url=base_url or s.llm_base_url,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
                **extra_kwargs,
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

    def with_guard(self, guard: BaseGuard | None = None) -> GuildBuilder:
        """Enable the guard agent for safety verification.

        Pass a custom :class:`BaseGuard` instance, or omit to use the
        built-in LLM-as-judge guard.
        """
        self._guard = guard if guard is not None else True
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

        if self._guard:
            custom = self._guard if isinstance(self._guard, BaseGuard) else None
            guild.enable_guard(custom)

        return guild
