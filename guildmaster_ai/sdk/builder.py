from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.config.settings import GuildSettings
from guildmaster_ai.llm.base_provider import BaseLLMProvider
from guildmaster_ai.sdk.guild import Guild

# Provider string → (module path, class name, required extra)
_PROVIDER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "openrouter": ("guildmaster_ai.llm.openrouter", "OpenRouterProvider", "openrouter"),
    "anthropic": ("guildmaster_ai.llm.anthropic", "AnthropicProvider", "anthropic"),
    "openai": ("guildmaster_ai.llm.openai", "OpenAIProvider", "openai"),
    "google": ("guildmaster_ai.llm.google", "GoogleProvider", "google"),
    "gemini": ("guildmaster_ai.llm.google", "GoogleProvider", "google"),
    "azure": ("guildmaster_ai.llm.azure", "AzureOpenAIProvider", "azure"),
    "bedrock": ("guildmaster_ai.llm.bedrock", "BedrockProvider", "bedrock"),
}


class GuildBuilder:
    """Fluent builder for constructing a :class:`Guild` instance.

    Supported provider strings for :meth:`with_llm_provider`:

    +-----------------+-------------------------------+----------------------------+
    | String          | Class                         | Extra                      |
    +=================+===============================+============================+
    | ``openrouter``  | ``OpenRouterProvider``        | ``openrouter``             |
    +-----------------+-------------------------------+----------------------------+
    | ``anthropic``   | ``AnthropicProvider``         | ``anthropic``              |
    +-----------------+-------------------------------+----------------------------+
    | ``openai``      | ``OpenAIProvider``            | ``openai``                 |
    +-----------------+-------------------------------+----------------------------+
    | ``google``      | ``GoogleProvider``            | ``google``                 |
    | ``gemini``      |                               |                            |
    +-----------------+-------------------------------+----------------------------+
    | ``azure``       | ``AzureOpenAIProvider``       | ``azure``                  |
    +-----------------+-------------------------------+----------------------------+
    | ``bedrock``     | ``BedrockProvider``           | ``bedrock``                |
    +-----------------+-------------------------------+----------------------------+
    """

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
        **provider_kwargs: object,
    ) -> GuildBuilder:
        """Configure the LLM provider.

        Pass a provider string (e.g. ``"openrouter"``) or a pre-constructed
        :class:`BaseLLMProvider` instance.  Extra keyword arguments are
        forwarded to the provider constructor — useful for Azure-specific
        params such as ``endpoint`` and ``api_version``, or Bedrock's
        ``region``.
        """
        if isinstance(provider, BaseLLMProvider):
            self._provider = provider
            return self

        if provider not in _PROVIDER_REGISTRY:
            known = ", ".join(f'"{k}"' for k in _PROVIDER_REGISTRY)
            raise ValueError(f"Unknown provider: {provider!r}. Known providers: {known}")

        module_path, class_name, _extra = _PROVIDER_REGISTRY[provider]

        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        self._provider = self._build_provider(
            cls,
            provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            **provider_kwargs,
        )
        return self

    def _build_provider(
        self,
        cls: type,
        provider_name: str,
        *,
        api_key: str | None,
        model: str | None,
        base_url: str | None,
        **extra: object,
    ) -> BaseLLMProvider:
        s = self._settings
        resolved_model = model or (
            s.llm_default_model if s.llm_default_model else None
        )
        resolved_base_url = base_url or (s.llm_base_url if s.llm_base_url else None)

        if provider_name in ("openrouter",):
            resolved_key = api_key or s.openrouter_api_key
            kwargs: dict[str, object] = {"api_key": resolved_key}
            if resolved_model:
                kwargs["default_model"] = resolved_model
            if resolved_base_url:
                kwargs["base_url"] = resolved_base_url
            return cls(**kwargs, **extra)  # type: ignore[return-value]

        if provider_name == "anthropic":
            resolved_key = api_key or s.anthropic_api_key
            kwargs = {"api_key": resolved_key}
            if resolved_model:
                kwargs["default_model"] = resolved_model
            if resolved_base_url:
                kwargs["base_url"] = resolved_base_url
            return cls(**kwargs, **extra)  # type: ignore[return-value]

        if provider_name == "openai":
            resolved_key = api_key or s.openai_api_key
            kwargs = {"api_key": resolved_key}
            if resolved_model:
                kwargs["default_model"] = resolved_model
            if resolved_base_url:
                kwargs["base_url"] = resolved_base_url
            return cls(**kwargs, **extra)  # type: ignore[return-value]

        if provider_name in ("google", "gemini"):
            resolved_key = api_key or s.google_api_key
            kwargs = {"api_key": resolved_key}
            if resolved_model:
                kwargs["default_model"] = resolved_model
            return cls(**kwargs, **extra)  # type: ignore[return-value]

        if provider_name == "azure":
            resolved_key = api_key or s.azure_openai_api_key
            endpoint = str(extra.pop("endpoint", None) or s.azure_openai_endpoint)
            api_version = str(extra.pop("api_version", None) or s.azure_openai_api_version)
            kwargs = {
                "api_key": resolved_key,
                "endpoint": endpoint,
                "api_version": api_version,
            }
            if resolved_model:
                kwargs["default_model"] = resolved_model
            return cls(**kwargs, **extra)  # type: ignore[return-value]

        if provider_name == "bedrock":
            region = str(extra.pop("region", None) or s.aws_region)
            kwargs = {"region": region}
            if s.aws_access_key_id:
                kwargs["aws_access_key_id"] = s.aws_access_key_id
            if s.aws_secret_access_key:
                kwargs["aws_secret_access_key"] = s.aws_secret_access_key
            if resolved_model:
                kwargs["default_model"] = resolved_model
            return cls(**kwargs, **extra)  # type: ignore[return-value]

        # Fallback — try to construct with api_key as first positional arg
        return cls(api_key or "", **extra)  # type: ignore[return-value]

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
