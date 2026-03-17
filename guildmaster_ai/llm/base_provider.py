from __future__ import annotations

from typing import Any

from guildmaster_ai.llm.types import GuildLLM


def _resolve_api_key(provider: str, api_key: str) -> str:
    """Return *api_key* as-is when non-empty, otherwise look it up in GuildSettings."""
    if api_key:
        return api_key
    from guildmaster_ai.config.settings import GuildSettings

    s = GuildSettings()
    key_map: dict[str, str] = {
        "openrouter": s.openrouter_api_key,
        "anthropic": s.anthropic_api_key,
        "openai": s.openai_api_key,
        "google": s.google_api_key,
        "gemini": s.google_api_key,
        "azure": s.azure_openai_api_key,
    }
    return key_map.get(provider, "")


def create_chat_model(
    provider: str = "openrouter",
    *,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> GuildLLM:
    """Factory that returns a configured LangChain chat model.

    When *api_key* or *model* are omitted the values are resolved from
    :class:`GuildSettings` (environment variables / ``.env`` file).

    Supported providers:

    - ``"openrouter"`` (default) — :class:`ChatOpenRouter`
    - ``"openai"`` — :class:`ChatOpenAIProvider`
    - ``"anthropic"`` — :class:`ChatAnthropic` (requires ``langchain-anthropic``)
    - ``"google"`` / ``"gemini"`` — :class:`ChatGoogle` (requires ``langchain-google-genai``)
    - ``"azure"`` — :class:`ChatAzureOpenAI` (requires ``azure_endpoint`` kwarg)
    - ``"bedrock"`` — :class:`ChatBedrock` (requires ``langchain-aws``)
    """
    resolved_key = _resolve_api_key(provider, api_key)

    if not model:
        from guildmaster_ai.config.settings import GuildSettings

        model = GuildSettings().llm_default_model

    common: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider == "openrouter":
        from guildmaster_ai.llm.openrouter import ChatOpenRouter

        return ChatOpenRouter(
            api_key=resolved_key,
            model=model,
            **({"base_url": base_url} if base_url else {}),
            **common,
            **kwargs,
        )

    if provider == "openai":
        from guildmaster_ai.llm.openai import ChatOpenAIProvider

        return ChatOpenAIProvider(
            api_key=resolved_key,
            model=model,
            **({"base_url": base_url} if base_url else {}),
            **common,
            **kwargs,
        )

    if provider == "anthropic":
        from guildmaster_ai.llm.anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=resolved_key,
            model=model,
            **({"base_url": base_url} if base_url else {}),
            **common,
            **kwargs,
        )

    if provider in ("google", "gemini"):
        from guildmaster_ai.llm.google import ChatGoogle

        return ChatGoogle(
            api_key=resolved_key,
            model=model,
            **common,
            **kwargs,
        )

    if provider == "azure":
        from guildmaster_ai.llm.azure import ChatAzureOpenAI

        return ChatAzureOpenAI(
            api_key=resolved_key,
            model=model,
            **common,
            **kwargs,
        )

    if provider == "bedrock":
        from guildmaster_ai.llm.bedrock import ChatBedrock

        return ChatBedrock(
            model_id=model,
            **common,
            **kwargs,
        )

    known = "openrouter, openai, anthropic, google, gemini, azure, bedrock"
    raise ValueError(f"Unknown provider: {provider!r}. Supported: {known}")
