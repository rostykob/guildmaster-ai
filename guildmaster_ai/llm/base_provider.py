from __future__ import annotations

from typing import Any

from guildmaster_ai.llm.types import GuildLLM


def create_chat_model(
    provider: str = "openrouter",
    *,
    api_key: str = "",
    model: str = "anthropic/claude-sonnet-4-20250514",
    base_url: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> GuildLLM:
    """Factory that returns a configured LangChain chat model.

    Supported providers:

    - ``"openrouter"`` (default) — :class:`ChatOpenRouter`
    - ``"openai"`` — :class:`ChatOpenAIProvider`
    - ``"anthropic"`` — :class:`ChatAnthropic` (requires ``langchain-anthropic``)
    - ``"google"`` / ``"gemini"`` — :class:`ChatGoogle` (requires ``langchain-google-genai``)
    - ``"azure"`` — :class:`ChatAzureOpenAI` (requires ``azure_endpoint`` kwarg)
    - ``"bedrock"`` — :class:`ChatBedrock` (requires ``langchain-aws``)
    """
    common: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider == "openrouter":
        from guildmaster_ai.llm.openrouter import ChatOpenRouter

        return ChatOpenRouter(
            api_key=api_key,
            model=model,
            **({"base_url": base_url} if base_url else {}),
            **common,
            **kwargs,
        )

    if provider == "openai":
        from guildmaster_ai.llm.openai import ChatOpenAIProvider

        return ChatOpenAIProvider(
            api_key=api_key,
            model=model,
            **({"base_url": base_url} if base_url else {}),
            **common,
            **kwargs,
        )

    if provider == "anthropic":
        from guildmaster_ai.llm.anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=api_key,
            model=model,
            **({"base_url": base_url} if base_url else {}),
            **common,
            **kwargs,
        )

    if provider in ("google", "gemini"):
        from guildmaster_ai.llm.google import ChatGoogle

        return ChatGoogle(
            api_key=api_key,
            model=model,
            **common,
            **kwargs,
        )

    if provider == "azure":
        from guildmaster_ai.llm.azure import ChatAzureOpenAI

        return ChatAzureOpenAI(
            api_key=api_key,
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
