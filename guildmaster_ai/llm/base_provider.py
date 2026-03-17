from __future__ import annotations

from typing import Any

from guildmaster_ai.llm.types import GuildLLM


def create_chat_model(
    provider: str = "openrouter",
    *,
    api_key: str = "",
    model: str = "anthropic/claude-sonnet-4-20250514",
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> GuildLLM:
    """Factory that returns a configured LangChain chat model.

    Supports ``"openrouter"`` (default) which uses :class:`ChatOpenRouter`.
    You can also pass ``"openai"`` to get a plain ``ChatOpenAI``.
    """
    if provider == "openrouter":
        from guildmaster_ai.llm.openrouter import ChatOpenRouter

        return ChatOpenRouter(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise ValueError(f"Unknown provider: {provider!r}. Supported: 'openrouter', 'openai'.")
