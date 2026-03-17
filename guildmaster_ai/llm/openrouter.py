from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ChatOpenRouter(ChatOpenAI):  # type: ignore[misc]
    """LangChain ChatModel backed by the OpenRouter API.

    Thin wrapper around ``ChatOpenAI`` that defaults to the OpenRouter
    endpoint and adds OpenRouter-specific HTTP headers.
    """

    def __init__(self, *, api_key: str = "", **kwargs: Any) -> None:
        if not api_key:
            from guildmaster_ai.config.settings import GuildSettings

            api_key = GuildSettings().openrouter_api_key
        super().__init__(
            api_key=api_key,  # type: ignore[arg-type]
            base_url=kwargs.pop("base_url", _OPENROUTER_BASE_URL),
            default_headers={
                "HTTP-Referer": "https://github.com/guildmaster-ai",
                "X-Title": "Guildmaster-AI",
                **kwargs.pop("default_headers", {}),
            },
            **kwargs,
        )
