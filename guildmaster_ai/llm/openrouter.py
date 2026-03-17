from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI


class ChatOpenRouter(ChatOpenAI):  # type: ignore[misc]
    """LangChain ChatModel backed by the OpenRouter API.

    Thin wrapper around ``ChatOpenAI`` that defaults to the OpenRouter
    endpoint and adds OpenRouter-specific HTTP headers.
    """

    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    def __init__(self, *, api_key: str = "", **kwargs: Any) -> None:
        super().__init__(
            api_key=api_key,  # type: ignore[arg-type]
            base_url=kwargs.pop("base_url", self.openrouter_base_url),
            default_headers={
                "HTTP-Referer": "https://github.com/guildmaster-ai",
                "X-Title": "Guildmaster-AI",
                **kwargs.pop("default_headers", {}),
            },
            **kwargs,
        )
