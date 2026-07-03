from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI as _ChatOpenAI


class ChatOpenAIProvider(_ChatOpenAI):
    """LangChain ChatModel backed by the OpenAI API.

    Thin wrapper around ``langchain_openai.ChatOpenAI`` that provides
    sensible defaults for the Guildmaster-AI framework.

    ``langchain-openai`` is a core dependency, so no extra install is needed.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            **kwargs,
        )
