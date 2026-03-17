from __future__ import annotations

from typing import Any

try:
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenAI

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class ChatGoogle(_ChatGoogleGenAI):  # type: ignore[misc]
    """LangChain ChatModel backed by Google Generative AI (Gemini).

    Thin wrapper around ``langchain_google_genai.ChatGoogleGenerativeAI``
    that checks for the optional dependency at construction time and provides
    sensible defaults.

    Install the extra with::

        pip install "guildmaster-ai[google]"
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        **kwargs: Any,
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'langchain-google-genai' package is required to use ChatGoogle. "
                "Install it with: pip install 'guildmaster-ai[google]'"
            )
        super().__init__(
            google_api_key=api_key,  # type: ignore[arg-type]
            model=model,
            **kwargs,
        )
