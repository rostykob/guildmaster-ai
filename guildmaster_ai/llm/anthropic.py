from __future__ import annotations

from typing import Any

try:
    from langchain_anthropic import ChatAnthropic as _ChatAnthropic

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class ChatAnthropic(_ChatAnthropic):
    """LangChain ChatModel backed by the Anthropic API.

    Thin wrapper around ``langchain_anthropic.ChatAnthropic`` that checks
    for the optional dependency at construction time and provides sensible
    defaults.

    Install the extra with::

        pip install "guildmaster-ai[anthropic]"
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "claude-sonnet-4-5-20250514",
        **kwargs: Any,
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'langchain-anthropic' package is required to use ChatAnthropic. "
                "Install it with: pip install 'guildmaster-ai[anthropic]'"
            )
        super().__init__(
            api_key=api_key,  # type: ignore[arg-type]
            model_name=model,
            **kwargs,
        )
