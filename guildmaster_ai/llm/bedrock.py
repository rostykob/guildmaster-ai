from __future__ import annotations

from typing import Any

try:
    from langchain_aws import ChatBedrockConverse as _ChatBedrockConverse

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class ChatBedrock(_ChatBedrockConverse):  # type: ignore[misc]
    """LangChain ChatModel backed by AWS Bedrock (Converse API).

    Thin wrapper around ``langchain_aws.ChatBedrockConverse`` that checks
    for the optional dependency at construction time and provides sensible
    defaults.

    Install the extra with::

        pip install "guildmaster-ai[bedrock]"
    """

    def __init__(
        self,
        *,
        region_name: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        **kwargs: Any,
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'langchain-aws' package is required to use ChatBedrock. "
                "Install it with: pip install 'guildmaster-ai[bedrock]'"
            )
        super().__init__(
            region_name=region_name,
            model_id=model_id,
            **kwargs,
        )
