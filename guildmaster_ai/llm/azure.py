from __future__ import annotations

from typing import Any

from langchain_openai import AzureChatOpenAI as _AzureChatOpenAI


class ChatAzureOpenAI(_AzureChatOpenAI):
    """LangChain ChatModel backed by Azure OpenAI.

    Thin wrapper around ``langchain_openai.AzureChatOpenAI`` that provides
    sensible defaults for the Guildmaster-AI framework.

    ``langchain-openai`` is a core dependency, so no extra install is needed.
    The ``openai`` package (also a core transitive dependency) provides the
    ``AsyncAzureOpenAI`` client under the hood.

    Required parameters:

    - ``api_key``: Azure OpenAI API key
    - ``azure_endpoint``: Full endpoint URL (e.g. ``https://my-resource.openai.azure.com/``)
    - ``api_version``: API version string (e.g. ``2024-08-01-preview``)
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2024-08-01-preview",
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,  # type: ignore[arg-type]
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=model,
            **kwargs,
        )
