from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse
from guildmaster_ai.llm.openai import OpenAIProvider

try:
    import openai as _openai_sdk

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI LLM provider using the official ``openai`` SDK.

    Install with::

        pip install "guildmaster-ai[azure]"

    Environment variables::

        AZURE_OPENAI_API_KEY=<your-key>
        AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
        # Optional — framework default: 2024-08-01-preview
        GUILD_AZURE_OPENAI_API_VERSION=2024-08-01-preview

    Azure OpenAI uses the same message/tool format as OpenAI.  The only
    difference is the client constructor which requires an ``azure_endpoint``
    and ``api_version`` instead of a plain ``base_url``.

    The ``default_model`` should be the **deployment name** you created in the
    Azure portal, e.g. ``"gpt-4o"``.

    Usage::

        provider = AzureOpenAIProvider(
            api_key="...",
            endpoint="https://my-resource.openai.azure.com/",
            api_version="2024-08-01-preview",
            default_model="gpt-4o",
        )
        response = await provider.complete(messages)
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str = "2024-08-01-preview",
        default_model: str = "gpt-4o",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required to use AzureOpenAIProvider. "
                "Install it with: pip install 'guildmaster-ai[azure]'"
            )
        # base_url is not meaningful for Azure (endpoint + api_version replace it)
        super().__init__(api_key=api_key, base_url=endpoint, default_model=default_model)
        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self._client = _openai_sdk.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        # Reuse the parse/message helpers from OpenAIProvider
        self._parse_response = OpenAIProvider._parse_response  # type: ignore[method-assign]
        self._to_sdk_messages = OpenAIProvider._to_sdk_messages  # type: ignore[method-assign]

    # ── public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": OpenAIProvider._to_sdk_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        res = await self._client.chat.completions.create(**kwargs)
        return OpenAIProvider._parse_response(res)

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": OpenAIProvider._to_sdk_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> AzureOpenAIProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
