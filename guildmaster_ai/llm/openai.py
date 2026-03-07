from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse

try:
    import openai as _openai_sdk

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-backed LLM provider using the official ``openai`` SDK.

    Install with::

        pip install "guildmaster-ai[openai]"

    The OpenAI message format matches our internal ``LLMMessage`` format, so
    no conversion is required.

    Usage::

        provider = OpenAIProvider(api_key="sk-...")
        response = await provider.complete(messages)
        async for chunk in provider.stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required to use OpenAIProvider. "
                "Install it with: pip install 'guildmaster-ai[openai]'"
            )
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        self._client = _openai_sdk.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_sdk_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name is not None:
                entry["name"] = msg.name
            if msg.tool_calls is not None:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                entry["tool_call_id"] = msg.tool_call_id
            result.append(entry)
        return result

    @staticmethod
    def _parse_response(res: Any) -> LLMResponse:
        choice = res.choices[0]
        message = choice.message

        tool_calls: list[dict[str, Any]] | None = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage: dict[str, int] = {}
        if res.usage:
            usage = {
                "prompt_tokens": res.usage.prompt_tokens,
                "completion_tokens": res.usage.completion_tokens,
                "total_tokens": res.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content or "",
            model=res.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

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
            "messages": self._to_sdk_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        res = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(res)

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
            "messages": self._to_sdk_messages(messages),
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

    async def __aenter__(self) -> OpenAIProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
