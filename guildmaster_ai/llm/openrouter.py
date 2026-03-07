from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse

try:
    from openrouter import OpenRouter as _OpenRouterSDK
    from openrouter.utils import eventstreaming  # noqa: F401 – confirms SDK is present

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter-backed LLM provider using the official ``openrouter`` SDK.

    The SDK is an optional dependency.  Install it with::

        pip install "guildmaster-ai[openrouter]"

    If the SDK is not installed a clear ``ImportError`` is raised at
    construction time so the problem is obvious rather than surfacing as a
    mysterious ``AttributeError`` later.

    Usage::

        provider = OpenRouterProvider(api_key="sk-or-...")
        response = await provider.complete(messages)
        async for chunk in provider.stream(messages):
            print(chunk, end="", flush=True)
        await provider.close()

    The provider is also usable as an async context manager::

        async with OpenRouterProvider(api_key="sk-or-...") as provider:
            response = await provider.complete(messages)
    """

    _HTTP_REFERER: str = "https://github.com/guildmaster-ai"
    _X_TITLE: str = "Guildmaster-AI"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "anthropic/claude-sonnet-4-20250514",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'openrouter' package is required to use OpenRouterProvider. "
                "Install it with: pip install 'guildmaster-ai[openrouter]'"
            )
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        self._sdk = _OpenRouterSDK(
            api_key=api_key,
            http_referer=self._HTTP_REFERER,
            x_title=self._X_TITLE,
            server_url=base_url,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_sdk_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert internal ``LLMMessage`` objects to the dict format expected by the SDK."""
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
    def _parse_sdk_response(res: Any) -> LLMResponse:
        """Convert an SDK ``ChatResponse`` into an internal ``LLMResponse``."""
        choice = res.choices[0]
        message = choice.message
        content: str = ""
        if message.content is not None:
            content = message.content if isinstance(message.content, str) else ""

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
        if res.usage is not None:
            raw = res.usage
            # The SDK usage object exposes prompt_tokens, completion_tokens, total_tokens
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(raw, field, None)
                if val is not None:
                    usage[field] = int(val)

        finish_reason: str | None = None
        if choice.finish_reason and choice.finish_reason is not None:
            finish_reason = str(choice.finish_reason)

        return LLMResponse(
            content=content,
            model=res.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
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
        """Send a non-streaming chat completion request and return the full response."""
        sdk_messages = self._to_sdk_messages(messages)
        kwargs: dict[str, Any] = {
            "messages": sdk_messages,
            "model": model or self.default_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        res = await self._sdk.chat.send_async(**kwargs)
        return self._parse_sdk_response(res)

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding text delta chunks as they arrive."""
        sdk_messages = self._to_sdk_messages(messages)
        kwargs: dict[str, Any] = {
            "messages": sdk_messages,
            "model": model or self.default_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        event_stream = await self._sdk.chat.send_async(**kwargs)
        async for chunk in event_stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content

    async def close(self) -> None:
        """Release the underlying SDK HTTP client."""
        await self._sdk.__aexit__(None, None, None)

    # ── async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> OpenRouterProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
