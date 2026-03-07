from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse

try:
    import anthropic as _anthropic_sdk

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic-backed LLM provider using the official ``anthropic`` SDK.

    Install with::

        pip install "guildmaster-ai[anthropic]"

    The Anthropic API differs from the OpenAI format in several ways:

    * The system prompt is a top-level ``system`` parameter, not a message.
    * Tool definitions use ``input_schema`` instead of ``parameters``.
    * Tool call results are sent as ``tool_result`` content blocks inside a
      ``user`` role message.

    This provider converts between the internal OpenAI-compatible format and
    the Anthropic format transparently.

    Usage::

        provider = AnthropicProvider(api_key="sk-ant-...")
        response = await provider.complete(messages)
        async for chunk in provider.stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        default_model: str = "claude-sonnet-4-5",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicProvider. "
                "Install it with: pip install 'guildmaster-ai[anthropic]'"
            )
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        self._client = _anthropic_sdk.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url if base_url != "https://api.anthropic.com" else None,
        )

    # ── format helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _split_system(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract the system prompt and convert the remaining messages."""
        system = ""
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
                continue

            if msg.role == "tool":
                # Tool result — must be wrapped as a user message with a
                # tool_result content block.
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                })
                continue

            entry: dict[str, Any] = {"role": msg.role}

            if msg.tool_calls:
                # Assistant message that made tool calls.
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    args = tc.get("function", {}).get("arguments", "{}")
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": json.loads(args) if isinstance(args, str) else args,
                    })
                entry["content"] = content_blocks
            else:
                entry["content"] = msg.content

            converted.append(entry)

        return system, converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-compatible tool specs to Anthropic format."""
        result = []
        for tool in tools:
            fn = tool.get("function", {})
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    @staticmethod
    def _parse_response(res: Any) -> LLMResponse:
        """Convert an Anthropic ``Message`` to an internal ``LLMResponse``."""
        text_parts = [block.text for block in res.content if block.type == "text"]
        content = "".join(text_parts)

        tool_calls: list[dict[str, Any]] | None = None
        tool_blocks = [b for b in res.content if b.type == "tool_use"]
        if tool_blocks:
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                    },
                }
                for b in tool_blocks
            ]

        usage = {
            "prompt_tokens": res.usage.input_tokens,
            "completion_tokens": res.usage.output_tokens,
            "total_tokens": res.usage.input_tokens + res.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            model=res.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=res.stop_reason,
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
        system, converted = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        res = await self._client.messages.create(**kwargs)
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
        system, converted = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> AnthropicProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
