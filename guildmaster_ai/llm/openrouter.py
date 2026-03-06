from __future__ import annotations

import json
import typing
from collections.abc import AsyncIterator
from typing import Any

import httpx

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter-backed LLM provider using the OpenAI-compatible API."""

    _HEADERS_EXTRA: typing.ClassVar[dict[str, str]] = {
        "HTTP-Referer": "https://github.com/guildmaster-ai",
        "X-Title": "Guildmaster-AI",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "anthropic/claude-sonnet-4-20250514",
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self._HEADERS_EXTRA,
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]
        message = choice["message"]
        return LLMResponse(
            content=message.get("content") or "",
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            tool_calls=message.get("tool_calls"),
            finish_reason=choice.get("finish_reason"),
        )

    # ── public API ───────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return self._parse_response(response.json())

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
        )
        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        await self._client.aclose()
