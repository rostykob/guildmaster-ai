from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

# ── Data models ──────────────────────────────────────────────────────────────


class LLMMessage(BaseModel):
    """A single message in a chat-completion conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    """Structured response returned by an LLM provider."""

    content: str
    model: str
    usage: dict[str, int]
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


# ── Abstract provider ────────────────────────────────────────────────────────


class BaseLLMProvider(ABC):
    """Interface that every LLM backend must implement."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]: ...
