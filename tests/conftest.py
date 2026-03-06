"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from guildmaster_ai.core.messages import QuestDraft
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        response_content: str = "Mock response",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(api_key="test-key", base_url="http://localhost", default_model="mock")
        self.response_content = response_content
        self.tool_calls = tool_calls
        self.call_count = 0
        self.last_messages: list[LLMMessage] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        return LLMResponse(
            content=self.response_content,
            model=model or self.default_model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            tool_calls=self.tool_calls if self.call_count == 1 else None,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        yield self.response_content


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def sample_quest() -> Quest:
    return Quest(
        title="Test Quest",
        description="A test quest for unit testing",
        required_talents=["general"],
        rank=QuestRank.E,
        acceptance_criteria=["Must return a result"],
    )


@pytest.fixture
def sample_draft() -> QuestDraft:
    return QuestDraft(
        title="Test Quest",
        description="A test quest",
        required_talents=["general"],
        acceptance_criteria=["Must succeed"],
    )
