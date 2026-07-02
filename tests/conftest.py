"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from guildmaster_ai.core.messages import QuestDraft
from guildmaster_ai.core.quest import Quest, QuestRank


class MockChatModel(BaseChatModel):  # type: ignore[misc]
    """Mock LangChain chat model for testing."""

    response_content: str = "Mock response"
    responses: list[str] = Field(default_factory=list)
    mock_tool_calls: list[dict[str, Any]] | None = None
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1

        # Use responses list if available, else fall back to response_content
        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]
        else:
            content = self.response_content

        # Return tool calls only on the first call (mimics a single tool round)
        tool_calls = self.mock_tool_calls if self.call_count == 1 and self.mock_tool_calls else []
        msg = AIMessage(
            content=content,
            tool_calls=tool_calls or [],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> MockChatModel:
        """Return self — tools are ignored in the mock."""
        return self


@pytest.fixture(autouse=True)
def isolated_guild_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point every Guild's persistent storage at a fresh temp directory.

    The persistence layer keys off ``GuildSettings.guild_home`` (default
    ``./.guildmaster``). Without isolation, quests/talents/results persist to a
    single on-disk SQLite DB that leaks state across test runs — e.g. restored
    talents make the guild skip LLM talent assessment, desynchronising mocked
    LLM response sequences. Overriding ``GUILD_GUILD_HOME`` per test keeps each
    test hermetic and avoids polluting the working tree.
    """
    monkeypatch.setenv("GUILD_GUILD_HOME", str(tmp_path / "guild_home"))
    # Run archival inline and skip the Chroma vector store so mocked-LLM call
    # sequences stay deterministic and tests don't spin up embedding backends.
    # Dedicated tests exercise the background + vector-store paths explicitly.
    monkeypatch.setenv("GUILD_BACKGROUND_ARCHIVAL", "false")
    monkeypatch.setenv("GUILD_ENABLE_VECTOR_STORE", "false")


@pytest.fixture
def mock_llm() -> MockChatModel:
    return MockChatModel()


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
