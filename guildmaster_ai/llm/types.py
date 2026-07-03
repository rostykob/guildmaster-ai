"""Guild-native wrappers around LangChain types.

Users of guildmaster-ai should import from this module (or the ``llm``
package) instead of importing ``langchain_core`` directly.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

__all__ = [
    "AIMessage",
    "BaseMessage",
    "GuildLLM",
    "GuildResponse",
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
    "guild_complete",
]

# ---------------------------------------------------------------------------
# Public type alias — every component that needs an LLM stores this type.
# Users can pass *any* LangChain ``BaseChatModel`` (ChatOpenAI, ChatAnthropic,
# ChatOpenRouter, …) wherever ``GuildLLM`` is expected.
# ---------------------------------------------------------------------------
GuildLLM = BaseChatModel


# ---------------------------------------------------------------------------
# GuildResponse — returned by BaseAdventurer._call_llm() and guild_complete()
# ---------------------------------------------------------------------------
class GuildResponse(BaseModel):
    """Structured response from an LLM call.

    Wraps the underlying LangChain ``AIMessage`` so that downstream code
    never needs to depend on ``langchain_core`` directly.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        """Return ``True`` when the LLM requested tool invocations."""
        return bool(self.tool_calls)

    @classmethod
    def from_ai_message(cls, msg: AIMessage) -> GuildResponse:
        """Build a ``GuildResponse`` from a LangChain ``AIMessage``."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return cls(content=content, tool_calls=[dict(tc) for tc in msg.tool_calls])


# ---------------------------------------------------------------------------
# guild_complete() — lightweight helper for non-adventurer components
# (Receptionist, Librarian) that only need a system + user → text reply.
# ---------------------------------------------------------------------------
async def guild_complete(
    llm: GuildLLM,
    *,
    system: str,
    user: str,
) -> str:
    """Send a system + user message pair to the LLM and return the text reply.

    This is intentionally simple — no tool calling, no conversation history.
    Components that need tool loops should use :class:`BaseAdventurer` instead.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response: AIMessage = await llm.ainvoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)
