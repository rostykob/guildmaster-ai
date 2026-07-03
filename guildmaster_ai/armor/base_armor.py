from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from guildmaster_ai.core.exceptions import ArmorBlockedError

ArmorVerdict = Literal["pass", "warn", "block"]


class ArmorResult(BaseModel):
    """Result returned by an armor check."""

    verdict: ArmorVerdict
    message: str | None = None
    modified_content: str | None = None


class BaseArmor(AgentMiddleware, ABC):
    """Abstract base class for all armor (guardrails).

    Armor pieces double as LangChain ``AgentMiddleware``.  The framework
    calls :meth:`pre_process` on the last human message before the agent
    starts and :meth:`post_process` on the final AI message after the
    agent finishes.

    Subclasses override :meth:`pre_process` and/or :meth:`post_process` —
    the middleware hooks are wired automatically.
    """

    # AgentMiddleware declares ``tools`` without a default; armor registers none.
    tools: Sequence[BaseTool] = []

    @property
    @abstractmethod
    def name(self) -> str: ...

    async def pre_process(self, content: str) -> ArmorResult:
        """Run before the LLM processes the content. Defaults to pass."""
        return ArmorResult(verdict="pass")

    async def post_process(self, content: str) -> ArmorResult:
        """Run after the LLM processes the content. Defaults to pass."""
        return ArmorResult(verdict="pass")

    # ── AgentMiddleware hooks ──────────────────────────────────────────

    async def abefore_agent(
        self, state: Any, runtime: Any
    ) -> dict[str, Any] | None:
        """Apply :meth:`pre_process` to the last human message."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                result = await self.pre_process(raw)
                if result.verdict == "block":
                    raise ArmorBlockedError(
                        self.name, result.message or "Input blocked"
                    )
                if result.modified_content is not None:
                    msg.content = result.modified_content
                break
        return None

    async def aafter_agent(
        self, state: Any, runtime: Any
    ) -> dict[str, Any] | None:
        """Apply :meth:`post_process` to the last AI message."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                result = await self.post_process(raw)
                if result.verdict == "block":
                    raise ArmorBlockedError(
                        self.name, result.message or "Output blocked"
                    )
                if result.modified_content is not None:
                    msg.content = result.modified_content
                break
        return None
