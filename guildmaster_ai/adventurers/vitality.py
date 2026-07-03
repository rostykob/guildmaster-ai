"""Vitality — AP/HP enforcement middleware for adventurer agents.

AP (action points) is the number of tool calls an adventurer may make during a
single quest. HP (hit points) is the number of tool errors it may survive.
When either reaches zero the agent run is aborted with
:class:`AdventurerDefeatedError` and the quest fails.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from guildmaster_ai.core.exceptions import AdventurerDefeatedError

logger = logging.getLogger("guildmaster.vitality")


class VitalityMiddleware(AgentMiddleware):
    """Per-quest AP/HP budget enforcement.

    A fresh instance is created for every agent build (one per quest
    execution), so the counters never leak between quests.

    - Every tool call costs 1 AP; when AP is exhausted the next tool call
      raises :class:`AdventurerDefeatedError` (``stat="ap"``).
    - Every tool error costs 1 HP; while HP remains the error is returned to
      the model as an error ``ToolMessage`` so it can retry, and when HP is
      exhausted :class:`AdventurerDefeatedError` (``stat="hp"``) is raised.
    """

    # AgentMiddleware declares ``tools`` without a default; vitality registers none.
    tools: Sequence[BaseTool] = []

    def __init__(self, ap: int, hp: int, adventurer: str) -> None:
        super().__init__()
        self._ap = ap
        self._hp = hp
        self._adventurer = adventurer

    @property
    def ap_remaining(self) -> int:
        return self._ap

    @property
    def hp_remaining(self) -> int:
        return self._hp

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        if self._ap <= 0:
            raise AdventurerDefeatedError(
                "ap",
                self._adventurer,
                "tool-call budget exhausted",
            )
        self._ap -= 1
        try:
            return await handler(request)
        except AdventurerDefeatedError:
            raise
        except Exception as exc:
            self._hp -= 1
            logger.warning(
                "Adventurer %s took a hit from tool %r (%s) — HP left: %d",
                self._adventurer,
                request.tool_call.get("name", "?"),
                exc,
                self._hp,
            )
            if self._hp <= 0:
                raise AdventurerDefeatedError(
                    "hp",
                    self._adventurer,
                    f"too many tool errors (last: {exc})",
                ) from exc
            return ToolMessage(
                content=f"Tool error: {exc}",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
