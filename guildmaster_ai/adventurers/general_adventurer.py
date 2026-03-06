from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.messages import QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.llm.base_provider import LLMMessage

_MAX_TOOL_ITERATIONS = 10


class GeneralAdventurer(BaseAdventurer):
    """A general-purpose adventurer capable of handling a wide range of tasks."""

    @property
    def system_prompt(self) -> str:
        return (
            "You are a versatile adventurer capable of handling general tasks. "
            "Use any tools at your disposal to accomplish the quest objective. "
            "Be thorough, accurate, and concise in your responses."
        )

    async def execute(self, quest: Quest) -> QuestResult:
        """Execute a quest using an LLM call loop with tool handling."""
        self._reset_conversation()
        self._conversation.append(LLMMessage(role="user", content=quest.description))

        for _ in range(_MAX_TOOL_ITERATIONS):
            response = await self._call_llm()

            # No tool calls — we have a final answer
            if not response.tool_calls:
                return QuestResult(
                    sender=self.name or self.id,
                    quest_id=quest.id,
                    success=True,
                    summary=response.content,
                )

            # Record the assistant message with tool calls
            self._conversation.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Process each tool call and add tool results
            for tool_call in response.tool_calls:
                tool_result = await self._handle_tool_call(tool_call)
                self._conversation.append(
                    LLMMessage(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call.get("id", ""),
                    )
                )

        # Exhausted iterations — return what we have
        return QuestResult(
            sender=self.name or self.id,
            quest_id=quest.id,
            success=False,
            summary="Reached maximum tool call iterations without a final answer.",
            failure_reason="max_iterations_exceeded",
        )
