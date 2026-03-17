from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from guildmaster_ai.armor.base_armor import BaseArmor
from guildmaster_ai.core.messages import AdventurerProfile, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.llm.types import (
    AIMessage,
    BaseMessage,
    GuildLLM,
    GuildResponse,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from guildmaster_ai.weapons.base_weapon import BaseWeapon

logger = logging.getLogger("guildmaster.adventurer")


class BaseAdventurer(ABC):
    """Abstract base class for all adventurer agents."""

    def __init__(
        self,
        adventurer_id: str | None = None,
        name: str = "",
        llm: GuildLLM | None = None,
        model: str | None = None,
    ) -> None:
        self.id = adventurer_id or str(uuid4())
        self.name = name
        self._llm = llm
        self._model = model
        self._weapons: dict[str, BaseWeapon] = {}
        self._armor: list[BaseArmor] = []
        self._talents: list[str] = []
        self._conversation: list[BaseMessage] = []

    # ── Talents ───────────────────────────────────────────────────────

    @property
    def talents(self) -> list[str]:
        """Return the list of talents assigned to this adventurer."""
        return list(self._talents)

    def grant_talents(self, talents: list[str]) -> None:
        """Add talents to this adventurer (called by guildmaster on registration)."""
        for t in talents:
            if t not in self._talents:
                self._talents.append(t)

    # ── Abstract interface ────────────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this adventurer."""
        ...

    @abstractmethod
    async def execute(self, quest: Quest) -> QuestResult:
        """Execute a quest and return the result."""
        ...

    # ── Equipment ─────────────────────────────────────────────────────

    def equip_weapon(self, weapon: BaseWeapon) -> None:
        """Add a weapon to this adventurer's loadout."""
        self._weapons[weapon.name] = weapon

    def wear_armor(self, armor: BaseArmor) -> None:
        """Add armor to this adventurer's defenses."""
        self._armor.append(armor)

    def profile(self) -> AdventurerProfile:
        """Return the public profile for this adventurer."""
        return AdventurerProfile(
            id=self.id,
            name=self.name,
            talents=self.talents,
            weapons=list(self._weapons.keys()),
            armor=[a.name for a in self._armor],
        )

    # ── Conversation helpers ──────────────────────────────────────────
    # These let subclasses build conversations without importing
    # LangChain message types directly.

    def _reset_conversation(self) -> None:
        """Clear the conversation and re-add the system prompt."""
        self._conversation.clear()
        self._conversation.append(SystemMessage(content=self.system_prompt))

    def _add_user_message(self, content: str) -> None:
        """Append a user (human) message to the conversation."""
        self._conversation.append(HumanMessage(content=content))

    def _add_assistant_response(self, response: GuildResponse) -> None:
        """Append an assistant message (with optional tool calls) to the conversation."""
        self._conversation.append(
            AIMessage(content=response.content, tool_calls=response.tool_calls)
        )

    def _add_tool_result(
        self, *, tool_call_id: str, name: str, content: str
    ) -> None:
        """Append a tool result message to the conversation."""
        self._conversation.append(
            ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
        )

    # ── LLM interaction ───────────────────────────────────────────────

    async def _call_llm(
        self,
        messages: list[BaseMessage] | None = None,
        tools: bool = True,
    ) -> GuildResponse:
        """Call the LLM with the current conversation.

        Runs armor pre-processing on the last user message before the call,
        and armor post-processing on the response content after.

        Returns a :class:`GuildResponse` — subclasses never need to work
        with raw LangChain message types.
        """
        if self._llm is None:
            raise RuntimeError("No LLM configured for this adventurer.")

        conv = messages if messages is not None else self._conversation
        logger.debug("%s calling LLM with %d messages", self.name or self.id, len(conv))

        # Armor pre-processing on the last user message
        for armor in self._armor:
            for msg in reversed(conv):
                if isinstance(msg, HumanMessage):
                    raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                    result = await armor.pre_process(raw)
                    if result.verdict == "block":
                        raise RuntimeError(f"Armor {armor.name} blocked input: {result.message}")
                    if result.modified_content is not None:
                        msg.content = result.modified_content
                    break

        # Bind tools if the adventurer has weapons
        llm = self._llm
        if tools and self._weapons:
            llm = llm.bind_tools(list(self._weapons.values()))

        ai_msg: AIMessage = await llm.ainvoke(conv)
        guild_resp = GuildResponse.from_ai_message(ai_msg)
        logger.debug(
            "%s LLM response: %d chars, %d tool calls",
            self.name or self.id,
            len(guild_resp.content),
            len(guild_resp.tool_calls),
        )

        # Armor post-processing on response content
        for armor in self._armor:
            result = await armor.post_process(guild_resp.content)
            if result.verdict == "block":
                raise RuntimeError(f"Armor {armor.name} blocked output: {result.message}")
            if result.modified_content is not None:
                guild_resp.content = result.modified_content

        return guild_resp

    async def _handle_tool_call(self, tool_call: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate weapon and return the result."""
        name = tool_call.get("name", "")
        arguments = tool_call.get("args", {})

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        weapon = self._weapons.get(name)
        if weapon is None:
            logger.warning("Unknown weapon requested: %s", name)
            return json.dumps({"error": f"Unknown weapon: {name}"})

        logger.info("%s using weapon %r with args %s", self.name or self.id, name, arguments)
        result = await weapon.execute(**arguments)
        logger.debug("Weapon %r result: %s", name, str(result)[:200])
        return json.dumps(result)
