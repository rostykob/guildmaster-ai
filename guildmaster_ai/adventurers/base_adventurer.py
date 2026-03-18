from __future__ import annotations

import json
import logging
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

_MAX_TOOL_ITERATIONS = 10


class BaseAdventurer:
    """Abstract base class for all adventurer agents.

    Subclasses **must** provide a ``system_prompt``.  This can be either a
    plain class variable or a ``@property`` — both patterns are supported::

        # Option 1 — class variable (simplest)
        class MyAdventurer(BaseAdventurer):
            system_prompt = "You are a helpful pirate."

        # Option 2 — property (dynamic prompts)
        class MyAdventurer(BaseAdventurer):
            @property
            def system_prompt(self) -> str:
                return f"You are {self.name}."
    """

    system_prompt: str = ""
    """The system prompt sent to the LLM.  Set as a class variable or property."""

    def __init__(
        self,
        adventurer_id: str | None = None,
        name: str = "",
        llm: GuildLLM | None = None,
    ) -> None:
        # Check for system_prompt — allow property to be resolved after __init__
        has_prompt = isinstance(type(self).__dict__.get("system_prompt"), property) or bool(
            self.system_prompt
        )
        if not has_prompt:
            raise TypeError(
                f"{type(self).__name__} must set a 'system_prompt' class variable or property."
            )
        self.id = adventurer_id or str(uuid4())
        self.name = name
        self._llm = llm
        self._weapons: dict[str, BaseWeapon] = {}
        self._armor: list[BaseArmor] = []
        self._talents: list[str] = []
        self._conversation: list[BaseMessage] = []
        self._logger = logging.getLogger(f"guildmaster.adventurer.{type(self).__name__}")

    # ── Public accessors ────────────────────────────────────────────────

    @property
    def weapons(self) -> dict[str, BaseWeapon]:
        """Return the equipped weapons keyed by name."""
        return dict(self._weapons)

    @property
    def weapon_names(self) -> list[str]:
        """Return a list of equipped weapon names."""
        return list(self._weapons.keys())

    @property
    def armor(self) -> list[BaseArmor]:
        """Return the worn armor pieces."""
        return list(self._armor)

    @property
    def llm(self) -> GuildLLM | None:
        """Return the configured LLM, or ``None``."""
        return self._llm

    @llm.setter
    def llm(self, value: GuildLLM | None) -> None:
        self._llm = value

    # ── Talents ───────────────────────────────────────────────────────

    @property
    def talents(self) -> list[str]:
        """Return the list of talents assigned to this adventurer."""
        return list(self._talents)

    def grant_talents(self, talents: list[str]) -> None:
        """Add talents to this adventurer (called by guildmaster on registration)."""
        added: list[str] = []
        for t in talents:
            if t not in self._talents:
                self._talents.append(t)
                added.append(t)
        if added:
            self._logger.info("Granted talents: %s", added)

    # ── Execute ────────────────────────────────────────────────────────

    async def execute(self, quest: Quest) -> QuestResult:
        """Execute a quest using an LLM tool-calling loop.

        The default implementation handles conversation setup, tool dispatch,
        and iteration — subclasses only need to override this when they have
        genuinely custom execution logic.
        """
        self._logger.info("Starting quest %s: %r", quest.id[:8], quest.title)
        # TODO: spawn a fresh adventurer instance per quest so conversation
        # state doesn't leak across quests — makes this reset unnecessary.
        self._reset_conversation()
        self._add_user_message(quest.description)
        # TODO: let the LLM signal completion via a stop keyword instead
        # of hard-capping at _MAX_TOOL_ITERATIONS iterations.
        for iteration in range(_MAX_TOOL_ITERATIONS):
            self._logger.debug("Iteration %d/%d", iteration + 1, _MAX_TOOL_ITERATIONS)
            response = await self._call_llm()

            if not response.has_tool_calls:
                self._logger.info(
                    "Quest %s completed in %d iteration(s)",
                    quest.id[:8],
                    iteration + 1,
                )
                return QuestResult(
                    sender=self.name or self.id,
                    quest_id=quest.id,
                    success=True,
                    summary=response.content,
                )

            self._add_assistant_response(response)

            for tool_call in response.tool_calls:
                tool_result = await self._handle_tool_call(tool_call)
                self._add_tool_result(
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_call.get("name", ""),
                    content=tool_result,
                )

        self._logger.warning(
            "Quest %s exceeded max iterations (%d)",
            quest.id[:8],
            _MAX_TOOL_ITERATIONS,
        )
        return QuestResult(
            sender=self.name or self.id,
            quest_id=quest.id,
            success=False,
            summary="Reached maximum tool call iterations without a final answer.",
            failure_reason="max_iterations_exceeded",
        )

    # ── Equipment ─────────────────────────────────────────────────────

    def equip_weapon(self, weapon: BaseWeapon) -> None:
        """Add a weapon to this adventurer's loadout."""
        self._weapons[weapon.name] = weapon
        self._logger.info("Equipped weapon: %s", weapon.name)

    def wear_armor(self, armor: BaseArmor) -> None:
        """Add armor to this adventurer's defenses."""
        self._armor.append(armor)
        self._logger.info("Wore armor: %s", armor.name)

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

    def _add_tool_result(self, *, tool_call_id: str, name: str, content: str) -> None:
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
        self._logger.debug("%s calling LLM with %d messages", self.name or self.id, len(conv))

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
        self._logger.debug(
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
            self._logger.warning("Unknown weapon requested: %s", name)
            return json.dumps({"error": f"Unknown weapon: {name}"})

        self._logger.info("%s using weapon %r with args %s", self.name or self.id, name, arguments)
        result = await weapon.execute(**arguments)
        self._logger.debug("Weapon %r result: %s", name, str(result)[:200])
        return json.dumps(result)
