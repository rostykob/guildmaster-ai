from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from guildmaster_ai.armor.base_armor import BaseArmor
from guildmaster_ai.core.messages import AdventurerProfile, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse
from guildmaster_ai.weapons.base_weapon import BaseWeapon


class BaseAdventurer(ABC):
    """Abstract base class for all adventurer agents."""

    def __init__(
        self,
        adventurer_id: str | None = None,
        name: str = "",
        llm_provider: BaseLLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self.id = adventurer_id or str(uuid4())
        self.name = name
        self._llm = llm_provider
        self._model = model
        self._weapons: dict[str, BaseWeapon] = {}
        self._armor: list[BaseArmor] = []
        self._talents: list[str] = []
        self._conversation: list[LLMMessage] = []

    @property
    def talents(self) -> list[str]:
        """Return the list of talents assigned to this adventurer."""
        return list(self._talents)

    def grant_talents(self, talents: list[str]) -> None:
        """Add talents to this adventurer (called by guildmaster on registration)."""
        for t in talents:
            if t not in self._talents:
                self._talents.append(t)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this adventurer."""
        ...

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
        )

    @abstractmethod
    async def execute(self, quest: Quest) -> QuestResult:
        """Execute a quest and return the result."""
        ...

    async def _call_llm(
        self,
        messages: list[LLMMessage] | None = None,
        tools: bool = True,
    ) -> LLMResponse:
        """Call the LLM provider with the current conversation.

        Runs armor pre-processing on the last user message before the call,
        and armor post-processing on the response content after.
        """
        if self._llm is None:
            raise RuntimeError("No LLM provider configured for this adventurer.")

        conv = messages if messages is not None else self._conversation

        # Armor pre-processing on the last user message
        for armor in self._armor:
            for msg in reversed(conv):
                if msg.role == "user":
                    result = await armor.pre_process(msg.content)
                    if result.verdict == "block":
                        raise RuntimeError(f"Armor {armor.name} blocked input: {result.message}")
                    if result.modified_content is not None:
                        msg.content = result.modified_content
                    break

        tool_specs = self._get_tool_specs() if tools and self._weapons else None

        response = await self._llm.complete(
            conv,
            model=self._model,
            tools=tool_specs,
        )

        # Armor post-processing on response content
        for armor in self._armor:
            result = await armor.post_process(response.content)
            if result.verdict == "block":
                raise RuntimeError(f"Armor {armor.name} blocked output: {result.message}")
            if result.modified_content is not None:
                response.content = result.modified_content

        return response

    def _get_tool_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool specs for all equipped weapons."""
        return [weapon.to_tool_spec() for weapon in self._weapons.values()]

    async def _handle_tool_call(self, tool_call: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate weapon and return the result."""
        function_info = tool_call.get("function", {})
        name = function_info.get("name", "")
        arguments = function_info.get("arguments", "{}")

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        weapon = self._weapons.get(name)
        if weapon is None:
            return json.dumps({"error": f"Unknown weapon: {name}"})

        result = await weapon.execute(**arguments)
        return json.dumps(result)

    def _reset_conversation(self) -> None:
        """Clear the conversation and re-add the system prompt."""
        self._conversation.clear()
        self._conversation.append(LLMMessage(role="system", content=self.system_prompt))
