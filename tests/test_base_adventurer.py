"""Tests for BaseAdventurer."""

from __future__ import annotations

from typing import Any

import pytest

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.armor.base_armor import ArmorResult, BaseArmor
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.weapons.base_weapon import BaseWeapon

from .conftest import MockChatModel

# ── Helpers ──────────────────────────────────────────────────────────────


class SimpleAdventurer(BaseAdventurer):
    system_prompt = "You are a test adventurer."


class PropertyPromptAdventurer(BaseAdventurer):
    @property
    def system_prompt(self) -> str:
        return f"You are {self.name or 'unnamed'}."


class EmptyPromptAdventurer(BaseAdventurer):
    pass  # system_prompt left as ""


class _DummyWeaponInput(BaseWeapon):
    """Minimal weapon for testing."""

    name: str = "dummy"
    description: str = "A dummy weapon"

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": "dummy_result"}


class _BlockingArmor(BaseArmor):
    """Armor that blocks everything."""

    _block_input: bool
    _block_output: bool

    def __init__(self, *, block_input: bool = False, block_output: bool = False) -> None:
        self._block_input = block_input
        self._block_output = block_output

    @property
    def name(self) -> str:
        return "blocking_armor"

    async def pre_process(self, content: str) -> ArmorResult:
        if self._block_input:
            return ArmorResult(verdict="block", message="Input blocked")
        return ArmorResult(verdict="pass")

    async def post_process(self, content: str) -> ArmorResult:
        if self._block_output:
            return ArmorResult(verdict="block", message="Output blocked")
        return ArmorResult(verdict="pass")


class _ModifyingArmor(BaseArmor):
    """Armor that modifies content."""

    @property
    def name(self) -> str:
        return "modifying_armor"

    async def pre_process(self, content: str) -> ArmorResult:
        return ArmorResult(verdict="pass", modified_content=content.upper())

    async def post_process(self, content: str) -> ArmorResult:
        return ArmorResult(verdict="pass", modified_content=content + " [reviewed]")


def _make_quest(**overrides: Any) -> Quest:
    defaults: dict[str, Any] = {
        "title": "Test Quest",
        "description": "Do the thing.",
        "required_talents": ["general"],
        "rank": QuestRank.E,
    }
    defaults.update(overrides)
    return Quest(**defaults)


# ── Construction / validation ────────────────────────────────────────────


class TestConstruction:
    def test_class_var_system_prompt(self) -> None:
        adv = SimpleAdventurer(name="Alice")
        assert adv.system_prompt == "You are a test adventurer."
        assert adv.name == "Alice"

    def test_property_system_prompt(self) -> None:
        adv = PropertyPromptAdventurer(name="Bob")
        assert adv.system_prompt == "You are Bob."

    def test_missing_system_prompt_raises(self) -> None:
        with pytest.raises(TypeError, match="system_prompt"):
            EmptyPromptAdventurer()

    def test_auto_generated_id(self) -> None:
        adv = SimpleAdventurer()
        assert adv.id  # non-empty UUID string

    def test_custom_id(self) -> None:
        adv = SimpleAdventurer(adventurer_id="custom-123")
        assert adv.id == "custom-123"


# ── Talents ──────────────────────────────────────────────────────────────


class TestTalents:
    def test_grant_talents(self) -> None:
        adv = SimpleAdventurer()
        adv.grant_talents(["coding", "research"])
        assert "coding" in adv.talents
        assert "research" in adv.talents

    def test_grant_talents_no_duplicates(self) -> None:
        adv = SimpleAdventurer()
        adv.grant_talents(["coding", "coding", "research"])
        assert adv.talents.count("coding") == 1

    def test_talents_returns_copy(self) -> None:
        adv = SimpleAdventurer()
        adv.grant_talents(["coding"])
        talents = adv.talents
        talents.append("hacking")
        assert "hacking" not in adv.talents


# ── Equipment ────────────────────────────────────────────────────────────


class TestEquipment:
    def test_equip_weapon(self) -> None:
        adv = SimpleAdventurer()
        weapon = _DummyWeaponInput()
        adv.equip_weapon(weapon)
        assert "dummy" in adv.weapon_names
        assert "dummy" in adv.weapons

    def test_wear_armor(self) -> None:
        adv = SimpleAdventurer()
        armor = _BlockingArmor(block_input=False)
        adv.wear_armor(armor)
        assert len(adv.armor) == 1
        assert adv.armor[0].name == "blocking_armor"

    def test_profile(self) -> None:
        adv = SimpleAdventurer(name="Hero")
        adv.equip_weapon(_DummyWeaponInput())
        adv.wear_armor(_BlockingArmor(block_input=False))
        adv.grant_talents(["general"])

        profile = adv.profile()
        assert profile.name == "Hero"
        assert "dummy" in profile.weapons
        assert "blocking_armor" in profile.armor
        assert "general" in profile.talents


# ── LLM accessors ───────────────────────────────────────────────────────


class TestLLMAccessor:
    def test_llm_getter_none_by_default(self) -> None:
        adv = SimpleAdventurer()
        assert adv.llm is None

    def test_llm_setter(self) -> None:
        adv = SimpleAdventurer()
        mock = MockChatModel()
        adv.llm = mock
        assert adv.llm is mock


# ── Execute (tool-calling loop) ─────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_no_tools_returns_immediately(self) -> None:
        llm = MockChatModel(response_content="Done!")
        adv = SimpleAdventurer(llm=llm)
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is True
        assert result.summary == "Done!"

    @pytest.mark.asyncio
    async def test_single_tool_round(self) -> None:
        llm = MockChatModel(
            response_content="Used tool.",
            mock_tool_calls=[{"id": "tc1", "name": "dummy", "args": {}}],
        )
        adv = SimpleAdventurer(llm=llm)
        adv.equip_weapon(_DummyWeaponInput())
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_weapon_returns_error(self) -> None:
        llm = MockChatModel(
            response_content="Tried unknown weapon.",
            mock_tool_calls=[{"id": "tc1", "name": "nonexistent", "args": {}}],
        )
        adv = SimpleAdventurer(llm=llm)
        quest = _make_quest()
        result = await adv.execute(quest)
        # Should still succeed because after tool error, LLM returns text
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_llm_raises(self) -> None:
        adv = SimpleAdventurer()
        quest = _make_quest()
        with pytest.raises(RuntimeError, match="No LLM configured"):
            await adv.execute(quest)

    @pytest.mark.asyncio
    async def test_max_iterations(self) -> None:
        """When every call returns tool calls, should hit max iterations."""

        class AlwaysToolsLLM(MockChatModel):
            def _generate(self, *args, **kwargs):
                self.call_count += 1
                from langchain_core.messages import AIMessage
                from langchain_core.outputs import ChatGeneration, ChatResult

                msg = AIMessage(
                    content="Still working...",
                    tool_calls=[{"id": "tc1", "name": "dummy", "args": {}}],
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])

        llm = AlwaysToolsLLM()
        adv = SimpleAdventurer(llm=llm)
        adv.equip_weapon(_DummyWeaponInput())
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is False
        assert result.failure_reason == "max_iterations_exceeded"


# ── Armor integration ────────────────────────────────────────────────────


class TestArmor:
    @pytest.mark.asyncio
    async def test_armor_blocks_input(self) -> None:
        llm = MockChatModel(response_content="Should not reach here")
        adv = SimpleAdventurer(llm=llm)
        adv.wear_armor(_BlockingArmor(block_input=True))
        quest = _make_quest()
        with pytest.raises(RuntimeError, match="blocked input"):
            await adv.execute(quest)

    @pytest.mark.asyncio
    async def test_armor_blocks_output(self) -> None:
        llm = MockChatModel(response_content="Bad output")
        adv = SimpleAdventurer(llm=llm)
        adv.wear_armor(_BlockingArmor(block_output=True))
        quest = _make_quest()
        with pytest.raises(RuntimeError, match="blocked output"):
            await adv.execute(quest)

    @pytest.mark.asyncio
    async def test_armor_modifies_content(self) -> None:
        llm = MockChatModel(response_content="result")
        adv = SimpleAdventurer(llm=llm)
        adv.wear_armor(_ModifyingArmor())
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is True
        assert "[reviewed]" in result.summary
