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


# ── Execute (agent-based) ──────────────────────────────────────────────────


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
    async def test_execute_no_state_leakage(self) -> None:
        """Two sequential executions on the same adventurer don't leak state."""
        llm = MockChatModel(response_content="Done!")
        adv = SimpleAdventurer(llm=llm)
        quest1 = _make_quest(title="Quest 1", description="First quest")
        quest2 = _make_quest(title="Quest 2", description="Second quest")

        result1 = await adv.execute(quest1)
        result2 = await adv.execute(quest2)

        assert result1.success is True
        assert result2.success is True
        # Both should succeed independently — no state carried over
        assert result1.quest_id != result2.quest_id

    @pytest.mark.asyncio
    async def test_spawn_creates_independent_instance(self) -> None:
        """spawn() returns a new instance with the same config but different id."""
        llm = MockChatModel(response_content="Done!")
        adv = SimpleAdventurer(name="Hero", llm=llm)
        adv.equip_weapon(_DummyWeaponInput())
        adv.grant_talents(["coding", "research"])

        clone = adv.spawn()
        assert clone.id != adv.id
        assert clone.name == adv.name
        assert clone.llm is adv.llm
        assert clone.weapon_names == adv.weapon_names
        assert clone.talents == adv.talents
        assert isinstance(clone, SimpleAdventurer)


# ── Armor integration ────────────────────────────────────────────────────


class TestArmor:
    @pytest.mark.asyncio
    async def test_armor_blocks_input(self) -> None:
        llm = MockChatModel(response_content="Should not reach here")
        adv = SimpleAdventurer(llm=llm)
        adv.wear_armor(_BlockingArmor(block_input=True))
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is False
        assert result.failure_reason == "armor_blocked"
        assert "blocking_armor" in result.summary

    @pytest.mark.asyncio
    async def test_armor_blocks_output(self) -> None:
        llm = MockChatModel(response_content="Bad output")
        adv = SimpleAdventurer(llm=llm)
        adv.wear_armor(_BlockingArmor(block_output=True))
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is False
        assert result.failure_reason == "armor_blocked"
        assert "blocking_armor" in result.summary

    @pytest.mark.asyncio
    async def test_armor_modifies_content(self) -> None:
        llm = MockChatModel(response_content="result")
        adv = SimpleAdventurer(llm=llm)
        adv.wear_armor(_ModifyingArmor())
        quest = _make_quest()
        result = await adv.execute(quest)
        assert result.success is True
        assert "[reviewed]" in result.summary


# ── _llm_complete convenience method ────────────────────────────────────


class TestLLMComplete:
    @pytest.mark.asyncio
    async def test_llm_complete_returns_text(self) -> None:
        llm = MockChatModel(response_content="Direct reply")
        adv = SimpleAdventurer(llm=llm)
        text = await adv._llm_complete("Hello")
        assert text == "Direct reply"

    @pytest.mark.asyncio
    async def test_llm_complete_custom_system(self) -> None:
        llm = MockChatModel(response_content="Custom system reply")
        adv = SimpleAdventurer(llm=llm)
        text = await adv._llm_complete("Hello", system="Be a pirate.")
        assert text == "Custom system reply"

    @pytest.mark.asyncio
    async def test_llm_complete_no_llm_raises(self) -> None:
        adv = SimpleAdventurer()
        with pytest.raises(RuntimeError, match="No LLM configured"):
            await adv._llm_complete("Hello")


# ── Description ──────────────────────────────────────────────────────────


class DescribedAdventurer(BaseAdventurer):
    system_prompt = "You are a pirate."
    description = "A salty pirate who answers in pirate speak."


class TestDescription:
    def test_default_description_empty(self) -> None:
        adv = SimpleAdventurer()
        assert adv.description == ""
        assert adv.profile().description == ""

    def test_class_var_description(self) -> None:
        adv = DescribedAdventurer()
        assert adv.description == "A salty pirate who answers in pirate speak."
        assert adv.profile().description == adv.description

    def test_constructor_overrides_class_var(self) -> None:
        adv = DescribedAdventurer(description="A retired pirate.")
        assert adv.description == "A retired pirate."

    def test_spawn_preserves_description(self) -> None:
        adv = SimpleAdventurer(description="Instance-level description.")
        clone = adv.spawn()
        assert clone.description == "Instance-level description."

    def test_config_hash_includes_description(self) -> None:
        a = SimpleAdventurer(description="one")
        b = SimpleAdventurer(description="two")
        assert a.config_hash != b.config_hash
