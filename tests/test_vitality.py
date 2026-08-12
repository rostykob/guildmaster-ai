"""Tests for adventurer AP/HP (vitality) budgets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.adventurers.general_hero import GeneralHero
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.weapons.base_weapon import BaseWeapon

from .conftest import MockChatModel


class _NoArgs(BaseModel):
    pass


class SteadyWeapon(BaseWeapon):
    name: str = "steady"
    description: str = "Always succeeds"
    args_schema: type[BaseModel] = _NoArgs

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


class ExplodingWeapon(BaseWeapon):
    name: str = "bomb"
    description: str = "Always fails"
    args_schema: type[BaseModel] = _NoArgs

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")


def _quest() -> Quest:
    return Quest(
        title="Vitality quest",
        description="Use your weapon",
        rank=QuestRank.E,
    )


def _tool_call(name: str) -> dict[str, Any]:
    return {"name": name, "args": {}, "id": "call-1", "type": "tool_call"}


class TestDefaults:
    def test_adventurer_defaults(self) -> None:
        adv = GeneralAdventurer()
        assert adv.ap == GeneralAdventurer.DEFAULT_AP
        assert adv.hp == GeneralAdventurer.DEFAULT_HP

    def test_hero_gets_larger_budget(self) -> None:
        hero = GeneralHero()
        assert hero.ap > GeneralAdventurer.DEFAULT_AP
        assert hero.hp >= GeneralAdventurer.DEFAULT_HP

    def test_custom_values_survive_spawn(self) -> None:
        adv = GeneralAdventurer(name="Scout", ap=5, hp=1)
        clone = adv.spawn()
        assert clone.ap == 5
        assert clone.hp == 1

    def test_profile_reports_vitality(self) -> None:
        adv = GeneralAdventurer(ap=7, hp=2)
        profile = adv.profile()
        assert profile.ap == 7
        assert profile.hp == 2


class TestApDepletion:
    async def test_zero_ap_fails_quest_on_first_tool_call(self) -> None:
        llm = MockChatModel(
            response_content="Using tool.",
            mock_tool_calls=[_tool_call("steady")],
        )
        adv = GeneralAdventurer(llm=llm, ap=0, hp=3)
        adv.equip_weapon(SteadyWeapon())

        result = await adv.execute(_quest())
        assert result.success is False
        assert result.failure_reason == "ap_depleted"

    async def test_sufficient_ap_completes(self) -> None:
        llm = MockChatModel(
            response_content="Finished.",
            mock_tool_calls=[_tool_call("steady")],
        )
        adv = GeneralAdventurer(llm=llm, ap=5, hp=3)
        adv.equip_weapon(SteadyWeapon())

        result = await adv.execute(_quest())
        assert result.success is True


class TestHpDepletion:
    async def test_tool_error_with_one_hp_fails_quest(self) -> None:
        llm = MockChatModel(
            response_content="Using tool.",
            mock_tool_calls=[_tool_call("bomb")],
        )
        adv = GeneralAdventurer(llm=llm, ap=5, hp=1)
        adv.equip_weapon(ExplodingWeapon())

        result = await adv.execute(_quest())
        assert result.success is False
        assert result.failure_reason == "hp_depleted"

    async def test_tool_error_survivable_with_hp_left(self) -> None:
        llm = MockChatModel(
            response_content="Recovered without the tool.",
            mock_tool_calls=[_tool_call("bomb")],
        )
        adv = GeneralAdventurer(llm=llm, ap=5, hp=3)
        adv.equip_weapon(ExplodingWeapon())

        # One failure costs 1 HP; the error is fed back to the model, which
        # answers without tools on the next round.
        result = await adv.execute(_quest())
        assert result.success is True
