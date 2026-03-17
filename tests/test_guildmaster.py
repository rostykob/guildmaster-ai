"""Tests for Guildmaster agent."""

from __future__ import annotations

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.adventurers.guildmaster import Guildmaster
from guildmaster_ai.core.messages import QuestDraft


class TestGuildmaster:
    def test_register_adventurer(self, mock_llm) -> None:
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)
        assert len(gm.roster) == 1

    def test_talents_assigned_on_registration(self, mock_llm) -> None:
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        assert adv.talents == []  # no talents before registration
        gm.register_adventurer(adv)
        assert len(adv.talents) > 0  # talents inferred from system prompt

    def test_talents_include_weapon_names(self, mock_llm) -> None:
        from guildmaster_ai.weapons.web_search import WebSearchWeapon

        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        adv.equip_weapon(WebSearchWeapon())
        gm.register_adventurer(adv)
        assert "web_search" in adv.talents

    def test_check_feasibility_feasible(self, mock_llm) -> None:
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)

        draft = QuestDraft(
            title="Test",
            description="A test",
            required_talents=["general"],
            acceptance_criteria=["Done"],
        )
        report = gm.check_feasibility(draft)
        assert report.feasible is True
        assert len(report.matched_adventurers) >= 1

    def test_check_feasibility_not_feasible(self, mock_llm) -> None:
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)

        # A "general" adventurer can attempt any quest, so an empty roster
        # is needed to get infeasibility.
        gm_empty = Guildmaster(llm=mock_llm)
        draft = QuestDraft(
            title="Test",
            description="A test",
            required_talents=["quantum_computing"],
            acceptance_criteria=["Done"],
        )
        report = gm_empty.check_feasibility(draft)
        assert report.feasible is False
        assert "quantum_computing" in report.missing_talents

    def test_match_adventurers(self, mock_llm, sample_quest) -> None:
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)
        matched = gm.match_adventurers(sample_quest)
        assert len(matched) >= 1
