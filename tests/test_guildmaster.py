"""Tests for Guildmaster agent."""

from __future__ import annotations

import json

import pytest

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.adventurers.guildmaster import Guildmaster
from guildmaster_ai.adventurers.librarian import Librarian
from guildmaster_ai.core.messages import QuestDraft, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.core.quest_board import QuestBoard

from .conftest import MockChatModel


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

    # ── Quest planning & decomposition ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_plan_quest_simple_returns_none(self) -> None:
        """When LLM says quest is simple, plan_quest returns None."""
        mock_llm = MockChatModel(response_content=json.dumps({"decompose": False}))
        gm = Guildmaster(llm=mock_llm)
        quest = Quest(
            title="Simple quest",
            description="Just do one thing",
            required_talents=["general"],
        )
        plan = await gm.plan_quest(quest)
        assert plan is None

    @pytest.mark.asyncio
    async def test_plan_quest_complex_returns_plan(self) -> None:
        """When LLM says quest is complex, plan_quest returns a QuestPlan."""
        mock_llm = MockChatModel(
            response_content=json.dumps(
                {
                    "decompose": True,
                    "strategy": "Divide and conquer",
                    "subtasks": [
                        {
                            "title": "Research",
                            "description": "Research the topic",
                            "required_talents": ["research"],
                            "acceptance_criteria": ["Findings documented"],
                        },
                        {
                            "title": "Write report",
                            "description": "Write a report",
                            "required_talents": ["writing"],
                            "acceptance_criteria": ["Report complete"],
                        },
                    ],
                }
            )
        )
        gm = Guildmaster(llm=mock_llm)
        quest = Quest(
            title="Complex quest",
            description="Research and write a report",
            required_talents=["research", "writing"],
        )
        plan = await gm.plan_quest(quest)
        assert plan is not None
        assert len(plan.subtasks) == 2
        assert plan.strategy == "Divide and conquer"
        assert plan.subtasks[0].title == "Research"

    @pytest.mark.asyncio
    async def test_plan_quest_with_librarian_observations(self) -> None:
        """plan_quest includes prior observations from the librarian."""
        mock_llm = MockChatModel(
            response_content=json.dumps(
                {
                    "decompose": True,
                    "strategy": "Use past knowledge",
                    "subtasks": [
                        {
                            "title": "Sub 1",
                            "description": "Do sub 1",
                        },
                        {
                            "title": "Sub 2",
                            "description": "Do sub 2",
                        },
                    ],
                }
            )
        )
        librarian = Librarian(llm=mock_llm)
        # Pre-populate librarian with observations
        from guildmaster_ai.core.messages import QuestObservation

        librarian._observations.append(
            QuestObservation(
                sender="librarian",
                quest_id="old-quest",
                summary="Similar quest succeeded with research approach",
                tags=["success"],
            )
        )

        gm = Guildmaster(llm=mock_llm, librarian=librarian)
        quest = Quest(
            title="Test",
            description="Similar quest to do research",
        )
        plan = await gm.plan_quest(quest)
        assert plan is not None
        assert len(plan.prior_observations) > 0

    @pytest.mark.asyncio
    async def test_plan_quest_no_llm_returns_none(self) -> None:
        """Without an LLM, plan_quest always returns None."""
        gm = Guildmaster(llm=None)
        quest = Quest(title="Test", description="Test")
        plan = await gm.plan_quest(quest)
        assert plan is None

    @pytest.mark.asyncio
    async def test_form_party_for_plan_assigns_subtasks(self) -> None:
        """form_party_for_plan creates child quests and assigns adventurers."""
        mock_llm = MockChatModel()
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)

        from guildmaster_ai.core.messages import QuestPlan, SubtaskSpec

        quest = Quest(
            title="Parent",
            description="Parent quest",
            required_talents=["general"],
        )
        board = QuestBoard()
        board.post(quest)

        plan = QuestPlan(
            sender="guildmaster",
            quest_id=quest.id,
            subtasks=[
                SubtaskSpec(title="Sub A", description="Do A"),
                SubtaskSpec(title="Sub B", description="Do B"),
            ],
        )

        party, child_quests = await gm.form_party_for_plan(
            quest,
            plan,
            board,
        )
        assert len(child_quests) == 2
        assert all(c.parent_quest_id == quest.id for c in child_quests)
        assert child_quests[0].subtask_index == 0
        assert child_quests[1].subtask_index == 1
        assert len(party.subtask_assignments) == 2
        assert party.leader_id != ""

    @pytest.mark.asyncio
    async def test_evaluate_quest_completion_all_success(self) -> None:
        """When all subtasks succeed, decision is 'done'."""
        gm = Guildmaster(llm=None)
        quest = Quest(title="Test", description="Test")
        results = [
            QuestResult(
                sender="adv",
                quest_id="sub-1",
                success=True,
                summary="Done 1",
            ),
            QuestResult(
                sender="adv",
                quest_id="sub-2",
                success=True,
                summary="Done 2",
            ),
        ]
        decision = await gm.evaluate_quest_completion(quest, results)
        assert decision.decision == "done"
        assert "Done 1" in decision.combined_summary
        assert "Done 2" in decision.combined_summary

    @pytest.mark.asyncio
    async def test_evaluate_quest_completion_with_retry(self) -> None:
        """When some subtasks fail, decision is 'retry'."""
        gm = Guildmaster(llm=None)
        quest = Quest(title="Test", description="Test")
        results = [
            QuestResult(
                sender="adv",
                quest_id="sub-1",
                success=True,
                summary="Done 1",
            ),
            QuestResult(
                sender="adv",
                quest_id="sub-2",
                success=False,
                summary="Failed",
                failure_reason="error",
            ),
        ]
        decision = await gm.evaluate_quest_completion(quest, results)
        assert decision.decision == "retry"
        assert 1 in decision.retry_subtask_indices
