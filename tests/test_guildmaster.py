"""Tests for Guildmaster agent."""

from __future__ import annotations

import json

import pytest

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.adventurers.guildmaster import Guildmaster
from guildmaster_ai.core.messages import QuestDraft, QuestResult
from guildmaster_ai.core.quest import Quest, QuestRank
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
    async def test_plan_quest_with_chronicle(self) -> None:
        """plan_quest carries the chronicle summary as prior observation context."""
        mock_llm = MockChatModel(
            response_content=json.dumps(
                {
                    "decompose": True,
                    "strategy": "Use past knowledge",
                    "subtasks": [
                        {"title": "Sub 1", "description": "Do sub 1"},
                        {"title": "Sub 2", "description": "Do sub 2"},
                    ],
                }
            )
        )
        gm = Guildmaster(llm=mock_llm)
        quest = Quest(
            title="Test",
            description="Similar quest to do research",
        )
        chronicle = "- Research quests succeed with the research approach"
        plan = await gm.plan_quest(quest, chronicle=chronicle)
        assert plan is not None
        assert plan.prior_observations == [chronicle]

    @pytest.mark.asyncio
    async def test_plan_quest_assigns_subtask_assignees(self) -> None:
        """Planned assignees are parsed and honored by form_party_for_plan."""
        mock_llm = MockChatModel()
        gm = Guildmaster(llm=mock_llm)
        adv_a = GeneralAdventurer(name="Alpha", llm=mock_llm)
        adv_b = GeneralAdventurer(name="Beta", llm=mock_llm)
        gm.register_adventurer(adv_a)
        gm.register_adventurer(adv_b)
        mock_llm.response_content = json.dumps(
            {
                "decompose": True,
                "strategy": "split",
                "subtasks": [
                    {"title": "Sub 1", "description": "Do 1", "assignee": adv_b.id},
                    {"title": "Sub 2", "description": "Do 2", "assignee": "Alpha"},
                ],
            }
        )
        quest = Quest(title="Parent", description="Two parts")
        plan = await gm.plan_quest(quest)
        assert plan is not None
        assert plan.subtasks[0].assignee == adv_b.id
        assert plan.subtasks[1].assignee == "Alpha"

        board = QuestBoard()
        board.post(quest)
        party, children = await gm.form_party_for_plan(quest, plan, board)
        assert party.subtask_assignments[children[0].id] == adv_b.id
        assert party.subtask_assignments[children[1].id] == adv_a.id

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

    # ── LLM-based talent assessment ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_assess_talents(self) -> None:
        """LLM-based talent assessment returns parsed talents."""
        mock_llm = MockChatModel(response_content='["coding", "research", "writing"]')
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        talents = await gm._llm_assess_talents(adv)
        assert "coding" in talents
        assert "research" in talents
        assert "writing" in talents

    @pytest.mark.asyncio
    async def test_llm_assess_talents_fallback_on_empty(self) -> None:
        """Falls back to keyword matching when LLM returns empty result."""
        mock_llm = MockChatModel(response_content="not valid json")
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        talents = await gm._llm_assess_talents(adv)
        # Should fall back to keyword-based — GeneralAdventurer has "general" and "versatile"
        assert len(talents) > 0

    # ── Quest triage ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_triage_returns_rank_and_candidates_by_id(self) -> None:
        """triage_quest parses rank and resolves adventurer ids in one call."""
        mock_llm = MockChatModel()
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)
        mock_llm.response_content = (
            f'{{"rank": "B", "adventurers": ["{adv.id}"], "infeasible_reason": null}}'
        )

        triage = await gm.triage_quest(QuestDraft(title="Build stuff", description="Two parts"))
        assert triage.rank is QuestRank.B
        assert triage.adventurer_ids == [adv.id]
        assert triage.infeasible_reason is None

    @pytest.mark.asyncio
    async def test_triage_resolves_adventurers_by_name(self) -> None:
        mock_llm = MockChatModel(
            response_content='{"rank": "E", "adventurers": ["Scout"]}'
        )
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(name="Scout", llm=mock_llm)
        gm.register_adventurer(adv)

        triage = await gm.triage_quest(QuestDraft(title="x", description="y"))
        assert triage.adventurer_ids == [adv.id]

    @pytest.mark.asyncio
    async def test_triage_empty_roster_is_infeasible_without_llm_call(self) -> None:
        mock_llm = MockChatModel()
        gm = Guildmaster(llm=mock_llm)
        triage = await gm.triage_quest(QuestDraft(title="x", description="y"))
        assert triage.adventurer_ids == []
        assert triage.infeasible_reason is not None
        assert mock_llm.call_count == 0

    @pytest.mark.asyncio
    async def test_triage_falls_back_to_full_roster_on_bad_json(self) -> None:
        """Unparseable responses use the full roster at an easy rank."""
        mock_llm = MockChatModel(response_content="not json")
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)

        triage = await gm.triage_quest(QuestDraft(title="x", description="y"))
        assert triage.rank is QuestRank.E
        assert triage.adventurer_ids == [adv.id]

    @pytest.mark.asyncio
    async def test_triage_unknown_names_fall_back_to_roster(self) -> None:
        """Hallucinated adventurer names must not brick the quest."""
        mock_llm = MockChatModel(
            response_content='{"rank": "E", "adventurers": ["Gandalf"]}'
        )
        gm = Guildmaster(llm=mock_llm)
        adv = GeneralAdventurer(llm=mock_llm)
        gm.register_adventurer(adv)

        triage = await gm.triage_quest(QuestDraft(title="x", description="y"))
        assert triage.adventurer_ids == [adv.id]
        assert triage.infeasible_reason is None

    @pytest.mark.asyncio
    async def test_triage_honors_infeasible_reason(self) -> None:
        mock_llm = MockChatModel(
            response_content='{"rank": "S", "adventurers": [], '
            '"infeasible_reason": "Requires quantum computing"}'
        )
        gm = Guildmaster(llm=mock_llm)
        gm.register_adventurer(GeneralAdventurer(llm=mock_llm))

        triage = await gm.triage_quest(QuestDraft(title="x", description="y"))
        assert triage.adventurer_ids == []
        assert triage.infeasible_reason == "Requires quantum computing"
