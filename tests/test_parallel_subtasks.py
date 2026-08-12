"""Tests for dependency-aware parallel subtask execution."""

from __future__ import annotations

from typing import Any

import pytest

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.messages import QuestPlan, SubtaskSpec
from guildmaster_ai.core.party import Party
from guildmaster_ai.core.quest import Quest, QuestRank, QuestStatus
from guildmaster_ai.sdk.guild import Guild

from .conftest import MockChatModel

# ── Helpers ──────────────────────────────────────────────────────────


class _StubAdventurer(BaseAdventurer):
    system_prompt = "You are a test adventurer."


def _make_quest(**overrides: Any) -> Quest:
    defaults: dict[str, Any] = {
        "title": "Test",
        "description": "test",
        "required_talents": ["general"],
        "rank": QuestRank.E,
    }
    defaults.update(overrides)
    return Quest(**defaults)


def _make_child_quest(parent: Quest, idx: int, title: str = "") -> Quest:
    child = Quest(
        title=title or f"Subtask {idx}",
        description=f"Do subtask {idx}",
        required_talents=["general"],
        rank=parent.rank,
        parent_quest_id=parent.id,
        subtask_index=idx,
    )
    return child


def _build_guild(llm: MockChatModel) -> Guild:
    """Create a minimal Guild with one adventurer registered."""
    guild = Guild(llm=llm)
    adv = _StubAdventurer(name="Worker", llm=llm)
    guild.register_adventurer(adv)
    return guild


# ── Tests ────────────────────────────────────────────────────────────


class TestWaveExecution:
    @pytest.mark.asyncio
    async def test_independent_subtasks_run_in_parallel(self) -> None:
        """Subtasks with no dependencies should all be in the same wave."""
        llm = MockChatModel(response_content="Done!")
        guild = _build_guild(llm)

        parent = _make_quest(title="Parent")
        children = [_make_child_quest(parent, i) for i in range(3)]

        # Register children so _execute_subtask can find them
        for c in children:
            guild._quests[c.id] = c
            c.transition(QuestStatus.POSTED, "test")
            c.transition(QuestStatus.ASSIGNED, "test")

        plan = QuestPlan(
            sender="guildmaster",
            quest_id=parent.id,
            subtasks=[
                SubtaskSpec(title=f"Subtask {i}", description=f"Do {i}", depends_on=[])
                for i in range(3)
            ],
        )

        party = Party(name="test", leader_id="worker", quest_id=parent.id)
        adv = next(iter(guild._guildmaster._roster.values()))
        for c in children:
            party.assign_subtask(c.id, adv.id)

        results = await guild._run_subtasks_with_deps(children, plan, party)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dependent_subtask_waits(self) -> None:
        """Subtask 2 depends on 0 and 1 — should run in a later wave."""
        llm = MockChatModel(response_content="Done!")
        guild = _build_guild(llm)

        parent = _make_quest(title="Parent")
        children = [_make_child_quest(parent, i) for i in range(3)]

        for c in children:
            guild._quests[c.id] = c
            c.transition(QuestStatus.POSTED, "test")
            c.transition(QuestStatus.ASSIGNED, "test")

        plan = QuestPlan(
            sender="guildmaster",
            quest_id=parent.id,
            subtasks=[
                SubtaskSpec(title="A", description="Do A", depends_on=[]),
                SubtaskSpec(title="B", description="Do B", depends_on=[]),
                SubtaskSpec(title="C", description="Do C (needs A+B)", depends_on=[0, 1]),
            ],
        )

        party = Party(name="test", leader_id="worker", quest_id=parent.id)
        adv = next(iter(guild._guildmaster._roster.values()))
        for c in children:
            party.assign_subtask(c.id, adv.id)

        results = await guild._run_subtasks_with_deps(children, plan, party)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades(self) -> None:
        """If subtask 0 fails, subtask 1 (which depends on 0) should be marked failed."""
        # LLM returns failure-indicating content; we need to make _execute_subtask fail.
        # Easiest: don't assign an adventurer for subtask 0 so it fails with "no adventurer".
        llm = MockChatModel(response_content="Done!")
        guild = _build_guild(llm)

        parent = _make_quest(title="Parent")
        children = [_make_child_quest(parent, i) for i in range(2)]

        for c in children:
            guild._quests[c.id] = c
            c.transition(QuestStatus.POSTED, "test")
            c.transition(QuestStatus.ASSIGNED, "test")

        plan = QuestPlan(
            sender="guildmaster",
            quest_id=parent.id,
            subtasks=[
                SubtaskSpec(title="A", description="Do A", depends_on=[]),
                SubtaskSpec(title="B", description="Needs A", depends_on=[0]),
            ],
        )

        party = Party(name="test", leader_id="worker", quest_id=parent.id)
        # Don't assign subtask 0 to any adventurer → it will fail
        adv = next(iter(guild._guildmaster._roster.values()))
        party.assign_subtask(children[1].id, adv.id)
        # children[0] has no assignment → _execute_subtask gets None adventurer → fail

        results = await guild._run_subtasks_with_deps(children, plan, party)

        assert len(results) == 2
        assert not results[0].success  # Failed: no adventurer
        assert not results[1].success  # Failed: dependency_failed
        assert results[1].failure_reason == "dependency_failed"
