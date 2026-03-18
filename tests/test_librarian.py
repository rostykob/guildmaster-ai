"""Tests for Librarian agent."""

from __future__ import annotations

import pytest

from guildmaster_ai.adventurers.librarian import Librarian
from guildmaster_ai.core.messages import QuestResult
from guildmaster_ai.core.quest import Quest, QuestRank, QuestStatus


def _make_quest(**overrides) -> Quest:
    defaults = {
        "title": "Test Quest",
        "description": "A test quest",
        "required_talents": ["general"],
        "rank": QuestRank.E,
    }
    defaults.update(overrides)
    return Quest(**defaults)


def _make_result(quest: Quest, success: bool = True) -> QuestResult:
    return QuestResult(
        sender="adventurer",
        quest_id=quest.id,
        success=success,
        summary="Completed." if success else "Failed.",
        failure_reason=None if success else "timeout",
    )


class TestLibrarian:
    @pytest.mark.asyncio
    async def test_archive_produces_observation(self) -> None:
        lib = Librarian()
        quest = _make_quest()
        result = _make_result(quest)
        archive = await lib.archive(quest, result)
        assert "observation" in archive
        assert archive["observation"]["quest_id"] == quest.id

    @pytest.mark.asyncio
    async def test_analyze_tags_success(self) -> None:
        lib = Librarian()
        quest = _make_quest()
        result = _make_result(quest, success=True)
        obs = await lib.analyze_quest(quest, result)
        assert "success" in obs.tags
        assert "talent:general" in obs.tags

    @pytest.mark.asyncio
    async def test_analyze_tags_failure(self) -> None:
        lib = Librarian()
        quest = _make_quest()
        result = _make_result(quest, success=False)
        obs = await lib.analyze_quest(quest, result)
        assert "failure" in obs.tags
        assert "reason:timeout" in obs.tags

    @pytest.mark.asyncio
    async def test_analyze_detects_failures_in_history(self) -> None:
        lib = Librarian()
        quest = _make_quest()
        quest.transition(QuestStatus.POSTED, "receptionist")
        quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        quest.transition(QuestStatus.IN_PROGRESS, "adventurer")
        quest.transition(QuestStatus.FAILED, "adventurer", {"reason": "timeout"})
        # Re-post
        quest.transition(QuestStatus.POSTED, "guildmaster")
        quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        quest.transition(QuestStatus.IN_PROGRESS, "adventurer")
        quest.transition(QuestStatus.COMPLETED, "adventurer")

        result = _make_result(quest, success=True)
        obs = await lib.analyze_quest(quest, result)
        assert "pattern:had_failures" in obs.tags
        assert any("failed" in lesson.lower() for lesson in obs.lessons_learned)

    @pytest.mark.asyncio
    async def test_query_observations_by_tag(self) -> None:
        lib = Librarian()
        q1 = _make_quest(title="Q1", required_talents=["coding"])
        q2 = _make_quest(title="Q2", required_talents=["research"])

        await lib.archive(q1, _make_result(q1))
        await lib.archive(q2, _make_result(q2))

        coding_obs = lib.query_observations(tags=["talent:coding"])
        assert len(coding_obs) == 1
        assert coding_obs[0].quest_id == q1.id

    @pytest.mark.asyncio
    async def test_query_observations_by_quest_id(self) -> None:
        lib = Librarian()
        quest = _make_quest()
        await lib.archive(quest, _make_result(quest))

        obs = lib.query_observations(quest_id=quest.id)
        assert len(obs) == 1

    @pytest.mark.asyncio
    async def test_analyze_batch(self) -> None:
        lib = Librarian()
        pairs = [
            (_make_quest(title=f"Q{i}"), _make_result(_make_quest(title=f"Q{i}")))
            for i in range(3)
        ]
        observations = await lib.analyze_batch(pairs)
        assert len(observations) == 3
