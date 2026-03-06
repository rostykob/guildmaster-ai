"""Tests for Quest domain models."""

from __future__ import annotations

import pytest

from guildmaster_ai.core.exceptions import InvalidQuestTransitionError
from guildmaster_ai.core.quest import Quest, QuestRank, QuestStatus


class TestQuestRank:
    def test_rank_ordering(self) -> None:
        assert QuestRank.F < QuestRank.E < QuestRank.D < QuestRank.C
        assert QuestRank.C < QuestRank.B < QuestRank.A < QuestRank.S

    def test_rank_values(self) -> None:
        assert QuestRank.F == 0
        assert QuestRank.S == 6


class TestQuest:
    def test_create_quest(self, sample_quest: Quest) -> None:
        assert sample_quest.title == "Test Quest"
        assert sample_quest.status == QuestStatus.DRAFT
        assert sample_quest.rank == QuestRank.E
        assert len(sample_quest.history) == 0

    def test_valid_transition(self, sample_quest: Quest) -> None:
        sample_quest.transition(QuestStatus.POSTED, "receptionist")
        assert sample_quest.status == QuestStatus.POSTED
        assert len(sample_quest.history) == 1
        assert sample_quest.history[0].actor == "receptionist"
        assert sample_quest.history[0].event_type == "transition"

    def test_invalid_transition(self, sample_quest: Quest) -> None:
        with pytest.raises(InvalidQuestTransitionError):
            sample_quest.transition(QuestStatus.COMPLETED, "guildmaster")

    def test_transition_chain(self, sample_quest: Quest) -> None:
        sample_quest.transition(QuestStatus.POSTED, "receptionist")
        sample_quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        sample_quest.transition(QuestStatus.IN_PROGRESS, "adventurer")
        sample_quest.transition(QuestStatus.COMPLETED, "adventurer")
        sample_quest.transition(QuestStatus.ARCHIVED, "librarian")
        assert sample_quest.status == QuestStatus.ARCHIVED
        assert len(sample_quest.history) == 5

    def test_failure_path(self, sample_quest: Quest) -> None:
        sample_quest.transition(QuestStatus.POSTED, "receptionist")
        sample_quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        sample_quest.transition(QuestStatus.IN_PROGRESS, "adventurer")
        sample_quest.transition(QuestStatus.FAILED, "adventurer", {"reason": "too hard"})
        assert sample_quest.status == QuestStatus.FAILED
        assert sample_quest.history[-1].payload["reason"] == "too hard"

    def test_is_terminal(self, sample_quest: Quest) -> None:
        assert not sample_quest.is_terminal
        sample_quest.transition(QuestStatus.POSTED, "receptionist")
        sample_quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        sample_quest.transition(QuestStatus.IN_PROGRESS, "adventurer")
        sample_quest.transition(QuestStatus.COMPLETED, "adventurer")
        sample_quest.transition(QuestStatus.ARCHIVED, "librarian")
        assert sample_quest.is_terminal

    def test_add_history(self, sample_quest: Quest) -> None:
        sample_quest.add_history("test_actor", "custom_event", {"key": "value"})
        assert len(sample_quest.history) == 1
        assert sample_quest.history[0].event_type == "custom_event"
