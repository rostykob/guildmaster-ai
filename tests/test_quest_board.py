"""Tests for QuestBoard."""

from __future__ import annotations

import pytest

from guildmaster_ai.core.quest import Quest, QuestRank, QuestStatus
from guildmaster_ai.core.quest_board import QuestBoard


class TestQuestBoard:
    def test_post_quest(self) -> None:
        board = QuestBoard()
        quest = Quest(
            title="Find treasure",
            description="Locate the hidden treasure",
            required_talents=["exploration"],
            rank=QuestRank.D,
            acceptance_criteria=["Treasure found"],
        )
        board.post(quest)
        assert quest.status == QuestStatus.POSTED
        assert board.get(quest.id) is not None

    def test_get_posted_quests(self) -> None:
        board = QuestBoard()
        q1 = Quest(
            title="Q1",
            description="d1",
            required_talents=["web_search"],
            rank=QuestRank.E,
            acceptance_criteria=[],
        )
        q2 = Quest(
            title="Q2",
            description="d2",
            required_talents=["coding"],
            rank=QuestRank.C,
            acceptance_criteria=[],
        )
        board.post(q1)
        board.post(q2)

        posted = board.get_posted()
        assert len(posted) == 2
        # Higher rank first
        assert posted[0].rank >= posted[1].rank

    def test_get_posted_by_talent(self) -> None:
        board = QuestBoard()
        q1 = Quest(
            title="Q1",
            description="d1",
            required_talents=["web_search"],
            rank=QuestRank.E,
            acceptance_criteria=[],
        )
        q2 = Quest(
            title="Q2",
            description="d2",
            required_talents=["coding"],
            rank=QuestRank.C,
            acceptance_criteria=[],
        )
        board.post(q1)
        board.post(q2)

        web_quests = board.get_posted(talents=["web_search"])
        assert len(web_quests) == 1
        assert web_quests[0].title == "Q1"

    def test_assign_quest(self) -> None:
        board = QuestBoard()
        quest = Quest(
            title="Q1",
            description="d1",
            required_talents=["general"],
            rank=QuestRank.E,
            acceptance_criteria=[],
        )
        board.post(quest)
        board.assign(quest.id, "party-1", "guildmaster")
        assert quest.status == QuestStatus.ASSIGNED
        assert quest.assigned_party_id == "party-1"

    def test_remove_quest(self) -> None:
        board = QuestBoard()
        quest = Quest(
            title="Q1",
            description="d1",
            required_talents=["general"],
            rank=QuestRank.E,
            acceptance_criteria=[],
        )
        board.post(quest)
        removed = board.remove(quest.id)
        assert removed.id == quest.id
        with pytest.raises((KeyError, Exception)):
            board.get(quest.id)
