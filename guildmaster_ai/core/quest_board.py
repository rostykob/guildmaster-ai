from __future__ import annotations

import logging

from guildmaster_ai.core.exceptions import QuestBoardEmptyError
from guildmaster_ai.core.quest import Quest, QuestStatus

logger = logging.getLogger("guildmaster.quest_board")


class QuestBoard:
    """In-memory priority queue of quests."""

    def __init__(self) -> None:
        self._quests: dict[str, Quest] = {}

    def post(self, quest: Quest) -> Quest:
        """Validate that *quest* is DRAFT, transition to POSTED, and add to the board."""
        if quest.status != QuestStatus.DRAFT:
            raise ValueError(f"Only DRAFT quests can be posted, got {quest.status.value!r}")
        quest.transition(QuestStatus.POSTED, actor="quest_board")
        self._quests[quest.id] = quest
        logger.debug("Quest posted: %s (%r)", quest.id[:8], quest.title)
        return quest

    def assign(self, quest_id: str, party_id: str, actor: str) -> Quest:
        """Assign a posted quest to a party."""
        quest = self.get(quest_id)
        quest.transition(QuestStatus.ASSIGNED, actor=actor, payload={"party_id": party_id})
        quest.assigned_party_id = party_id
        logger.debug("Quest %s assigned to party %s", quest_id[:8], party_id[:8])
        return quest

    def get_posted(self, talents: list[str] | None = None) -> list[Quest]:
        """Return posted quests, optionally filtered by required talents.

        Results are sorted by rank descending, then created_at ascending.
        """
        posted = [q for q in self._quests.values() if q.status == QuestStatus.POSTED]

        if talents is not None:
            talent_set = set(talents)
            posted = [q for q in posted if set(q.required_talents).issubset(talent_set)]

        posted.sort(key=lambda q: (-q.rank.value, q.created_at))
        return posted

    def get(self, quest_id: str) -> Quest:
        """Retrieve a quest by id or raise."""
        try:
            return self._quests[quest_id]
        except KeyError as err:
            if not self._quests:
                raise QuestBoardEmptyError from err
            raise KeyError(f"Quest {quest_id!r} not found on the board") from err

    def remove(self, quest_id: str) -> Quest:
        """Remove and return a quest from the board."""
        try:
            return self._quests.pop(quest_id)
        except KeyError as err:
            raise KeyError(f"Quest {quest_id!r} not found on the board") from err
