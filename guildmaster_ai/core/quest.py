from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from guildmaster_ai.core.exceptions import InvalidQuestTransitionError
from guildmaster_ai.core.utils import _utcnow


class QuestRank(enum.IntEnum):
    """Quest difficulty rank, orderable from lowest (F) to highest (S)."""

    F = 0
    E = 1
    D = 2
    C = 3
    B = 4
    A = 5
    S = 6


class QuestStatus(enum.StrEnum):
    """Lifecycle status of a quest."""

    DRAFT = "draft"
    POSTED = "posted"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


VALID_TRANSITIONS: dict[QuestStatus, set[QuestStatus]] = {
    QuestStatus.DRAFT: {QuestStatus.POSTED},
    QuestStatus.POSTED: {QuestStatus.ASSIGNED, QuestStatus.ARCHIVED},
    QuestStatus.ASSIGNED: {QuestStatus.IN_PROGRESS, QuestStatus.POSTED},
    QuestStatus.IN_PROGRESS: {QuestStatus.COMPLETED, QuestStatus.FAILED},
    QuestStatus.COMPLETED: {QuestStatus.ARCHIVED},
    QuestStatus.FAILED: {QuestStatus.POSTED, QuestStatus.ARCHIVED},
    QuestStatus.ARCHIVED: set(),
}


class QuestHistoryEntry(BaseModel):
    """Single event in a quest's audit trail."""

    timestamp: datetime = Field(default_factory=_utcnow)
    actor: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Quest(BaseModel):
    """Core quest domain object."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    required_talents: list[str] = Field(default_factory=list)
    rank: QuestRank = QuestRank.E
    status: QuestStatus = QuestStatus.DRAFT
    acceptance_criteria: list[str] = Field(default_factory=list)
    assigned_party_id: str | None = None
    result: dict[str, Any] | None = None
    history: list[QuestHistoryEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Composite quest fields
    parent_quest_id: str | None = None
    subtask_index: int | None = None
    is_composite: bool = False

    def transition(
        self,
        new_status: QuestStatus,
        actor: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Move the quest to *new_status*, validating against VALID_TRANSITIONS."""
        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidQuestTransitionError(self.status.value, new_status.value)

        old_status = self.status
        self.status = new_status
        self.updated_at = _utcnow()
        self.add_history(
            actor=actor,
            event_type="transition",
            payload={
                "from": old_status.value,
                "to": new_status.value,
                **(payload or {}),
            },
        )

    def add_history(
        self,
        actor: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append an entry to the quest's history log."""
        self.history.append(
            QuestHistoryEntry(
                actor=actor,
                event_type=event_type,
                payload=payload or {},
            )
        )

    @property
    def is_subtask(self) -> bool:
        """Return True if this quest is a subtask of a composite quest."""
        return self.parent_quest_id is not None

    @property
    def is_terminal(self) -> bool:
        """Return True if the quest is in a terminal state."""
        return self.status in {QuestStatus.COMPLETED, QuestStatus.FAILED, QuestStatus.ARCHIVED}
