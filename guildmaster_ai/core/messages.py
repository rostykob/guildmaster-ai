from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from guildmaster_ai.core.quest import QuestRank


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BaseMessage(BaseModel):
    """Base schema for all inter-agent messages."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=_utcnow)
    sender: str
    receiver: str | None = None


class QuestDraft(BaseModel):
    """Lightweight quest proposal before full Quest creation."""

    title: str
    description: str
    required_talents: list[str] = Field(default_factory=list)
    rank: QuestRank | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)


class QuestClarificationRequest(BaseMessage):
    """Sent when a quest draft needs more information."""

    quest_draft: QuestDraft
    questions: list[str]


class QuestClarificationResponse(BaseMessage):
    """Answers to clarification questions."""

    answers: dict[str, str]


class AdventurerProfile(BaseModel):
    """Public profile of an adventurer."""

    id: str
    name: str
    talents: list[str] = Field(default_factory=list)
    weapons: list[str] = Field(default_factory=list)
    armor: list[str] = Field(default_factory=list)
    available: bool = True


class QuestFeasibilityReport(BaseMessage):
    """Assessment of whether a quest can be staffed."""

    feasible: bool
    matched_adventurers: list[AdventurerProfile] = Field(default_factory=list)
    missing_talents: list[str] = Field(default_factory=list)
    recommended_rank: QuestRank


class QuestResult(BaseMessage):
    """Outcome of a completed (or failed) quest."""

    quest_id: str
    success: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


class QuestObservation(BaseMessage):
    """Tagged observation produced by the librarian after analysing a quest."""

    quest_id: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)


class GuardMetrics(BaseModel):
    """Quantitative evaluation metrics produced by a guard."""

    hallucination: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    toxicity: float = Field(default=0.0, ge=0.0, le=1.0)


class GuardVerdict(BaseMessage):
    """Safety / policy verdict from the guard agent."""

    verdict: Literal["pass", "warn", "block"]
    reason: str
    metrics: GuardMetrics = Field(default_factory=GuardMetrics)
    details: dict[str, Any] = Field(default_factory=dict)
