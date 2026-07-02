from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from guildmaster_ai.core.quest import QuestRank
from guildmaster_ai.core.utils import _utcnow


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
    scrolls: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    available: bool = True
    ap: int = 0
    hp: int = 0


class QuestTriage(BaseMessage):
    """Guildmaster triage: difficulty rank + candidate adventurers, one LLM call.

    ``adventurer_ids`` is ordered best-first. An empty list together with an
    ``infeasible_reason`` means no registered adventurer can handle the quest.
    """

    rank: QuestRank = QuestRank.E
    adventurer_ids: list[str] = Field(default_factory=list)
    infeasible_reason: str | None = None


class QuestResult(BaseMessage):
    """Outcome of a completed (or failed) quest."""

    quest_id: str
    success: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    # Full agent message transcript, captured by the executing adventurer.
    # Excluded from inter-agent serialization (it is persisted separately by
    # the guild); carried here only to hand the conversation to the store.
    transcript: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


class QuestTicket(BaseMessage):
    """Acknowledgement returned immediately when a quest is submitted.

    Execution happens in the background — use the ticket's ``quest_id`` with
    ``Receptionist.check_status()`` or ``Guild.wait_for_quest()``.
    """

    quest_id: str
    title: str
    status: str = "submitted"


class QuestStatusReport(BaseMessage):
    """Point-in-time status of a quest, produced by the receptionist."""

    quest_id: str
    title: str
    status: str
    finished: bool = False
    success: bool | None = None
    summary: str | None = None
    failure_reason: str | None = None


class QuestObservation(BaseMessage):
    """Tagged observation produced by the librarian after analysing a quest."""

    quest_id: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)

    @field_validator("tags", "lessons_learned", mode="before")
    @classmethod
    def _coerce_str_items(cls, value: Any) -> list[str]:
        """Coerce list items to strings so real-LLM output never fails validation.

        Models frequently emit tags as dicts (``{"talent": "general"}`` →
        ``"talent:general"``) or mix in non-strings. Normalise at the model
        boundary so no caller can construct an invalid observation.
        """
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                items.extend(f"{k}:{v}" for k, v in item.items())
            elif item is not None:
                items.append(str(item))
        return items


class SubtaskSpec(BaseModel):
    """Specification for a single subtask in a quest decomposition."""

    title: str
    description: str
    required_talents: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(
        default_factory=list,
        description="0-based indices of prerequisite subtasks that must complete first.",
    )
    assignee: str = Field(
        default="",
        description="Adventurer id (or name) chosen by the guildmaster for this subtask.",
    )


class QuestPlan(BaseMessage):
    """Decomposition plan for a complex quest."""

    quest_id: str
    subtasks: list[SubtaskSpec]
    strategy: str = ""
    prior_observations: list[str] = Field(default_factory=list)


class PartyLeaderDecision(BaseMessage):
    """Decision made by the party leader after evaluating subtask results."""

    quest_id: str
    decision: Literal["done", "failed", "retry"]
    reason: str
    retry_subtask_indices: list[int] = Field(default_factory=list)
    combined_summary: str = ""


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
