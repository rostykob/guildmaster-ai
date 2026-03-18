from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from guildmaster_ai.core.utils import _utcnow


class PartyMember(BaseModel):
    """A single member of an adventuring party."""

    adventurer_id: str
    role: str = "member"
    assigned_subtask_ids: list[str] = Field(default_factory=list)


class Party(BaseModel):
    """An adventuring party formed to tackle a quest."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    leader_id: str
    members: list[PartyMember] = Field(default_factory=list)
    quest_id: str | None = None
    formed_at: datetime = Field(default_factory=_utcnow)
    disbanded_at: datetime | None = None
    subtask_assignments: dict[str, str] = Field(default_factory=dict)

    def add_member(self, adventurer_id: str, role: str = "member") -> None:
        """Add a member to the party."""
        self.members.append(PartyMember(adventurer_id=adventurer_id, role=role))

    def disband(self) -> None:
        """Mark the party as disbanded."""
        self.disbanded_at = _utcnow()

    def assign_subtask(self, subtask_quest_id: str, adventurer_id: str) -> None:
        """Assign a subtask quest to an adventurer within the party."""
        self.subtask_assignments[subtask_quest_id] = adventurer_id
        for member in self.members:
            if member.adventurer_id == adventurer_id:
                member.assigned_subtask_ids.append(subtask_quest_id)
                break

    @property
    def member_ids(self) -> list[str]:
        """Return a list of all member adventurer ids."""
        return [m.adventurer_id for m in self.members]
