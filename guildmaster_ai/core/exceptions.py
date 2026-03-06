from __future__ import annotations

from typing import Any


class GuildmasterError(Exception):
    """Base exception for all guildmaster-ai domain errors."""


class QuestFailedError(GuildmasterError):
    """A quest could not be completed."""

    def __init__(
        self,
        reason: str,
        partial_results: Any | None = None,
    ) -> None:
        self.reason = reason
        self.partial_results = partial_results
        super().__init__(reason)


class NoEligibleAdventurersError(GuildmasterError):
    """No adventurers match the required talents."""

    def __init__(self, required_talents: list[str]) -> None:
        self.required_talents = required_talents
        super().__init__(f"No adventurers with talents: {required_talents}")


class QuestBoardEmptyError(GuildmasterError):
    """The quest board has no posted quests."""


class InvalidQuestTransitionError(GuildmasterError):
    """An illegal status transition was attempted."""

    def __init__(self, from_status: str, to_status: str) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid transition from {from_status!r} to {to_status!r}"
        )
