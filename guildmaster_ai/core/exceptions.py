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


class QuestDecompositionError(GuildmasterError):
    """Raised when quest planning or decomposition fails."""


class InvalidQuestTransitionError(GuildmasterError):
    """An illegal status transition was attempted."""

    def __init__(self, from_status: str, to_status: str) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(f"Invalid transition from {from_status!r} to {to_status!r}")


class ArmorBlockedError(GuildmasterError):
    """Armor blocked the input or output during quest execution."""

    def __init__(self, armor_name: str, message: str) -> None:
        self.armor_name = armor_name
        self.message = message
        super().__init__(f"Armor {armor_name!r} blocked: {message}")


class ScrollValidationError(GuildmasterError):
    """A scroll failed validation."""


class AdventurerDefeatedError(GuildmasterError):
    """An adventurer ran out of AP (tool calls) or HP (error retries) mid-quest.

    ``stat`` is ``"ap"`` when the tool-call budget was exhausted and ``"hp"``
    when too many tool errors occurred.
    """

    def __init__(self, stat: str, adventurer: str, detail: str) -> None:
        self.stat = stat
        self.adventurer = adventurer
        self.detail = detail
        super().__init__(f"Adventurer {adventurer!r} defeated ({stat} depleted): {detail}")
