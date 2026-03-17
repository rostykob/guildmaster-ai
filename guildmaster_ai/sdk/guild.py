from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.adventurers.guard import Guard
from guildmaster_ai.adventurers.guildmaster import Guildmaster
from guildmaster_ai.adventurers.librarian import Librarian
from guildmaster_ai.adventurers.receptionist import Receptionist
from guildmaster_ai.config.settings import GuildSettings
from guildmaster_ai.core.messages import QuestResult
from guildmaster_ai.core.quest import Quest, QuestStatus
from guildmaster_ai.core.quest_board import QuestBoard
from guildmaster_ai.llm.types import GuildLLM


class Guild:
    """Top-level runtime that orchestrates the full quest lifecycle."""

    def __init__(
        self,
        llm: GuildLLM,
        settings: GuildSettings | None = None,
    ) -> None:
        self.settings = settings or GuildSettings()
        self._llm = llm
        self._board = QuestBoard()
        self._guildmaster = Guildmaster(llm=llm)
        self._receptionist = Receptionist(
            llm=llm,
            max_rounds=self.settings.max_clarification_rounds,
        )
        self._librarian = Librarian(llm=llm)
        self._guard: Guard | None = None

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Register an adventurer with the guild, injecting the LLM if needed."""
        if adventurer._llm is None:
            adventurer._llm = self._llm
        self._guildmaster.register_adventurer(adventurer)

    def enable_guard(self) -> None:
        """Activate the guard agent for safety verification."""
        self._guard = Guard(llm=self._llm)

    async def post_quest(self, request: str) -> QuestResult:
        """Full 4-phase quest lifecycle."""
        # Phase 1: Planning
        draft = await self._receptionist.intake(request)
        feasibility = self._guildmaster.check_feasibility(draft)

        if not feasibility.feasible:
            return QuestResult(
                sender="guild",
                quest_id="",
                success=False,
                summary="Quest is not feasible",
                data={"missing_talents": feasibility.missing_talents},
                failure_reason=(
                    f"No adventurers with required talents: {feasibility.missing_talents}"
                ),
            )

        quest = Quest(
            title=draft.title,
            description=draft.description,
            required_talents=draft.required_talents,
            rank=feasibility.recommended_rank,
            acceptance_criteria=draft.acceptance_criteria,
        )

        # Post to board
        self._board.post(quest)

        # Phase 2: Execution
        await self._guildmaster.assign_quest(quest, self._board)
        adventurers = self._guildmaster.match_adventurers(quest)

        if not adventurers:
            quest.transition(
                QuestStatus.FAILED,
                "guildmaster",
                {"reason": "No adventurers matched"},
            )
            return QuestResult(
                sender="guild",
                quest_id=quest.id,
                success=False,
                summary="No adventurers available",
                data={},
                failure_reason="No adventurers matched the quest requirements",
            )

        leader = adventurers[0]
        quest.transition(QuestStatus.IN_PROGRESS, leader.name)
        result = await leader.execute(quest)

        # Phase 3: Verification
        if self._guard:
            verdict = await self._guard.evaluate(result.summary, quest.acceptance_criteria)
            if verdict.verdict == "block":
                quest.transition(QuestStatus.FAILED, "guard", {"reason": verdict.reason})
                result.success = False
                result.failure_reason = f"Blocked by guard: {verdict.reason}"

        accepted = await self._guildmaster.verify_result(quest, result)
        if accepted:
            quest.transition(QuestStatus.COMPLETED, "guildmaster")
        else:
            quest.transition(QuestStatus.FAILED, "guildmaster", {"reason": "Verification failed"})

        # Phase 4: Archival
        await self._librarian.archive(quest, result)
        quest.transition(QuestStatus.ARCHIVED, "librarian")

        return result

    @property
    def quest_board(self) -> QuestBoard:
        """Access the quest board."""
        return self._board

    @property
    def roster(self):
        """List registered adventurer profiles."""
        return self._guildmaster.roster
