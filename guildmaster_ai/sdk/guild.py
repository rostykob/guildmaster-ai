from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.adventurers.base_guard import BaseGuard
from guildmaster_ai.adventurers.guard import Guard
from guildmaster_ai.adventurers.guildmaster import Guildmaster
from guildmaster_ai.adventurers.librarian import Librarian
from guildmaster_ai.adventurers.receptionist import Receptionist
from guildmaster_ai.config.settings import GuildSettings
from guildmaster_ai.core.messages import AdventurerProfile, QuestResult
from guildmaster_ai.core.quest import Quest, QuestStatus
from guildmaster_ai.core.quest_board import QuestBoard
from guildmaster_ai.llm.types import GuildLLM

logger = logging.getLogger("guildmaster.guild")


class GuildInfo(BaseModel):
    """Snapshot of the guild's current state."""

    id: str
    total_quests: int = 0
    completed: int = 0
    failed: int = 0
    in_progress: int = 0
    archived: int = 0
    adventurers: list[AdventurerProfile] = Field(default_factory=list)
    guard_enabled: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)


class Guild:
    """Top-level runtime that orchestrates the full quest lifecycle."""

    def __init__(
        self,
        llm: GuildLLM,
        settings: GuildSettings | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.settings = settings or GuildSettings()
        self._llm = llm
        self._board = QuestBoard()
        self._guildmaster = Guildmaster(llm=llm)
        self._receptionist = Receptionist(
            llm=llm,
            max_rounds=self.settings.max_clarification_rounds,
        )
        self._librarian = Librarian(llm=llm)
        self._guard: BaseGuard | None = None

        # Configure logging for the guildmaster namespace
        self._configure_logging()

        # Quest registry — every quest ever created, keyed by quest.id.
        # Independent of the board so quests are visible in any state
        # (draft, in_progress, failed, retrying, archived, etc.).
        self._quests: dict[str, Quest] = {}
        self._results: dict[str, QuestResult] = {}

    def _configure_logging(self) -> None:
        """Set up the guildmaster logger based on settings."""
        root = logging.getLogger("guildmaster")
        level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        root.setLevel(level)
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            )
            root.addHandler(handler)

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Register an adventurer with the guild, injecting the LLM if needed."""
        if adventurer._llm is None:
            adventurer._llm = self._llm
        self._guildmaster.register_adventurer(adventurer)

    def enable_guard(self, guard: BaseGuard | None = None) -> None:
        """Activate the guard agent for safety verification.

        Pass a custom :class:`BaseGuard` instance, or omit to use the
        built-in LLM-as-judge :class:`Guard`.
        """
        if guard is not None:
            self._guard = guard
        else:
            self._guard = Guard(llm=self._llm)
        logger.info("Guard enabled: %s", self._guard.name)

    # ── Info ───────────────────────────────────────────────────────────

    def _count_by_status(self, status: QuestStatus) -> int:
        return sum(1 for q in self._quests.values() if q.status == status)

    @property
    def info(self) -> GuildInfo:
        """Return a snapshot of the guild's current state."""
        return GuildInfo(
            id=self.id,
            total_quests=len(self._quests),
            completed=self._count_by_status(QuestStatus.COMPLETED),
            failed=self._count_by_status(QuestStatus.FAILED),
            in_progress=self._count_by_status(QuestStatus.IN_PROGRESS),
            archived=self._count_by_status(QuestStatus.ARCHIVED),
            adventurers=self._guildmaster.roster,
            guard_enabled=self._guard is not None,
            settings={
                "llm_provider": self.settings.llm_provider,
                "llm_default_model": self.settings.llm_default_model,
                "llm_temperature": self.settings.llm_temperature,
                "llm_max_tokens": self.settings.llm_max_tokens,
            },
        )

    # ── Quest lifecycle ────────────────────────────────────────────────

    async def post_quest(self, request: str) -> QuestResult:
        """Full 4-phase quest lifecycle."""
        logger.info("=== New quest request: %s ===", request[:100])

        # Phase 1: Planning
        logger.info("Phase 1: Planning")
        draft = await self._receptionist.intake(request)
        feasibility = self._guildmaster.check_feasibility(draft)

        if not feasibility.feasible:
            logger.warning("Quest not feasible: missing %s", feasibility.missing_talents)
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

        # Register in guild quest registry
        self._quests[quest.id] = quest
        logger.info("Quest created: %s (%r)", quest.id[:8], quest.title)

        # Post to board
        self._board.post(quest)

        # Phase 2: Execution
        logger.info("Phase 2: Execution")
        await self._guildmaster.assign_quest(quest, self._board)
        adventurers = self._guildmaster.match_adventurers(quest)

        if not adventurers:
            logger.warning("No adventurers matched quest %s", quest.id[:8])
            quest.transition(
                QuestStatus.FAILED,
                "guildmaster",
                {"reason": "No adventurers matched"},
            )
            result = QuestResult(
                sender="guild",
                quest_id=quest.id,
                success=False,
                summary="No adventurers available",
                data={},
                failure_reason="No adventurers matched the quest requirements",
            )
            self._results[quest.id] = result
            return result

        leader = adventurers[0]
        logger.info("Quest leader: %s", leader.name or leader.id)
        quest.transition(QuestStatus.IN_PROGRESS, leader.name)
        result = await leader.execute(quest)

        # Phase 3: Verification
        logger.info("Phase 3: Verification")
        if self._guard:
            logger.info("Running guard evaluation (%s)", self._guard.name)
            verdict = await self._guard.evaluate(
                result.summary,
                quest.acceptance_criteria,
                context=quest.description,
            )
            logger.info(
                "Guard verdict: %s (reason: %s)",
                verdict.verdict,
                verdict.reason[:100],
            )
            if verdict.verdict == "block":
                quest.transition(QuestStatus.FAILED, "guard", {"reason": verdict.reason})
                result.success = False
                result.failure_reason = f"Blocked by guard: {verdict.reason}"
        else:
            logger.debug("No guard configured — skipping verification")

        accepted = await self._guildmaster.verify_result(quest, result)
        if accepted:
            quest.transition(QuestStatus.COMPLETED, "guildmaster")
            logger.info("Quest %s completed successfully", quest.id[:8])
        else:
            quest.transition(QuestStatus.FAILED, "guildmaster", {"reason": "Verification failed"})
            logger.warning("Quest %s failed verification", quest.id[:8])

        # Phase 4: Archival
        logger.info("Phase 4: Archival")
        await self._librarian.archive(quest, result)
        quest.transition(QuestStatus.ARCHIVED, "librarian")

        self._results[quest.id] = result
        logger.info("=== Quest %s finished: success=%s ===", quest.id[:8], result.success)
        return result

    # ── Quest lookup ───────────────────────────────────────────────────

    def get_quest(self, quest_id: str) -> Quest:
        """Retrieve a quest by its UUID from the guild registry.

        Works for quests in any state — draft, in_progress, failed, archived, etc.
        Raises ``KeyError`` if the quest is not found.
        """
        try:
            return self._quests[quest_id]
        except KeyError:
            raise KeyError(f"Quest {quest_id!r} not found in this guild") from None

    def get_result(self, quest_id: str) -> QuestResult | None:
        """Return the result for *quest_id*, or ``None`` if not yet available."""
        return self._results.get(quest_id)

    @property
    def quests(self) -> list[Quest]:
        """Return all quests tracked by this guild (newest first)."""
        return sorted(
            self._quests.values(),
            key=lambda q: q.created_at,
            reverse=True,
        )

    @property
    def quest_board(self) -> QuestBoard:
        """Access the quest board."""
        return self._board

    @property
    def roster(self) -> list[AdventurerProfile]:
        """List registered adventurer profiles."""
        return self._guildmaster.roster

    def __repr__(self) -> str:
        i = self.info
        return (
            f"Guild(id={i.id!r}, adventurers={len(i.adventurers)}, "
            f"completed={i.completed}, failed={i.failed}, in_progress={i.in_progress})"
        )
