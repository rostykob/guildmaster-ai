from __future__ import annotations

import asyncio
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
from guildmaster_ai.core.messages import AdventurerProfile, QuestPlan, QuestResult
from guildmaster_ai.core.party import Party
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
        self._librarian = Librarian(llm=llm)
        self._guildmaster = Guildmaster(llm=llm, librarian=self._librarian)
        self._receptionist = Receptionist(
            llm=llm,
            max_rounds=self.settings.max_clarification_rounds,
        )
        self._guard: BaseGuard | None = None
        self._talents_refined = False

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
            handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
            root.addHandler(handler)

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Register an adventurer with the guild, injecting the LLM if needed."""
        if adventurer.llm is None:
            adventurer.llm = self._llm
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
        """Count quests in the given status."""
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
        """Run the full quest lifecycle: Plan → Decompose → Execute → Verify → Archive.

        Simple quests are handled by a single adventurer. Complex quests are
        decomposed into subtasks and distributed across a party.
        """
        logger.info("=== New quest request: %s ===", request[:100])

        # Refine adventurer talents via LLM on first quest
        if not self._talents_refined:
            await self._guildmaster.refine_all_talents()
            self._talents_refined = True

        # Phase 1: Planning
        logger.info("Phase 1: Planning")
        draft = await self._receptionist.intake(request)

        # Guildmaster determines required talents (not the receptionist)
        if not draft.required_talents:
            talents = await self._guildmaster.assess_quest_talents(draft)
            draft.required_talents = talents
            logger.debug("Guildmaster assessed talents: %s", talents)

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

        # Phase 1.5: Decomposition
        plan = await self._guildmaster.plan_quest(quest)
        if plan and len(plan.subtasks) > 1:
            quest.is_composite = True
            logger.info("Quest %s is composite — %d subtasks", quest.id[:8], len(plan.subtasks))
            result = await self._execute_complex_quest(quest, plan)
        else:
            result = await self._execute_simple_quest(quest)

        # Phase 4: Archival
        logger.info("Phase 4: Archival")
        await self._librarian.archive(quest, result)
        quest.transition(QuestStatus.ARCHIVED, "librarian")

        self._results[quest.id] = result
        logger.info("=== Quest %s finished: success=%s ===", quest.id[:8], result.success)
        return result

    async def _execute_simple_quest(self, quest: Quest) -> QuestResult:
        """Execute a quest with a single adventurer (original path)."""
        logger.info("Phase 2: Execution (simple)")
        await self._guildmaster.assign_quest(quest, self._board)
        adventurers = self._guildmaster.match_adventurers(quest)

        if not adventurers:
            logger.warning("No adventurers matched quest %s", quest.id[:8])
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

        leader = adventurers[0].spawn()
        leader_label = leader.name or leader.id
        logger.info("Quest leader: %s", leader_label)
        quest.transition(QuestStatus.IN_PROGRESS, leader_label)
        result = await leader.execute(quest)

        return await self._verify_and_complete(quest, result)

    async def _execute_complex_quest(
        self,
        quest: Quest,
        plan: QuestPlan,
    ) -> QuestResult:
        """Execute a composite quest by distributing subtasks across a party."""
        logger.info("Phase 2: Execution (complex)")
        party, child_quests = await self._guildmaster.form_party_for_plan(
            quest,
            plan,
            self._board,
        )

        # Register child quests
        for child in child_quests:
            self._quests[child.id] = child

        quest.transition(QuestStatus.ASSIGNED, "guildmaster", {"party_id": party.id})
        quest.assigned_party_id = party.id
        quest.transition(QuestStatus.IN_PROGRESS, "guildmaster")

        # Group subtasks by assigned adventurer for sequential execution
        adventurer_tasks: dict[str, list[Quest]] = {}
        for child in child_quests:
            adv_id = party.subtask_assignments.get(child.id, "")
            adventurer_tasks.setdefault(adv_id, []).append(child)

        subtask_results = await self._run_subtasks(adventurer_tasks, party)

        # Evaluate completion
        max_retries = self.settings.max_quest_retries
        for attempt in range(max_retries):
            decision = await self._guildmaster.evaluate_quest_completion(
                quest,
                subtask_results,
            )

            if decision.decision == "done":
                result = QuestResult(
                    sender="guild",
                    quest_id=quest.id,
                    success=True,
                    summary=decision.combined_summary
                    or "\n\n".join(r.summary for r in subtask_results),
                    data={"subtask_count": len(child_quests)},
                )
                return await self._verify_and_complete(quest, result)

            if decision.decision == "failed" or attempt == max_retries - 1:
                quest.transition(QuestStatus.FAILED, "guildmaster")
                return QuestResult(
                    sender="guild",
                    quest_id=quest.id,
                    success=False,
                    summary=decision.reason,
                    data={"subtask_count": len(child_quests)},
                    failure_reason=decision.reason,
                )

            # Retry failed subtasks
            logger.info(
                "Retrying subtasks %s (attempt %d/%d)",
                decision.retry_subtask_indices,
                attempt + 2,
                max_retries,
            )
            retry_tasks: dict[str, list[Quest]] = {}
            for idx in decision.retry_subtask_indices:
                if idx < len(child_quests):
                    child = child_quests[idx]
                    # Only retry subtasks that actually failed
                    if child.status != QuestStatus.FAILED:
                        continue
                    child.transition(QuestStatus.POSTED, "guildmaster")
                    adv_id = party.subtask_assignments.get(child.id, "")
                    # Re-assign on the board
                    self._board.assign(
                        child.id,
                        party.id,
                        actor="guildmaster",
                    )
                    retry_tasks.setdefault(adv_id, []).append(child)

            retry_results = await self._run_subtasks(retry_tasks, party)
            # Update subtask results with retry results
            for idx, new_result in zip(
                decision.retry_subtask_indices,
                retry_results,
                strict=False,
            ):
                if idx < len(subtask_results):
                    subtask_results[idx] = new_result

        # Should not reach here, but just in case
        quest.transition(QuestStatus.FAILED, "guildmaster")
        return QuestResult(
            sender="guild",
            quest_id=quest.id,
            success=False,
            summary="Quest exhausted all retries.",
            failure_reason="max_retries_exceeded",
        )

    async def _run_subtasks(
        self,
        adventurer_tasks: dict[str, list[Quest]],
        _party: Party,
    ) -> list[QuestResult]:
        """Run subtasks grouped by adventurer.

        Different adventurers run concurrently via ``asyncio.gather``;
        subtasks assigned to the *same* adventurer run sequentially
        to avoid concurrent conversation state corruption.
        """
        total = sum(len(t) for t in adventurer_tasks.values())
        logger.debug(
            "Running %d subtasks across %d adventurer(s)",
            total,
            len(adventurer_tasks),
        )

        async def _run_sequential(
            adv_id: str,
            tasks: list[Quest],
        ) -> list[QuestResult]:
            adventurer = self._guildmaster._roster.get(adv_id)
            results: list[QuestResult] = []
            for task in tasks:
                result = await self._execute_subtask(task, adventurer)
                results.append(result)
            return results

        coroutines = [_run_sequential(adv_id, tasks) for adv_id, tasks in adventurer_tasks.items()]
        grouped_results = await asyncio.gather(*coroutines)

        # Flatten and re-sort by subtask_index to restore original ordering
        all_results: list[tuple[int, QuestResult]] = []
        for group in grouped_results:
            for result in group:
                child = self._quests.get(result.quest_id)
                idx = child.subtask_index if child and child.subtask_index is not None else 0
                all_results.append((idx, result))

        all_results.sort(key=lambda x: x[0])
        return [r for _, r in all_results]

    async def _execute_subtask(
        self,
        subtask: Quest,
        adventurer: BaseAdventurer | None,
    ) -> QuestResult:
        """Execute a single subtask quest through an adventurer."""
        if adventurer is None:
            subtask.transition(QuestStatus.FAILED, "guild")
            return QuestResult(
                sender="guild",
                quest_id=subtask.id,
                success=False,
                summary="No adventurer assigned.",
                failure_reason="no_adventurer",
            )

        # Spawn a fresh instance so no state leaks between subtasks
        runner = adventurer.spawn()
        logger.info(
            "Executing subtask %s (%r) with %s",
            subtask.id[:8],
            subtask.title,
            runner.name or runner.id,
        )
        # Subtask is already ASSIGNED from form_party_for_plan
        subtask.transition(QuestStatus.IN_PROGRESS, runner.name or runner.id)
        result = await runner.execute(subtask)

        if result.success:
            subtask.transition(QuestStatus.COMPLETED, runner.name or runner.id)
        else:
            subtask.transition(QuestStatus.FAILED, runner.name or runner.id)

        self._results[subtask.id] = result
        return result

    async def _verify_and_complete(
        self,
        quest: Quest,
        result: QuestResult,
    ) -> QuestResult:
        """Run guard + guildmaster verification and transition quest state."""
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
                return result
        else:
            logger.debug("No guard configured — skipping verification")

        accepted = await self._guildmaster.verify_result(quest, result)
        if accepted:
            quest.transition(QuestStatus.COMPLETED, "guildmaster")
            logger.info("Quest %s completed successfully", quest.id[:8])
        else:
            quest.transition(QuestStatus.FAILED, "guildmaster", {"reason": "Verification failed"})
            logger.warning("Quest %s failed verification", quest.id[:8])

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
