from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.adventurers.base_guard import BaseGuard
from guildmaster_ai.adventurers.base_hero import BaseHero
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
from guildmaster_ai.memory.sqlite_store import SQLiteStore
from guildmaster_ai.scrolls.catalog import ScrollCatalog

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
        scroll_catalog: ScrollCatalog | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.settings = settings or GuildSettings()
        self._llm = llm
        self._scroll_catalog = scroll_catalog
        self._board = QuestBoard()
        self._librarian = Librarian(llm=llm)
        self._guildmaster = Guildmaster(llm=llm, librarian=self._librarian)
        self._receptionist = Receptionist(
            llm=llm,
            max_rounds=self.settings.max_clarification_rounds,
        )
        self._guard: BaseGuard | None = None
        self._talents_refined = False

        # Cumulative counters — quests move to ARCHIVED after completion,
        # so counting by current status always yields 0 for COMPLETED/FAILED.
        self._completed_count = 0
        self._failed_count = 0

        # Persistent storage
        self.settings.guild_home.mkdir(parents=True, exist_ok=True)
        self._store = SQLiteStore(db_path=str(self.settings.db_path))
        self._store_initialized = False

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
        """Register an adventurer with the guild, injecting the LLM and catalog."""
        if adventurer.llm is None:
            adventurer.llm = self._llm
        if (
            isinstance(adventurer, BaseHero)
            and adventurer.scroll_catalog is None
            and self._scroll_catalog is not None
        ):
            adventurer.scroll_catalog = self._scroll_catalog
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
            completed=self._completed_count,
            failed=self._failed_count,
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
            await self._ensure_store()
            await self._restore_guild_counters()
            restored = await self._restore_talents_if_unchanged()
            if not restored:
                await self._guildmaster.refine_all_talents()
                await self._persist_talents()
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

        # Persist quest to DB
        await self._store.save_quest(quest)

        # Post to board
        self._board.post(quest)

        # Phase 1.5: Decomposition
        plan = await self._guildmaster.plan_quest(quest)
        if plan and len(plan.subtasks) > 1:
            quest.is_composite = True
            logger.info("Quest %s is composite — %d subtasks", quest.id[:8], len(plan.subtasks))

            # Use hero-led path if a BaseHero is available, else manual party
            hero = self._find_hero_for_quest(quest)
            if hero is not None:
                result = await self._execute_hero_led_quest(quest, plan, hero)
            else:
                result = await self._execute_manual_party_quest(quest, plan)
        else:
            result = await self._execute_simple_quest(quest)

        # Phase 4: Archival
        logger.info("Phase 4: Archival")
        await self._librarian.archive(quest, result)
        quest.transition(QuestStatus.ARCHIVED, "librarian")

        self._results[quest.id] = result

        # Persist final state
        await self._store.save_quest(quest)
        await self._persist_result(quest.id, result)
        await self._persist_guild_counters()

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
        result = await self._run_and_capture(leader, quest)

        return await self._verify_and_complete(quest, result)

    def _find_hero_for_quest(self, quest: Quest) -> BaseHero | None:
        """Find a registered BaseHero eligible for this quest."""
        matched = self._guildmaster.match_adventurers(quest)
        for adv in matched:
            if isinstance(adv, BaseHero):
                return adv
        return None

    async def _execute_hero_led_quest(
        self,
        quest: Quest,
        plan: QuestPlan,  # kept for API consistency with _execute_manual_party_quest
        hero: BaseHero,
    ) -> QuestResult:
        """Hero-led execution: recruit matched adventurers as subagents.

        The hero's deep agent handles delegation internally — the plan is not
        used directly here (the hero re-decomposes via its own planning).
        This is the preferred path when a BaseHero is available; the manual
        party path is the fallback.
        """
        logger.info("Phase 2: Execution (hero-led)")
        matched = self._guildmaster.match_adventurers(quest)

        # Spawn a fresh hero and recruit matched adventurers as members
        leader = hero.spawn()
        for adv in matched:
            if adv.id != hero.id:
                leader.recruit(adv.spawn())

        # Build party metadata for tracking
        party = Party(
            name=f"Party for {quest.title}",
            leader_id=leader.id,
            quest_id=quest.id,
        )
        for member_id in leader.members:
            party.add_member(member_id)

        quest.assigned_party_id = party.id
        quest.transition(QuestStatus.ASSIGNED, "guildmaster", {"party_id": party.id})
        quest.transition(
            QuestStatus.IN_PROGRESS,
            leader.name or leader.id,
        )

        logger.info(
            "Hero %s leading party with %d member(s)",
            leader.name or leader.id,
            len(leader.members),
        )

        # The deep agent coordinates members via its subagents
        result = await self._run_and_capture(leader, quest)
        return await self._verify_and_complete(quest, result)

    async def _execute_manual_party_quest(
        self,
        quest: Quest,
        plan: QuestPlan,
    ) -> QuestResult:
        """Execute a composite quest by distributing subtasks across a party."""
        logger.info("Phase 2: Execution (manual party)")
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

        subtask_results = await self._run_subtasks_with_deps(
            child_quests, plan, party,
        )
        await self._persist_subtask_findings(quest.id, 0, subtask_results)

        # Retry loop: evaluate subtask results, retry failed subtasks up to
        # max_retries times. Each iteration re-evaluates the full result set
        # (including updated retries) to decide done/retry/failed.
        max_retries = self.settings.max_quest_retries
        for attempt in range(max_retries):
            decision = await self._guildmaster.evaluate_quest_completion(
                quest,
                subtask_results,
            )
            if self._store_initialized:
                await self._store.save_finding(
                    quest.id,
                    attempt,
                    "leader_decision",
                    decision.reason,
                    {
                        "decision": decision.decision,
                        "retry_subtask_indices": decision.retry_subtask_indices,
                        "combined_summary": decision.combined_summary,
                    },
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
                self._failed_count += 1
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
            await self._persist_subtask_findings(quest.id, attempt + 1, retry_results)
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

    async def _run_subtasks_with_deps(
        self,
        child_quests: list[Quest],
        plan: QuestPlan,
        party: Party,
    ) -> list[QuestResult]:
        """Run subtasks respecting ``depends_on`` constraints.

        Executes in waves: each wave contains all subtasks whose dependencies
        have already completed.  Tasks within a wave run concurrently.
        """
        n = len(child_quests)
        deps: list[set[int]] = [
            set(plan.subtasks[i].depends_on) if i < len(plan.subtasks) else set()
            for i in range(n)
        ]

        results: list[QuestResult | None] = [None] * n
        completed: set[int] = set()
        failed: set[int] = set()

        while len(completed) + len(failed) < n:
            # Find the next wave: tasks not yet done whose deps are all completed
            wave = [
                i
                for i in range(n)
                if i not in completed
                and i not in failed
                and deps[i].issubset(completed)
            ]

            if not wave:
                # Deadlock: remaining tasks depend on failed tasks
                for i in range(n):
                    if i not in completed and i not in failed:
                        failed.add(i)
                        # Transition through IN_PROGRESS to reach FAILED
                        if child_quests[i].status == QuestStatus.ASSIGNED:
                            child_quests[i].transition(QuestStatus.IN_PROGRESS, "guild")
                        child_quests[i].transition(QuestStatus.FAILED, "guild")
                        results[i] = QuestResult(
                            sender="guild",
                            quest_id=child_quests[i].id,
                            success=False,
                            summary="Blocked by failed dependency.",
                            failure_reason="dependency_failed",
                        )
                break

            logger.info(
                "Running wave of %d subtask(s): %s",
                len(wave),
                [child_quests[i].title for i in wave],
            )

            async def _exec(idx: int) -> tuple[int, QuestResult]:
                adv_id = party.subtask_assignments.get(child_quests[idx].id, "")
                adventurer = self._guildmaster._roster.get(adv_id)
                return idx, await self._execute_subtask(child_quests[idx], adventurer)

            wave_results = await asyncio.gather(*[_exec(i) for i in wave])

            for idx, result in wave_results:
                results[idx] = result
                if result.success:
                    completed.add(idx)
                else:
                    failed.add(idx)

        # Return results in original order (None slots shouldn't exist but guard)
        return [
            r
            if r is not None
            else QuestResult(
                sender="guild",
                quest_id=child_quests[i].id,
                success=False,
                summary="Subtask not executed.",
                failure_reason="not_executed",
            )
            for i, r in enumerate(results)
        ]

    async def _run_subtasks(
        self,
        adventurer_tasks: dict[str, list[Quest]],
        _party: Party,  # reserved for future per-party tracking
    ) -> list[QuestResult]:
        """Run subtasks grouped by adventurer.

        Concurrency model:
        - Different adventurers run **concurrently** via ``asyncio.gather``
          (they have independent state and LLM conversations).
        - Subtasks assigned to the **same** adventurer run **sequentially**
          to avoid concurrent conversation state corruption.

        Results are flattened and re-sorted by ``subtask_index`` so the
        caller always sees them in the original plan order.
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
            subtask.transition(QuestStatus.IN_PROGRESS, "guild")
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
        result = await self._run_and_capture(runner, subtask)

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
                self._failed_count += 1
                result.success = False
                result.failure_reason = f"Blocked by guard: {verdict.reason}"
                return result
        else:
            logger.debug("No guard configured — skipping verification")

        accepted = await self._guildmaster.verify_result(quest, result)
        if accepted:
            quest.transition(QuestStatus.COMPLETED, "guildmaster")
            self._completed_count += 1
            logger.info("Quest %s completed successfully", quest.id[:8])
        else:
            quest.transition(QuestStatus.FAILED, "guildmaster", {"reason": "Verification failed"})
            self._failed_count += 1
            result.success = False
            result.failure_reason = "Guildmaster verification failed"
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

    # ── Persistent storage ────────────────────────────────────────────

    async def _ensure_store(self) -> None:
        """Lazily initialize the SQLite store on first use."""
        if not self._store_initialized:
            await self._store.initialize()
            self._store_initialized = True

    async def _restore_talents_if_unchanged(self) -> bool:
        """Try to restore talents from DB if all adventurer configs match."""
        for adv in self._guildmaster._roster.values():
            cached = await self._store.get_adventurer_talents(adv.config_hash)
            if cached is None:
                return False

        # All matched — restore talents
        for adv in self._guildmaster._roster.values():
            cached = await self._store.get_adventurer_talents(adv.config_hash)
            if cached is not None:
                adv._talents.clear()
                adv.grant_talents(cached)
                logger.debug(
                    "Restored talents for %s from DB: %s",
                    adv.name or adv.id,
                    cached,
                )
        logger.info("Restored talents from DB — skipping LLM assessment")
        return True

    async def _persist_talents(self) -> None:
        """Save current adventurer talents to the DB."""
        for adv in self._guildmaster._roster.values():
            await self._store.save_adventurer_talents(
                adv.config_hash,
                type(adv).__name__,
                adv.talents,
            )

    async def _run_and_capture(
        self,
        runner: BaseAdventurer,
        quest: Quest,
    ) -> QuestResult:
        """Execute *quest* with *runner* and persist its conversation transcript.

        Centralises adventurer execution so every quest — simple, hero-led, or
        subtask — records its message transcript to memory.
        """
        result = await runner.execute(quest)
        if self._store_initialized and result.transcript:
            await self._store.save_conversation(
                quest.id,
                runner.name or runner.id,
                result.transcript,
            )
        return result

    async def _persist_subtask_findings(
        self,
        quest_id: str,
        iteration: int,
        results: list[QuestResult],
    ) -> None:
        """Record each subtask outcome for *quest_id* at *iteration*."""
        if not self._store_initialized:
            return
        for r in results:
            await self._store.save_finding(
                quest_id,
                iteration,
                "subtask_result",
                r.summary,
                {
                    "subtask_quest_id": r.quest_id,
                    "success": r.success,
                    "failure_reason": r.failure_reason,
                },
            )

    async def _persist_result(self, quest_id: str, result: QuestResult) -> None:
        """Persist a quest result to the DB."""
        await self._store.save_result(
            quest_id,
            {
                "success": result.success,
                "summary": result.summary,
                "data": result.data,
                "failure_reason": result.failure_reason,
            },
        )

    async def _persist_guild_counters(self) -> None:
        """Save cumulative counters to guild_state."""
        await self._store.save_guild_state(
            "completed_count", str(self._completed_count)
        )
        await self._store.save_guild_state(
            "failed_count", str(self._failed_count)
        )
        await self._store.save_guild_state("guild_id", self.id)

    async def _restore_guild_counters(self) -> None:
        """Restore cumulative counters from guild_state if available."""
        guild_id = await self._store.get_guild_state("guild_id")
        if guild_id:
            self.id = guild_id
        completed = await self._store.get_guild_state("completed_count")
        if completed is not None:
            self._completed_count = int(completed)
        failed = await self._store.get_guild_state("failed_count")
        if failed is not None:
            self._failed_count = int(failed)

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the persistent store and release resources."""
        if self._store_initialized:
            await self._store.close()
            self._store_initialized = False
            logger.debug("Guild store closed")

    async def __aenter__(self) -> Guild:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        i = self.info
        return (
            f"Guild(id={i.id!r}, adventurers={len(i.adventurers)}, "
            f"completed={i.completed}, failed={i.failed}, in_progress={i.in_progress})"
        )
