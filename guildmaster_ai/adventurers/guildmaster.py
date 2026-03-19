from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.messages import (
    AdventurerProfile,
    PartyLeaderDecision,
    QuestDraft,
    QuestFeasibilityReport,
    QuestPlan,
    QuestResult,
    SubtaskSpec,
)
from guildmaster_ai.core.party import Party, PartyMember
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.core.quest_board import QuestBoard
from guildmaster_ai.core.utils import parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete

if TYPE_CHECKING:
    from guildmaster_ai.adventurers.librarian import Librarian

logger = logging.getLogger("guildmaster.guildmaster")


class Guildmaster:
    """Central coordinator that manages adventurers, quests, and party formation."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        librarian: Librarian | None = None,
    ) -> None:
        self._roster: dict[str, BaseAdventurer] = {}
        self._llm = llm
        self._librarian = librarian

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Add an adventurer to the guild roster.

        Grants initial talents derived from equipped weapons and armor.
        Full LLM-based talent assessment happens later via
        :meth:`refine_all_talents`.
        """
        talents = self._derive_equipment_talents(adventurer)
        adventurer.grant_talents(talents)
        self._roster[adventurer.id] = adventurer
        logger.info(
            "Registered adventurer %r with talents %s",
            adventurer.name or adventurer.id,
            talents,
        )

    def unregister_adventurer(self, adventurer_id: str) -> None:
        """Remove an adventurer from the guild roster."""
        self._roster.pop(adventurer_id, None)

    @property
    def roster(self) -> list[AdventurerProfile]:
        """Return profiles of all registered adventurers."""
        return [adv.profile() for adv in self._roster.values()]

    # ── Talent assessment ──────────────────────────────────────────────

    @staticmethod
    def _derive_equipment_talents(adventurer: BaseAdventurer) -> list[str]:
        """Derive initial talents from an adventurer's weapons and armor.

        This is a lightweight, sync-safe method used at registration time.
        Full LLM-based assessment is deferred to :meth:`refine_all_talents`.
        """
        talents: list[str] = []

        for weapon_name in adventurer.weapon_names:
            weapon_talent = weapon_name.lower().replace(" ", "_")
            if weapon_talent not in talents:
                talents.append(weapon_talent)

        for armor_piece in adventurer.armor:
            armor_talent = armor_piece.name.lower().replace(" ", "_")
            if armor_talent not in talents:
                talents.append(armor_talent)

        # Every adventurer gets at least "general"
        if not talents:
            talents.append("general")

        return talents

    async def _llm_assess_talents(self, adventurer: BaseAdventurer) -> list[str]:
        """Use the LLM to assess an adventurer's talents from their full profile.

        Analyzes the system prompt, equipped weapons (with descriptions), and
        worn armor to determine the adventurer's capabilities.
        """
        assert self._llm is not None

        weapons_info = {
            name: weapon.description
            for name, weapon in adventurer.weapons.items()
        }
        armor_info = [a.name for a in adventurer.armor]
        known = self._known_talents()

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master assessing an adventurer's talents. "
                "Given their system prompt, weapons, and armor, return ONLY "
                "a JSON array of talent strings. Prefer talents from this "
                f"known list when applicable: {known}. "
                "You may add new talent names if none fit. "
                "Include at least one talent."
            ),
            user=(
                f"System prompt: {adventurer.system_prompt}\n"
                f"Weapons: {json.dumps(weapons_info)}\n"
                f"Armor: {armor_info}"
            ),
        )

        talents = self._parse_string_list(raw)
        if not talents:
            return self._derive_equipment_talents(adventurer)
        return talents

    async def refine_all_talents(self) -> None:
        """Assess talents for all registered adventurers using the LLM.

        Called lazily on first quest when an LLM is available. Replaces
        the initial equipment-derived talents with richer LLM-derived ones.
        """
        if self._llm is None:
            return

        logger.info("Assessing adventurer talents via LLM (%d adventurers)", len(self._roster))
        for adv in self._roster.values():
            old_talents = adv.talents
            new_talents = await self._llm_assess_talents(adv)
            adv._talents.clear()
            adv.grant_talents(new_talents)

            if set(new_talents) != set(old_talents):
                logger.debug(
                    "Assessed %s: %s -> %s",
                    adv.name or adv.id,
                    old_talents,
                    new_talents,
                )

    async def assess_quest_talents(self, draft: QuestDraft) -> list[str]:
        """Determine which talents a quest requires using the LLM.

        Returns an empty list when no LLM is configured (all adventurers
        with the "general" talent will still match).
        """
        if self._llm is None:
            return []

        known = self._known_talents()
        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master assessing what talents are needed "
                "for a quest. Given the quest description, return ONLY a "
                "JSON array of talent strings. Prefer talents from this "
                f"known list when applicable: {known}. "
                "You may add new talent names if none fit."
            ),
            user=f"Title: {draft.title}\nDescription: {draft.description}",
        )
        return self._parse_string_list(raw)

    # ── Feasibility ────────────────────────────────────────────────────

    def check_feasibility(self, draft: QuestDraft) -> QuestFeasibilityReport:
        """Check whether the guild can staff a quest draft."""
        logger.info(
            "Checking feasibility for %r (requires: %s)",
            draft.title,
            draft.required_talents,
        )
        all_talents: set[str] = set()
        matched: list[AdventurerProfile] = []

        for adv in self._roster.values():
            adv_talents = set(adv.talents)
            all_talents.update(adv_talents)
            # "general" talent acts as a wildcard — the adventurer can attempt any quest
            if "general" in adv_talents or set(draft.required_talents) & adv_talents:
                matched.append(adv.profile())

        # If no specific talents required, all adventurers are potential matches
        if not draft.required_talents:
            matched = [adv.profile() for adv in self._roster.values()]

        # Only report truly missing talents if no general-purpose adventurer is available
        has_general = any("general" in set(a.talents) for a in self._roster.values())
        missing = [] if has_general else [t for t in draft.required_talents if t not in all_talents]
        feasible = len(missing) == 0 and len(matched) > 0
        logger.info(
            "Feasibility: feasible=%s matched=%d missing=%s",
            feasible,
            len(matched),
            missing,
        )

        return QuestFeasibilityReport(
            sender="guildmaster",
            feasible=feasible,
            matched_adventurers=matched,
            missing_talents=missing,
            recommended_rank=draft.rank or QuestRank.E,
        )

    # ── Adventurer matching ────────────────────────────────────────────

    def match_adventurers(self, quest: Quest) -> list[BaseAdventurer]:
        """Return adventurers eligible for a quest.

        Uses keyword-based talent overlap. The "general" talent acts as
        a wildcard — an adventurer with it can attempt any quest.
        """
        required = set(quest.required_talents)
        eligible: list[BaseAdventurer] = []

        for adv in self._roster.values():
            adv_talents = set(adv.talents)
            if required and "general" not in adv_talents and not (required & adv_talents):
                continue
            eligible.append(adv)

        return eligible

    async def rank_adventurers(
        self,
        quest: Quest,
    ) -> list[BaseAdventurer]:
        """Return adventurers ranked by suitability for the quest.

        Uses the LLM when available to score each adventurer's profile
        against the quest; falls back to :meth:`match_adventurers`.
        """
        candidates = self.match_adventurers(quest)
        if not candidates or self._llm is None or len(candidates) <= 1:
            return candidates

        profiles = {
            adv.id: {
                "name": adv.name or adv.id,
                "talents": adv.talents,
                "weapons": adv.weapon_names,
            }
            for adv in candidates
        }

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master choosing the best adventurer for a quest. "
                "Given the quest and adventurer profiles, return ONLY a JSON array "
                "of adventurer IDs ordered from best to worst fit."
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Required talents: {quest.required_talents}\n\n"
                f"Adventurers:\n{json.dumps(profiles, indent=2)}"
            ),
        )

        ranked_ids = self._parse_string_list(raw)
        id_to_adv = {adv.id: adv for adv in candidates}
        ranked = [id_to_adv[aid] for aid in ranked_ids if aid in id_to_adv]
        # Append any candidates the LLM missed
        for adv in candidates:
            if adv not in ranked:
                ranked.append(adv)
        return ranked

    # ── Quest assignment ───────────────────────────────────────────────

    async def assign_quest(self, quest: Quest, board: QuestBoard) -> Party | None:
        """Match adventurers, form a party, and assign the quest on the board."""
        logger.info("Assigning quest %s: %r", quest.id[:8], quest.title)
        if self._llm is not None:
            matched = await self.rank_adventurers(quest)
        else:
            matched = self.match_adventurers(quest)

        if not matched:
            logger.warning("No adventurers matched quest %s", quest.id[:8])
            return None

        leader = matched[0]
        party = Party(
            name=f"Party for {quest.title}",
            leader_id=leader.id,
            quest_id=quest.id,
            members=[
                PartyMember(
                    adventurer_id=adv.id,
                    role="leader" if adv.id == leader.id else "member",
                )
                for adv in matched
            ],
        )

        board.assign(quest.id, party.id, actor="guildmaster")
        return party

    # ── Verification ───────────────────────────────────────────────────

    async def verify_result(self, quest: Quest, result: QuestResult) -> bool:
        """Verify a quest result.

        Uses the LLM to check whether the result satisfies the quest's
        acceptance criteria. Falls back to ``result.success`` when no LLM
        is configured or criteria are empty.
        """
        if not result.success:
            return False

        if self._llm is None or not quest.acceptance_criteria:
            return result.success

        logger.info("Verifying quest %s result via LLM", quest.id[:8])
        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master verifying quest results. "
                "Check whether the result meets ALL acceptance criteria. "
                'Respond with ONLY a JSON object: {"accepted": true/false, '
                '"reason": "brief explanation"}.'
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Acceptance criteria:\n"
                + "\n".join(f"- {c}" for c in quest.acceptance_criteria)
                + f"\n\nResult:\n{result.summary}"
            ),
        )

        try:
            data = parse_llm_json(raw)
            accepted = bool(data.get("accepted", True))
            reason = data.get("reason", "")
            if not accepted:
                logger.warning(
                    "Quest %s failed verification: %s",
                    quest.id[:8],
                    reason,
                )
            return accepted
        except (ValueError, KeyError):
            logger.debug("Could not parse verification response — accepting")
            return result.success

    # ── Quest decomposition ──────────────────────────────────────────

    async def plan_quest(self, quest: Quest) -> QuestPlan | None:
        """Analyze a quest and decide whether to decompose it into subtasks.

        Returns a :class:`QuestPlan` if the quest should be decomposed,
        or ``None`` if it should be handled as a simple quest.
        """
        if self._llm is None:
            logger.debug("No LLM configured — skipping quest planning")
            return None

        logger.info("Analyzing quest %s for decomposition", quest.id[:8])

        # Search librarian for similar past quests
        prior_observations: list[str] = []
        if self._librarian is not None:
            similar = await self._librarian.search_similar_quests(
                quest.description,
                n_results=3,
            )
            prior_observations = [obs.summary for obs in similar]

        obs_text = ""
        if prior_observations:
            obs_text = "\n\nPrior observations from similar quests:\n" + "\n".join(
                f"- {o}" for o in prior_observations
            )

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master analyzing quest complexity. "
                "Decide whether this quest should be broken into subtasks. "
                "If the quest is simple (can be done by one adventurer), "
                'respond with: {"decompose": false}\n'
                "If the quest is complex, respond with:\n"
                '{"decompose": true, "strategy": "brief strategy", '
                '"subtasks": [{"title": "...", "description": "...", '
                '"required_talents": [...], "acceptance_criteria": [...]}]}\n'
                "Respond with ONLY a JSON object."
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Required talents: {quest.required_talents}\n"
                f"Acceptance criteria: {quest.acceptance_criteria}"
                f"{obs_text}"
            ),
        )

        try:
            data = parse_llm_json(raw)
            if not data.get("decompose", False):
                return None

            subtasks = [
                SubtaskSpec(
                    title=st.get("title", ""),
                    description=st.get("description", ""),
                    required_talents=st.get("required_talents", []),
                    acceptance_criteria=st.get("acceptance_criteria", []),
                )
                for st in data.get("subtasks", [])
            ]

            if not subtasks:
                return None

            logger.info(
                "Quest %s decomposed into %d subtasks",
                quest.id[:8],
                len(subtasks),
            )
            return QuestPlan(
                sender="guildmaster",
                quest_id=quest.id,
                subtasks=subtasks,
                strategy=data.get("strategy", ""),
                prior_observations=prior_observations,
            )
        except (ValueError, KeyError):
            logger.debug("Could not parse plan response — treating as simple quest")
            return None

    async def form_party_for_plan(
        self,
        quest: Quest,
        plan: QuestPlan,
        board: QuestBoard,
    ) -> tuple[Party, list[Quest]]:
        """Form a party and create child quests for each subtask in the plan.

        Returns the party and the list of child quests.
        """
        child_quests: list[Quest] = []
        for idx, spec in enumerate(plan.subtasks):
            child = Quest(
                title=spec.title,
                description=spec.description,
                required_talents=spec.required_talents,
                rank=quest.rank,
                acceptance_criteria=spec.acceptance_criteria,
                parent_quest_id=quest.id,
                subtask_index=idx,
            )
            child_quests.append(child)
            board.post(child)

        # Match adventurers to subtasks and build party
        best_leader: BaseAdventurer | None = None
        best_leader_score = -1

        members: list[PartyMember] = []
        member_ids_seen: set[str] = set()

        party = Party(
            name=f"Party for {quest.title}",
            leader_id="",  # set after matching
            quest_id=quest.id,
        )

        for child in child_quests:
            matched = self.match_adventurers(child)
            if not matched:
                matched = list(self._roster.values())
                logger.debug(
                    "No talent match for subtask %r — falling back to full roster",
                    child.title,
                )

            if matched:
                adventurer = matched[0]
                party.assign_subtask(child.id, adventurer.id)
                logger.debug(
                    "Subtask %r -> adventurer %s",
                    child.title,
                    adventurer.name or adventurer.id,
                )

                # Track best leader by talent overlap
                overlap = len(set(adventurer.talents) & set(quest.required_talents))
                if overlap > best_leader_score:
                    best_leader_score = overlap
                    best_leader = adventurer

                if adventurer.id not in member_ids_seen:
                    member_ids_seen.add(adventurer.id)
                    members.append(PartyMember(adventurer_id=adventurer.id, role="member"))

                # Assign quest on the board
                board.assign(child.id, party.id, actor="guildmaster")

        if best_leader is not None:
            party.leader_id = best_leader.id
            for m in members:
                if m.adventurer_id == best_leader.id:
                    m.role = "leader"

        party.members = members
        logger.info(
            "Party formed for quest %s: %d members, %d subtasks",
            quest.id[:8],
            len(members),
            len(child_quests),
        )
        return party, child_quests

    async def evaluate_quest_completion(
        self,
        quest: Quest,
        subtask_results: list[QuestResult],
    ) -> PartyLeaderDecision:
        """Evaluate whether a composite quest is complete based on subtask results.

        Uses the LLM when available; falls back to rule-based evaluation.
        """
        logger.info(
            "Evaluating completion for quest %s (%d subtask results)",
            quest.id[:8],
            len(subtask_results),
        )
        all_success = all(r.success for r in subtask_results)
        failed_indices = [i for i, r in enumerate(subtask_results) if not r.success]

        if self._llm is not None:
            return await self._llm_evaluate_completion(
                quest,
                subtask_results,
                all_success,
                failed_indices,
            )

        # Rule-based fallback
        if all_success:
            combined = "\n\n".join(r.summary for r in subtask_results)
            return PartyLeaderDecision(
                sender="guildmaster",
                quest_id=quest.id,
                decision="done",
                reason="All subtasks completed successfully.",
                combined_summary=combined,
            )

        if failed_indices:
            return PartyLeaderDecision(
                sender="guildmaster",
                quest_id=quest.id,
                decision="retry",
                reason=f"Subtasks at indices {failed_indices} failed.",
                retry_subtask_indices=failed_indices,
            )

        return PartyLeaderDecision(
            sender="guildmaster",
            quest_id=quest.id,
            decision="failed",
            reason="Quest could not be completed.",
        )

    async def _llm_evaluate_completion(
        self,
        quest: Quest,
        subtask_results: list[QuestResult],
        all_success: bool,
        failed_indices: list[int],
    ) -> PartyLeaderDecision:
        """Use the LLM to evaluate composite quest completion."""
        assert self._llm is not None

        results_text = "\n".join(
            f"Subtask {i}: {'SUCCESS' if r.success else 'FAILED'} — {r.summary}"
            for i, r in enumerate(subtask_results)
        )

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a party leader evaluating quest completion. "
                "Given the subtask results, decide the overall outcome. "
                "Respond with ONLY a JSON object:\n"
                '{"decision": "done|failed|retry", "reason": "...", '
                '"retry_subtask_indices": [...], "combined_summary": "..."}'
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Acceptance criteria: {quest.acceptance_criteria}\n\n"
                f"Subtask results:\n{results_text}"
            ),
        )

        try:
            data = parse_llm_json(raw)
            decision = data.get("decision", "done")
            if decision not in ("done", "failed", "retry"):
                decision = "done" if all_success else "failed"
            return PartyLeaderDecision(
                sender="guildmaster",
                quest_id=quest.id,
                decision=decision,
                reason=data.get("reason", ""),
                retry_subtask_indices=data.get("retry_subtask_indices", []),
                combined_summary=data.get("combined_summary", ""),
            )
        except (ValueError, KeyError):
            # Fall back to rule-based
            if all_success:
                combined = "\n\n".join(r.summary for r in subtask_results)
                return PartyLeaderDecision(
                    sender="guildmaster",
                    quest_id=quest.id,
                    decision="done",
                    reason="All subtasks completed successfully.",
                    combined_summary=combined,
                )
            return PartyLeaderDecision(
                sender="guildmaster",
                quest_id=quest.id,
                decision="retry" if failed_indices else "failed",
                reason=f"Subtasks at indices {failed_indices} failed.",
                retry_subtask_indices=failed_indices,
            )

    # ── Helpers ────────────────────────────────────────────────────────

    def _known_talents(self) -> list[str]:
        """Collect all unique talents currently assigned across the roster."""
        talents: set[str] = set()
        for adv in self._roster.values():
            talents.update(adv.talents)
        return sorted(talents)

    @staticmethod
    def _parse_string_list(raw: str) -> list[str]:
        """Parse a JSON string-array from LLM output, tolerating markdown fences."""
        try:
            data = parse_llm_json(raw)
            if isinstance(data, list):
                return [str(x) for x in data]
        except (ValueError, KeyError):
            pass
        return []
