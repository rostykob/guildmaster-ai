from __future__ import annotations

import json
import logging

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.adventurers.base_hero import BaseHero
from guildmaster_ai.core.messages import (
    AdventurerProfile,
    PartyLeaderDecision,
    QuestDraft,
    QuestPlan,
    QuestResult,
    QuestTriage,
    SubtaskSpec,
)
from guildmaster_ai.core.party import Party, PartyMember
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.core.quest_board import QuestBoard
from guildmaster_ai.core.utils import charter_block, parse_llm_json, safe_parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete

logger = logging.getLogger("guildmaster.guildmaster")


def _chronicle_block(chronicle: str) -> str:
    """Format the librarian chronicle for prompt injection (empty when absent)."""
    if not chronicle:
        return ""
    return f"\n\nGuild chronicle (lessons from past quests):\n{chronicle}"


class Guildmaster:
    """Central coordinator that manages adventurers, quests, and party formation."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        charter: str = "",
    ) -> None:
        self._roster: dict[str, BaseAdventurer] = {}
        self._llm = llm
        self._charter = charter

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Add an adventurer to the guild roster.

        Grants initial talents derived from equipped weapons and armor.
        Full LLM-based talent assessment happens later via
        :meth:`assess_talents_for` (driven by ``Guild.prepare()``).
        """
        talents = self._derive_equipment_talents(adventurer)
        adventurer.grant_talents(talents)
        self._roster[adventurer.id] = adventurer
        logger.info("Registered adventurer %r with talents %s", adventurer.label, talents)

    @property
    def roster(self) -> list[AdventurerProfile]:
        """Return profiles of all registered adventurers."""
        return [adv.profile() for adv in self._roster.values()]

    @property
    def adventurers(self) -> list[BaseAdventurer]:
        """Return all registered adventurers."""
        return list(self._roster.values())

    # ── Talent assessment ──────────────────────────────────────────────

    @staticmethod
    def _derive_equipment_talents(adventurer: BaseAdventurer) -> list[str]:
        """Derive initial talents from an adventurer's weapons and armor.

        This is a lightweight, sync-safe method used at registration time.
        Full LLM-based assessment is deferred to :meth:`assess_talents_for`.
        """
        names = adventurer.weapon_names + [a.name for a in adventurer.armor]
        talents: list[str] = []
        for name in names:
            talent = name.lower().replace(" ", "_")
            if talent not in talents:
                talents.append(talent)

        # Every adventurer gets at least "general"
        return talents or ["general"]

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
                "Given their description, system prompt, weapons, and armor, "
                "return ONLY a JSON array of talent strings. Prefer talents "
                f"from this known list when applicable: {known}. "
                "You may add new talent names if none fit. "
                "Include at least one talent."
            ),
            user=(
                f"Description: {adventurer.description or '(none)'}\n"
                f"System prompt: {adventurer.system_prompt}\n"
                f"Weapons: {json.dumps(weapons_info)}\n"
                f"Armor: {armor_info}"
            ),
        )

        talents = self._parse_string_list(raw)
        if not talents:
            return self._derive_equipment_talents(adventurer)
        return talents

    async def assess_talents_for(self, adventurer: BaseAdventurer) -> list[str]:
        """Assess one adventurer's talents, preserving the "general" wildcard.

        Wraps :meth:`_llm_assess_talents` and keeps the equipment-derived
        "general" talent when the LLM omits it, so a general-purpose adventurer
        never loses its wildcard eligibility.
        """
        assert self._llm is not None
        had_general = "general" in adventurer.talents
        new_talents = await self._llm_assess_talents(adventurer)
        if had_general and "general" not in new_talents:
            new_talents.append("general")
        return new_talents

    async def triage_quest(self, draft: QuestDraft, chronicle: str = "") -> QuestTriage:
        """Rank a quest and pick candidate adventurers in ONE LLM call.

        Replaces the old talent-extraction → feasibility → adventurer-ranking
        chain: the guildmaster already knows every adventurer's talents, so it
        matches the quest description against the roster directly. Returns
        candidates ordered best-first; an empty list with an
        ``infeasible_reason`` means nobody can take the quest.

        The rank drives routing: ranks below ``B`` are handled by a single
        adventurer (no decomposition), ``B`` forms a party, and ``A``/``S`` are
        hero-led. Falls back to rank ``E`` with the full roster when no LLM is
        configured or the response can't be parsed.

        *chronicle* is the librarian's rolling summary of lessons from past
        quests (empty when observations are disabled).
        """
        roster = list(self._roster.values())
        if not roster:
            return QuestTriage(
                sender="guildmaster",
                infeasible_reason="The guild has no registered adventurers",
            )
        roster_ids = [adv.id for adv in roster]
        if self._llm is None:
            return QuestTriage(sender="guildmaster", adventurer_ids=roster_ids)

        profiles = [
            {
                "id": adv.id,
                "name": adv.label,
                "description": adv.description,
                "hero": isinstance(adv, BaseHero),
                "talents": adv.talents,
                "weapons": adv.weapon_names,
            }
            for adv in roster
        ]
        chronicle_text = _chronicle_block(chronicle)
        hints = (
            f"\nUser-requested talents: {draft.required_talents}" if draft.required_talents else ""
        )

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master triaging a new quest. Decide two things:\n"
                "(1) Its difficulty rank:\n"
                "- F: trivial; one adventurer, no tools; single simple request\n"
                "- E: one adventurer with a basic tool; single request\n"
                "- D: one adventurer; multiple simple requests\n"
                "- C: one adventurer with tools; multiple requests\n"
                "- B: a party of adventurers; multiple INDEPENDENT subtasks\n"
                "- A: a hero leading; multiple subtasks WITH dependencies\n"
                "- S: a party led by a hero; advanced reasoning across subtasks\n"
                "Most quests are F-C (one adventurer). Only pick B or higher "
                "when the quest genuinely splits into multiple distinct subtasks.\n"
                "(2) Which adventurers should take it, ordered best-fit first, "
                "judged by their talents and weapons against the quest. Include "
                "every adventurer that could contribute; for rank A/S prefer a "
                "hero first. If NO adventurer can plausibly handle the quest, "
                "return an empty list and a short infeasible_reason.\n"
                'Respond with ONLY JSON: {"rank": "F|E|D|C|B|A|S", '
                '"adventurers": ["<id>", ...], "infeasible_reason": null}'
                f"{charter_block(self._charter)}"
            ),
            user=(
                f"Title: {draft.title}\n"
                f"Description: {draft.description}{hints}\n\n"
                f"Adventurers:\n{json.dumps(profiles, indent=2)}"
                f"{chronicle_text}"
            ),
        )

        data = safe_parse_llm_json(raw, context="triage_quest")
        if not isinstance(data, dict):
            logger.warning("Could not parse triage response — using full roster, rank E")
            return QuestTriage(sender="guildmaster", adventurer_ids=roster_ids)

        reason = data.get("infeasible_reason") or None
        candidate_ids = self._resolve_adventurer_refs(data.get("adventurers", []))
        if not candidate_ids and reason is None:
            # LLM answered but named nobody we know — don't brick the quest.
            candidate_ids = roster_ids
        return QuestTriage(
            sender="guildmaster",
            rank=self._parse_rank(data.get("rank")),
            adventurer_ids=candidate_ids,
            infeasible_reason=reason if not candidate_ids else None,
        )

    def _resolve_adventurer_refs(self, refs: object) -> list[str]:
        """Map LLM-provided adventurer ids or names onto roster ids (order kept)."""
        if not isinstance(refs, list):
            return []
        by_name = {
            (adv.label).lower(): adv.id for adv in self._roster.values()
        }
        resolved: list[str] = []
        for ref in refs:
            key = str(ref)
            adv_id = key if key in self._roster else by_name.get(key.lower())
            if adv_id is not None and adv_id not in resolved:
                resolved.append(adv_id)
        return resolved

    @staticmethod
    def _parse_rank(value: object) -> QuestRank:
        """Parse a rank letter (e.g. "B") into a :class:`QuestRank`, default E."""
        if isinstance(value, str):
            try:
                return QuestRank[value.strip().upper()]
            except KeyError:
                pass
        return QuestRank.E

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

    def get_adventurer(self, adventurer_id: str) -> BaseAdventurer | None:
        """Return a roster adventurer by id, or ``None``."""
        return self._roster.get(adventurer_id)

    def _resolve_assignee(self, ref: str) -> BaseAdventurer | None:
        """Resolve a planned assignee reference (id or name) to an adventurer."""
        if not ref:
            return None
        ids = self._resolve_adventurer_refs([ref])
        return self._roster.get(ids[0]) if ids else None

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
                f"{charter_block(self._charter)}"
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Acceptance criteria:\n"
                + "\n".join(f"- {c}" for c in quest.acceptance_criteria)
                + f"\n\nResult:\n{result.summary}"
            ),
        )

        data = safe_parse_llm_json(raw, context="verify_result")
        if data is None:
            logger.warning("Could not parse verification response — accepting by default")
            return result.success

        accepted = bool(data.get("accepted", True))
        reason = data.get("reason", "")
        if not accepted:
            logger.warning(
                "Quest %s failed verification: %s",
                quest.id[:8],
                reason,
            )
        return accepted

    # ── Quest decomposition ──────────────────────────────────────────

    async def plan_quest(self, quest: Quest, chronicle: str = "") -> QuestPlan | None:
        """Analyze a quest and decide whether to decompose it into subtasks.

        Returns a :class:`QuestPlan` if the quest should be decomposed,
        or ``None`` if it should be handled as a simple quest. Each subtask
        gets an ``assignee`` picked from the roster in the same call.

        *chronicle* is the librarian's rolling lessons summary — injected as
        context instead of raw retrieved observation chunks.
        """
        if self._llm is None:
            logger.debug("No LLM configured — skipping quest planning")
            return None

        logger.info("Analyzing quest %s for decomposition", quest.id[:8])

        profiles = [
            {
                "id": adv.id,
                "name": adv.label,
                "description": adv.description,
                "talents": adv.talents,
            }
            for adv in self._roster.values()
        ]
        chronicle_text = _chronicle_block(chronicle)

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild master analyzing quest complexity. "
                "Decide whether this quest should be broken into subtasks.\n\n"
                "IMPORTANT: Most quests are simple. Only decompose if the quest "
                "genuinely requires DIFFERENT skills or INDEPENDENT work streams. "
                "A question, a single writing task, or a focused analysis should "
                "NEVER be decomposed. When in doubt, keep it simple.\n\n"
                "If the quest is simple (can be done by one adventurer), "
                'respond with: {"decompose": false}\n'
                "If the quest is complex, respond with:\n"
                '{"decompose": true, "strategy": "brief strategy", '
                '"subtasks": [{"title": "...", "description": "...", '
                '"assignee": "<adventurer id>", "acceptance_criteria": [...], '
                '"depends_on": [...]}]}\n'
                '"assignee" is the id of the roster adventurer best suited for '
                "the subtask.\n"
                '"depends_on" is a list of 0-based subtask indices that must '
                "complete before this subtask can start. Use [] for independent subtasks.\n"
                "Respond with ONLY a JSON object."
                f"{charter_block(self._charter)}"
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Acceptance criteria: {quest.acceptance_criteria}\n\n"
                f"Adventurers:\n{json.dumps(profiles, indent=2)}"
                f"{chronicle_text}"
            ),
        )

        data = safe_parse_llm_json(raw, context="plan_quest")
        if data is None:
            logger.warning("Could not parse plan response — treating as simple quest")
            return None

        if not data.get("decompose", False):
            return None

        subtasks = [
            SubtaskSpec(
                title=st.get("title", ""),
                description=st.get("description", ""),
                required_talents=st.get("required_talents", []),
                acceptance_criteria=st.get("acceptance_criteria", []),
                depends_on=st.get("depends_on", []),
                assignee=str(st.get("assignee", "")),
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
            prior_observations=[chronicle] if chronicle else [],
        )

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

        # Match adventurers to subtasks and build party.
        # Leader selection: the adventurer with the highest talent overlap with
        # the parent quest's required_talents becomes the party leader.
        best_leader: BaseAdventurer | None = None
        best_leader_score = -1

        members: list[PartyMember] = []
        member_ids_seen: set[str] = set()

        party = Party(
            name=f"Party for {quest.title}",
            leader_id="",  # set after matching
            quest_id=quest.id,
        )

        for child, spec in zip(child_quests, plan.subtasks, strict=True):
            # Prefer the assignee the planning call already chose; fall back
            # to talent matching, then to the full roster.
            adventurer = self._resolve_assignee(spec.assignee)
            if adventurer is None:
                matched = self.match_adventurers(child) or list(self._roster.values())
                adventurer = matched[0] if matched else None
                logger.debug(
                    "No planned assignee for subtask %r — matched %s",
                    child.title,
                    adventurer.label if adventurer else None,
                )

            if adventurer is not None:
                party.assign_subtask(child.id, adventurer.id)
                logger.debug("Subtask %r -> adventurer %s", child.title, adventurer.label)

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
        if self._llm is not None:
            return await self._llm_evaluate_completion(quest, subtask_results)
        return self._rule_based_completion(quest, subtask_results)

    @staticmethod
    def _rule_based_completion(
        quest: Quest,
        subtask_results: list[QuestResult],
    ) -> PartyLeaderDecision:
        """Evaluate completion without an LLM: all-success → done, else retry."""
        failed_indices = [i for i, r in enumerate(subtask_results) if not r.success]
        if not failed_indices:
            return PartyLeaderDecision(
                sender="guildmaster",
                quest_id=quest.id,
                decision="done",
                reason="All subtasks completed successfully.",
                combined_summary="\n\n".join(r.summary for r in subtask_results),
            )
        return PartyLeaderDecision(
            sender="guildmaster",
            quest_id=quest.id,
            decision="retry",
            reason=f"Subtasks at indices {failed_indices} failed.",
            retry_subtask_indices=failed_indices,
        )

    async def _llm_evaluate_completion(
        self,
        quest: Quest,
        subtask_results: list[QuestResult],
    ) -> PartyLeaderDecision:
        """Use the LLM to evaluate composite quest completion.

        Tries LLM-based evaluation first; on parse failure, falls back to
        rule-based logic (all-success → done, any-failed → retry).
        """
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

        data = safe_parse_llm_json(raw, context="evaluate_completion")
        if data is None:
            # Fall back to rule-based when LLM response couldn't be parsed
            return self._rule_based_completion(quest, subtask_results)

        all_success = all(r.success for r in subtask_results)
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
        logger.warning("Could not parse string list from LLM: %.200s", raw)
        return []
