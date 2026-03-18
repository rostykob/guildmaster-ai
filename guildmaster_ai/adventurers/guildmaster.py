from __future__ import annotations

import json
import logging
import re

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.messages import (
    AdventurerProfile,
    QuestDraft,
    QuestFeasibilityReport,
    QuestResult,
)
from guildmaster_ai.core.party import Party, PartyMember
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.core.quest_board import QuestBoard
from guildmaster_ai.core.utils import parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete

# Keyword-to-talent mapping used for prompt-based talent inference.
logger = logging.getLogger("guildmaster.guildmaster")

# Base talent keyword mapping — extended dynamically as adventurers register.
_DEFAULT_TALENT_KEYWORDS: dict[str, list[str]] = {
    "general": ["general", "versatile", "broad", "wide range"],
    "reasoning": ["reason", "analyz", "logic", "think", "deduc"],
    "coding": ["code", "program", "software", "develop", "implement"],
    "web_search": ["search", "browse", "web", "internet", "lookup"],
    "writing": ["write", "draft", "compose", "author", "document"],
    "research": ["research", "investigate", "study", "explore"],
    "math": ["math", "calcul", "comput", "arithmetic", "equation"],
    "data_analysis": ["data", "dataset", "statistic", "csv", "tabular"],
    "file_operations": ["file", "read", "write", "directory", "filesystem"],
    "summarization": ["summar", "condense", "distill", "brief"],
}


class Guildmaster:
    """Central coordinator that manages adventurers, quests, and party formation."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
    ) -> None:
        self._roster: dict[str, BaseAdventurer] = {}
        self._llm = llm
        self._talent_keywords = dict(_DEFAULT_TALENT_KEYWORDS)

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Add an adventurer to the guild roster, assessing and assigning talents."""
        talents = self._assess_talents(adventurer)
        adventurer.grant_talents(talents)
        self._roster[adventurer.id] = adventurer

        # Extend talent keywords with any new talents discovered
        for talent in talents:
            if talent not in self._talent_keywords:
                self._talent_keywords[talent] = [talent.replace("_", " ")]

        logger.info(
            "Registered adventurer %r with talents %s",
            adventurer.name or adventurer.id, talents,
        )

    def unregister_adventurer(self, adventurer_id: str) -> None:
        """Remove an adventurer from the guild roster."""
        self._roster.pop(adventurer_id, None)

    @property
    def roster(self) -> list[AdventurerProfile]:
        """Return profiles of all registered adventurers."""
        return [adv.profile() for adv in self._roster.values()]

    # ── Talent assessment ──────────────────────────────────────────────

    def _assess_talents(self, adventurer: BaseAdventurer) -> list[str]:
        """Infer talents from the adventurer's system prompt, weapons, and armor."""
        talents: list[str] = []
        prompt_lower = adventurer.system_prompt.lower()

        # Extract talents from system prompt via keyword matching
        for talent, keywords in self._talent_keywords.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}", prompt_lower):
                    talents.append(talent)
                    break

        # Derive talents from equipped weapons
        for weapon_name in adventurer.weapon_names:
            weapon_talent = weapon_name.lower().replace(" ", "_")
            if weapon_talent not in talents:
                talents.append(weapon_talent)

        # Derive talents from worn armor
        for armor_piece in adventurer.armor:
            armor_talent = armor_piece.name.lower().replace(" ", "_")
            if armor_talent not in talents:
                talents.append(armor_talent)

        # Every adventurer gets at least "general"
        if not talents:
            talents.append("general")

        return talents

    async def assess_quest_talents(self, draft: QuestDraft) -> list[str]:
        """Determine which talents a quest requires.

        Uses the LLM when available; falls back to keyword matching.
        """
        if self._llm is not None:
            return await self._llm_assess_quest_talents(draft)
        return self._keyword_assess_quest_talents(draft)

    def _keyword_assess_quest_talents(self, draft: QuestDraft) -> list[str]:
        """Determine quest talents via keyword matching against the description."""
        desc_lower = draft.description.lower()
        talents: list[str] = []
        for talent, keywords in self._talent_keywords.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}", desc_lower):
                    talents.append(talent)
                    break
        return talents

    async def _llm_assess_quest_talents(self, draft: QuestDraft) -> list[str]:
        """Use the LLM to decide which talents a quest needs."""
        assert self._llm is not None
        known = list(self._talent_keywords.keys())

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
            draft.title, draft.required_talents,
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
        missing = (
            []
            if has_general
            else [t for t in draft.required_talents if t not in all_talents]
        )
        feasible = len(missing) == 0 and len(matched) > 0
        logger.info(
            "Feasibility: feasible=%s matched=%d missing=%s",
            feasible, len(matched), missing,
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
        self, quest: Quest,
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
                    quest.id[:8], reason,
                )
            return accepted
        except (ValueError, KeyError):
            logger.debug("Could not parse verification response — accepting")
            return result.success

    # ── Helpers ────────────────────────────────────────────────────────

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
