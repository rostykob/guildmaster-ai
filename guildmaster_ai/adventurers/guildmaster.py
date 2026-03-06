from __future__ import annotations

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
from guildmaster_ai.llm.base_provider import BaseLLMProvider

# Keyword-to-talent mapping used for prompt-based talent inference.
_TALENT_KEYWORDS: dict[str, list[str]] = {
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
        llm_provider: BaseLLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self._roster: dict[str, BaseAdventurer] = {}
        self._llm = llm_provider
        self._model = model

    def register_adventurer(self, adventurer: BaseAdventurer) -> None:
        """Add an adventurer to the guild roster, assessing and assigning talents."""
        talents = self._assess_talents(adventurer)
        adventurer.grant_talents(talents)
        self._roster[adventurer.id] = adventurer

    def unregister_adventurer(self, adventurer_id: str) -> None:
        """Remove an adventurer from the guild roster."""
        self._roster.pop(adventurer_id, None)

    @property
    def roster(self) -> list[AdventurerProfile]:
        """Return profiles of all registered adventurers."""
        return [adv.profile() for adv in self._roster.values()]

    def _assess_talents(self, adventurer: BaseAdventurer) -> list[str]:
        """Infer talents from the adventurer's system prompt and equipped weapons."""
        talents: list[str] = []
        prompt_lower = adventurer.system_prompt.lower()

        # Extract talents from system prompt via keyword matching
        for talent, keywords in _TALENT_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}", prompt_lower):
                    talents.append(talent)
                    break

        # Derive talents from equipped weapons
        for weapon_name in adventurer._weapons:
            # Normalise weapon name into a talent (e.g. "web_search" stays "web_search")
            weapon_talent = weapon_name.lower().replace(" ", "_")
            if weapon_talent not in talents:
                talents.append(weapon_talent)

        # Every adventurer gets at least "general"
        if not talents:
            talents.append("general")

        return talents

    def check_feasibility(self, draft: QuestDraft) -> QuestFeasibilityReport:
        """Check whether the guild can staff a quest draft."""
        all_talents: set[str] = set()
        matched: list[AdventurerProfile] = []

        for adv in self._roster.values():
            adv_talents = set(adv.talents)
            all_talents.update(adv_talents)
            if set(draft.required_talents) & adv_talents:
                matched.append(adv.profile())

        # If no specific talents required, all adventurers are potential matches
        if not draft.required_talents:
            matched = [adv.profile() for adv in self._roster.values()]

        missing = [t for t in draft.required_talents if t not in all_talents]
        feasible = len(missing) == 0 and len(matched) > 0

        return QuestFeasibilityReport(
            sender="guildmaster",
            feasible=feasible,
            matched_adventurers=matched,
            missing_talents=missing,
            recommended_rank=draft.rank or QuestRank.E,
        )

    def match_adventurers(self, quest: Quest) -> list[BaseAdventurer]:
        """Return adventurers eligible for a quest by talent overlap."""
        required = set(quest.required_talents)
        eligible: list[BaseAdventurer] = []

        for adv in self._roster.values():
            if required and not (required & set(adv.talents)):
                continue
            eligible.append(adv)

        return eligible

    async def assign_quest(self, quest: Quest, board: QuestBoard) -> Party | None:
        """Match adventurers, form a party, and assign the quest on the board."""
        matched = self.match_adventurers(quest)
        if not matched:
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

    async def verify_result(self, quest: Quest, result: QuestResult) -> bool:
        """Verify a quest result. Placeholder — returns result.success."""
        return result.success
