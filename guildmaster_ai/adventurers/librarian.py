from __future__ import annotations

import json
from typing import Any

from guildmaster_ai.core.messages import QuestObservation, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.llm.types import GuildLLM, guild_complete


class Librarian:
    """Archives quest results, analyses history, and produces tagged observations."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._observations: list[QuestObservation] = []

    async def archive(
        self, quest: Quest, result: QuestResult
    ) -> dict[str, Any]:
        """Produce a summary dict and generate observations from the quest."""
        observation = await self.analyze_quest(quest, result)
        self._observations.append(observation)

        return {
            "quest_id": quest.id,
            "title": quest.title,
            "status": quest.status.value,
            "success": result.success,
            "summary": result.summary,
            "observation": observation.model_dump(),
        }

    async def analyze_quest(
        self, quest: Quest, result: QuestResult
    ) -> QuestObservation:
        """Analyze a quest's history and result to produce a tagged observation."""
        if self._llm is not None:
            return await self._llm_analyze(quest, result)
        return self._rule_based_analyze(quest, result)

    async def analyze_batch(
        self, quests: list[tuple[Quest, QuestResult]]
    ) -> list[QuestObservation]:
        """Analyze multiple quests and return observations for each."""
        observations: list[QuestObservation] = []
        for quest, result in quests:
            obs = await self.analyze_quest(quest, result)
            observations.append(obs)
            self._observations.append(obs)
        return observations

    def query_observations(
        self,
        tags: list[str] | None = None,
        quest_id: str | None = None,
    ) -> list[QuestObservation]:
        """Query stored observations, optionally filtered by tags or quest id."""
        results = self._observations
        if quest_id is not None:
            results = [o for o in results if o.quest_id == quest_id]
        if tags:
            tag_set = set(tags)
            results = [o for o in results if tag_set & set(o.tags)]
        return results

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _rule_based_analyze(quest: Quest, result: QuestResult) -> QuestObservation:
        """Generate an observation using rule-based heuristics (no LLM)."""
        tags: list[str] = []
        lessons: list[str] = []

        # Tag by outcome
        if result.success:
            tags.append("success")
        else:
            tags.append("failure")
            if result.failure_reason:
                tags.append(f"reason:{result.failure_reason}")

        # Tag by required talents
        for talent in quest.required_talents:
            tags.append(f"talent:{talent}")

        # Analyze history for patterns
        transition_count = sum(
            1 for h in quest.history if h.event_type == "transition"
        )
        if transition_count > 5:
            tags.append("complex_lifecycle")
            lessons.append("Quest went through many state transitions.")

        # Check for retries (FAILED -> POSTED path)
        failed_count = sum(
            1
            for h in quest.history
            if h.event_type == "transition" and h.payload.get("to") == "failed"
        )
        if failed_count > 0:
            tags.append("had_failures")
            lessons.append(f"Quest failed {failed_count} time(s) before resolution.")

        summary = (
            f"Quest '{quest.title}' {'succeeded' if result.success else 'failed'}. "
            f"{len(quest.history)} history events recorded."
        )

        return QuestObservation(
            sender="librarian",
            quest_id=quest.id,
            summary=summary,
            tags=tags,
            lessons_learned=lessons,
        )

    async def _llm_analyze(
        self, quest: Quest, result: QuestResult
    ) -> QuestObservation:
        """Use the LLM to generate a richer observation."""
        assert self._llm is not None

        history_text = "\n".join(
            f"- [{h.timestamp.isoformat()}] {h.actor}: {h.event_type} {h.payload}"
            for h in quest.history
        )

        content = await guild_complete(
            self._llm,
            system=(
                "You are a guild librarian. Analyze the quest history and result, "
                "then respond with ONLY a JSON object containing: "
                '"summary" (string), "tags" (list of strings), '
                '"lessons_learned" (list of strings). '
                "Tags should categorize the quest outcome, skills used, and patterns."
            ),
            user=(
                f"Quest: {quest.title}\n"
                f"Description: {quest.description}\n"
                f"Required talents: {quest.required_talents}\n"
                f"Status: {quest.status.value}\n"
                f"Success: {result.success}\n"
                f"Result summary: {result.summary}\n"
                f"Failure reason: {result.failure_reason or 'N/A'}\n\n"
                f"History:\n{history_text}"
            ),
        )
        return self._parse_observation_response(content, quest, result)

    def _parse_observation_response(
        self, response_content: str, quest: Quest, result: QuestResult
    ) -> QuestObservation:
        """Parse the LLM response into a QuestObservation, falling back to rules."""
        try:
            text = response_content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
            return QuestObservation(
                sender="librarian",
                quest_id=quest.id,
                summary=data.get("summary", ""),
                tags=data.get("tags", []),
                lessons_learned=data.get("lessons_learned", []),
            )
        except (json.JSONDecodeError, KeyError):
            return self._rule_based_analyze(quest, result)
