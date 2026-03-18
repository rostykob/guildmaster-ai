from __future__ import annotations

import logging
from typing import Any

from guildmaster_ai.core.messages import QuestObservation, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.core.utils import parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete
from guildmaster_ai.memory.chroma_store import ChromaStore

logger = logging.getLogger("guildmaster.librarian")

# Structured tag taxonomy for quest observations.
# Tags follow the pattern "category:value" for machine-queryable indexing.
#
# Outcome:   success, failure
# Category:  observation, bug, know_how, rule, lesson_learned
# Reason:    reason:<detail>         — why the quest failed
# Talent:    talent:<name>           — talents used / required
# Pattern:   pattern:<name>          — recurring patterns detected
# Missing:   missing:<talent>        — talent that was needed but absent


class Librarian:
    """Archives quest results, analyses history, and produces tagged observations."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        vector_store: ChromaStore | None = None,
    ) -> None:
        self._llm = llm
        self._observations: list[QuestObservation] = []
        self._vector_store = vector_store

    async def archive(self, quest: Quest, result: QuestResult) -> dict[str, Any]:
        """Produce a summary dict and generate observations from the quest."""
        logger.info("Archiving quest %s: %r", quest.id[:8], quest.title)
        observation = await self.analyze_quest(quest, result)
        self._observations.append(observation)
        logger.debug("Observation tags: %s", observation.tags)

        # Persist to vector store for semantic search in future quests
        if self._vector_store is not None:
            await self._store_observation(observation)

        return {
            "quest_id": quest.id,
            "title": quest.title,
            "status": quest.status.value,
            "success": result.success,
            "summary": result.summary,
            "observation": observation.model_dump(),
        }

    async def analyze_quest(
        self, quest: Quest, result: QuestResult,
    ) -> QuestObservation:
        """Analyze a quest's history and result to produce a tagged observation."""
        if self._llm is not None:
            logger.debug("Analyzing quest via LLM")
            return await self._llm_analyze(quest, result)
        logger.debug("Analyzing quest via rule-based heuristics")
        return self._rule_based_analyze(quest, result)

    async def analyze_batch(
        self, quests: list[tuple[Quest, QuestResult]],
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

    async def search_observations(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over archived observations via the vector store."""
        if self._vector_store is None:
            logger.debug("No vector store — falling back to tag query")
            return [o.model_dump() for o in self.query_observations(tags=tags)]

        where = {"tags": {"$in": tags}} if tags else None
        return await self._vector_store.query(
            query, n_results=n_results, where=where,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _rule_based_analyze(quest: Quest, result: QuestResult) -> QuestObservation:
        """Generate an observation using rule-based heuristics (no LLM)."""
        tags: list[str] = []
        lessons: list[str] = []

        # Outcome tag
        if result.success:
            tags.append("success")
            tags.append("observation")
        else:
            tags.append("failure")
            if result.failure_reason:
                tags.append(f"reason:{result.failure_reason}")
            tags.append("lesson_learned")

        # Tag by required talents
        for talent in quest.required_talents:
            tags.append(f"talent:{talent}")

        # Analyze history for patterns
        transition_count = sum(
            1 for h in quest.history if h.event_type == "transition"
        )
        if transition_count > 5:
            tags.append("pattern:complex_lifecycle")
            lessons.append("Quest went through many state transitions.")

        # Check for retries (FAILED -> POSTED path)
        failed_count = sum(
            1
            for h in quest.history
            if h.event_type == "transition"
            and h.payload.get("to") == "failed"
        )
        if failed_count > 0:
            tags.append("pattern:had_failures")
            lessons.append(
                f"Quest failed {failed_count} time(s) before resolution.",
            )

        # Detect missing talents from failure reasons
        if result.failure_reason and "talent" in result.failure_reason.lower():
            tags.append("bug")

        summary = (
            f"Quest '{quest.title}' "
            f"{'succeeded' if result.success else 'failed'}. "
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
        self, quest: Quest, result: QuestResult,
    ) -> QuestObservation:
        """Use the LLM to generate a richer observation."""
        assert self._llm is not None

        history_text = "\n".join(
            f"- [{h.timestamp.isoformat()}] {h.actor}: "
            f"{h.event_type} {h.payload}"
            for h in quest.history
        )

        content = await guild_complete(
            self._llm,
            system=(
                "You are a guild librarian. Analyze the quest and produce a "
                "structured observation. Respond with ONLY a JSON object:\n"
                '{"summary": "...", "tags": [...], "lessons_learned": [...]}\n\n'
                "Tag taxonomy (use these patterns):\n"
                "- Outcome: 'success' or 'failure'\n"
                "- Category: 'observation', 'bug', 'know_how', 'rule', "
                "'lesson_learned'\n"
                "- Prefixed: 'reason:<detail>', 'talent:<name>', "
                "'pattern:<name>', 'missing:<talent>'\n\n"
                "Apply multiple tags. Always include an outcome tag and "
                "at least one category tag. If the quest failed, include a "
                "'reason:' tag and identify any missing talents."
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
        self, response_content: str, quest: Quest, result: QuestResult,
    ) -> QuestObservation:
        """Parse LLM response into a QuestObservation, falling back to rules."""
        try:
            data = parse_llm_json(response_content)
            return QuestObservation(
                sender="librarian",
                quest_id=quest.id,
                summary=data.get("summary", ""),
                tags=data.get("tags", []),
                lessons_learned=data.get("lessons_learned", []),
            )
        except (ValueError, KeyError):
            return self._rule_based_analyze(quest, result)

    async def _store_observation(self, obs: QuestObservation) -> None:
        """Persist an observation to the vector store."""
        assert self._vector_store is not None
        metadata: dict[str, Any] = {
            "quest_id": obs.quest_id,
            "tags": ",".join(obs.tags),
        }
        try:
            await self._vector_store.store(
                doc_id=obs.id,
                content=obs.summary,
                metadata=metadata,
            )
            logger.debug("Stored observation %s in vector store", obs.id[:8])
        except Exception:
            logger.exception("Failed to store observation in vector store")
