from __future__ import annotations

import logging
from typing import Any

from guildmaster_ai.core.messages import QuestObservation, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.core.utils import safe_parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete
from guildmaster_ai.memory.chroma_store import ChromaStore
from guildmaster_ai.memory.sqlite_store import SQLiteStore

logger = logging.getLogger("guildmaster.librarian")

# guild_state key holding the rolling chronicle (see Librarian.get_chronicle).
_CHRONICLE_KEY = "librarian_chronicle"
# Hard cap so the chronicle stays a cheap prompt injection, never a transcript.
_CHRONICLE_MAX_CHARS = 2000

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
        store: SQLiteStore | None = None,
    ) -> None:
        self._llm = llm
        self._observations: list[QuestObservation] = []
        self._vector_store = vector_store
        self._store = store
        self._chronicle: str | None = None  # lazy-loaded cache of the stored chronicle

    async def archive(self, quest: Quest, result: QuestResult) -> dict[str, Any]:
        """Produce a summary dict and generate observations from the quest."""
        logger.info("Archiving quest %s: %r", quest.id[:8], quest.title)
        observation = await self.analyze_quest(quest, result)
        self._observations.append(observation)
        logger.debug("Observation tags: %s", observation.tags)

        # Persist observation durably to SQLite (always) so it survives the
        # process and is queryable on demand.
        if self._store is not None:
            await self._store.save_observation(
                obs_id=observation.id,
                quest_id=observation.quest_id,
                summary=observation.summary,
                tags=observation.tags,
                lessons_learned=observation.lessons_learned,
            )

        # Persist to the vector store for semantic search in future quests.
        if self._vector_store is not None:
            await self._store_observation(observation)

        # Fold the observation into the rolling chronicle so future quests get
        # a compact lessons summary instead of raw observation chunks.
        await self._update_chronicle(observation)

        return {
            "quest_id": quest.id,
            "title": quest.title,
            "status": quest.status.value,
            "success": result.success,
            "summary": result.summary,
            "observation": observation.model_dump(),
        }

    # ── Chronicle (rolling lessons summary) ───────────────────────────

    async def get_chronicle(self) -> str:
        """Return the rolling summary of lessons distilled from past quests.

        This is the cheap read side: a single stored string, no LLM call and
        no vector query. The guildmaster injects it into triage/planning
        prompts. Updated on the write side by :meth:`archive`.
        """
        if self._chronicle is None:
            if self._store is not None:
                self._chronicle = await self._store.get_guild_state(_CHRONICLE_KEY) or ""
            else:
                self._chronicle = ""
        return self._chronicle

    async def _update_chronicle(self, observation: QuestObservation) -> None:
        """Fold *observation* into the chronicle (runs on the archival path).

        With an LLM the old chronicle and the new observation are distilled
        into a fresh summary; without one, lesson lines are appended and the
        oldest are dropped. Either way the result is capped so injecting it
        into prompts stays cheap.
        """
        current = await self.get_chronicle()

        if self._llm is not None:
            updated = await guild_complete(
                self._llm,
                system=(
                    "You are a guild librarian maintaining the guild chronicle — "
                    "a compact summary of durable lessons from past quests, used "
                    "as planning context for future quests. Merge the new "
                    "observation into the chronicle: keep only actionable, "
                    "recurring lessons (what worked, what failed and why, which "
                    "adventurer/skill fits what), drop one-off trivia, and stay "
                    "under 15 bullet points. Respond with ONLY the updated "
                    "chronicle as a markdown bullet list."
                ),
                user=(
                    f"Current chronicle:\n{current or '(empty)'}\n\n"
                    f"New observation:\n"
                    f"Summary: {observation.summary}\n"
                    f"Tags: {observation.tags}\n"
                    f"Lessons: {observation.lessons_learned}"
                ),
            )
        else:
            lines = [ln for ln in current.splitlines() if ln.strip()]
            additions = observation.lessons_learned or [observation.summary]
            lines.extend(f"- {a}" for a in additions if a)
            updated = "\n".join(lines[-15:])

        self._chronicle = updated.strip()[:_CHRONICLE_MAX_CHARS]
        if self._store is not None:
            await self._store.save_guild_state(_CHRONICLE_KEY, self._chronicle)
        logger.debug("Chronicle updated (%d chars)", len(self._chronicle))

    async def analyze_quest(
        self,
        quest: Quest,
        result: QuestResult,
    ) -> QuestObservation:
        """Analyze a quest's history and result to produce a tagged observation."""
        if self._llm is not None:
            logger.debug("Analyzing quest via LLM")
            return await self._llm_analyze(quest, result)
        logger.debug("Analyzing quest via rule-based heuristics")
        return self._rule_based_analyze(quest, result)

    async def analyze_batch(
        self,
        quests: list[tuple[Quest, QuestResult]],
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

    async def recent_observations(
        self,
        quest_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch persisted observations from the store (on-demand, newest first).

        Reads durable history from SQLite rather than the in-memory list, so it
        works across processes. Falls back to the in-memory observations when no
        store is configured.
        """
        if self._store is not None:
            return await self._store.get_observations(quest_id=quest_id, limit=limit)
        obs = self.query_observations(quest_id=quest_id)
        return [o.model_dump() for o in obs[:limit]]

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
            query,
            n_results=n_results,
            where=where,
        )

    async def search_similar_quests(
        self,
        quest_description: str,
        n_results: int = 3,
    ) -> list[QuestObservation]:
        """Search for past observations similar to the given quest description.

        Uses the vector store for semantic search when available, falling back
        to keyword matching against in-memory observations.
        """
        logger.debug(
            "Searching for similar quests (n_results=%d, vector_store=%s)",
            n_results,
            self._vector_store is not None,
        )

        if self._vector_store is not None:
            raw_results = await self._vector_store.query(
                quest_description,
                n_results=n_results,
            )
            observations: list[QuestObservation] = []
            for doc in raw_results:
                # ChromaStore returns dicts with metadata nested under
                # "metadata" key; extract tags from there or top-level.
                meta = doc.get("metadata", {})
                raw_tags = meta.get("tags", doc.get("tags", ""))
                tags = raw_tags.split(",") if isinstance(raw_tags, str) else raw_tags
                observations.append(
                    QuestObservation(
                        sender="librarian",
                        quest_id=meta.get("quest_id", doc.get("quest_id", "")),
                        summary=doc.get("document", doc.get("summary", "")),
                        tags=tags,
                    )
                )
            logger.debug("Vector search returned %d results", len(observations))
            return observations[:n_results]

        # Fallback: keyword matching against in-memory observations
        desc_words = quest_description.lower().split()
        scored: list[tuple[int, QuestObservation]] = []
        for obs in self._observations:
            summary_lower = obs.summary.lower()
            score = sum(1 for word in desc_words if word in summary_lower)
            if score > 0:
                scored.append((score, obs))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [obs for _, obs in scored[:n_results]]
        logger.debug("Keyword search returned %d results", len(results))
        return results

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
        transition_count = sum(1 for h in quest.history if h.event_type == "transition")
        if transition_count > 5:
            tags.append("pattern:complex_lifecycle")
            lessons.append("Quest went through many state transitions.")

        # Check for retries (FAILED -> POSTED path)
        failed_count = sum(
            1
            for h in quest.history
            if h.event_type == "transition" and h.payload.get("to") == "failed"
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
        self,
        quest: Quest,
        result: QuestResult,
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
        self,
        response_content: str,
        quest: Quest,
        result: QuestResult,
    ) -> QuestObservation:
        """Parse LLM response into a QuestObservation, falling back to rules."""
        data = safe_parse_llm_json(response_content, context="observation_response")
        if data is not None:
            return QuestObservation(
                sender="librarian",
                quest_id=quest.id,
                summary=str(data.get("summary", "")),
                tags=self._coerce_tags(data.get("tags", [])),
                lessons_learned=self._coerce_str_list(data.get("lessons_learned", [])),
            )
        return self._rule_based_analyze(quest, result)

    @staticmethod
    def _coerce_tags(raw: Any) -> list[str]:
        """Normalise LLM-provided tags into ``category:value`` strings.

        Models often return the taxonomy as dicts (e.g. ``{"talent": "general"}``
        or ``{"outcome": "success"}``) instead of the expected ``"talent:general"``
        strings. Flatten those to strings so validation never rejects a whole
        observation over tag shape.
        """
        if not isinstance(raw, list):
            raw = [raw]
        tags: list[str] = []
        for item in raw:
            if isinstance(item, str):
                tags.append(item)
            elif isinstance(item, dict):
                for key, value in item.items():
                    # {"outcome": "success"} / {"category": "bug"} → bare value;
                    # {"talent": "general"} → "talent:general".
                    if key in ("outcome", "category"):
                        tags.append(str(value))
                    else:
                        tags.append(f"{key}:{value}")
            elif item is not None:
                tags.append(str(item))
        return tags

    @staticmethod
    def _coerce_str_list(raw: Any) -> list[str]:
        """Coerce an arbitrary LLM value into a list of strings."""
        if not isinstance(raw, list):
            raw = [raw]
        return [str(item) for item in raw if item is not None]

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
