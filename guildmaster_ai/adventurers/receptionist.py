from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from guildmaster_ai.core.messages import QuestDraft, QuestResult, QuestStatusReport
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.core.utils import charter_block, parse_llm_json, safe_parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete

logger = logging.getLogger("guildmaster.receptionist")

# Type alias for a callback that presents questions to a user and returns answers.
ClarificationCallback = Callable[[list[str]], Awaitable[dict[str, str]]]

# Lookups injected by the Guild so the receptionist can answer status queries.
QuestLookup = Callable[[str], Quest]
ResultLookup = Callable[[str], QuestResult | None]

# Provider injected by the Guild: returns the weapon input schemas available
# across the roster, so intake can spot inputs the user forgot to supply.
ArmouryProvider = Callable[[], list[dict[str, Any]]]


class Receptionist:
    """Handles user intake, clarification, status queries, and result presentation."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        max_rounds: int = 3,
        charter: str = "",
    ) -> None:
        self._llm = llm
        self._max_rounds = max_rounds
        self._charter = charter
        self._quest_lookup: QuestLookup | None = None
        self._result_lookup: ResultLookup | None = None
        self._armoury: ArmouryProvider | None = None

    def connect_registry(
        self,
        quest_lookup: QuestLookup,
        result_lookup: ResultLookup,
    ) -> None:
        """Wire the guild's quest/result registries for status queries."""
        self._quest_lookup = quest_lookup
        self._result_lookup = result_lookup

    def connect_armoury(self, armoury: ArmouryProvider) -> None:
        """Wire a provider of roster weapon schemas for intake gap checks.

        With an armoury connected (and an LLM configured), ``intake`` can ask
        the user for weapon inputs the request is missing — e.g. a file path
        for a file-reading weapon — instead of letting the quest reach an
        adventurer that cannot execute it.
        """
        self._armoury = armoury

    def check_status(self, quest_id: str) -> QuestStatusReport:
        """Report the current status of a submitted quest by its UUID.

        Raises ``KeyError`` when the quest is unknown and ``RuntimeError``
        when the receptionist is not connected to a guild.
        """
        if self._quest_lookup is None or self._result_lookup is None:
            raise RuntimeError(
                "Receptionist is not connected to a guild — "
                "status queries require a Guild-managed receptionist."
            )
        quest = self._quest_lookup(quest_id)
        result = self._result_lookup(quest_id)
        return QuestStatusReport(
            sender="receptionist",
            quest_id=quest.id,
            title=quest.title,
            status=quest.status.value,
            finished=result is not None,
            success=result.success if result is not None else None,
            summary=result.summary if result is not None else None,
            failure_reason=result.failure_reason if result is not None else None,
        )

    async def intake(
        self,
        user_request: str,
        clarify: ClarificationCallback | None = None,
    ) -> QuestDraft:
        """Create a QuestDraft from a user request.

        If an LLM is configured the request is refined via the LLM.
        When *clarify* is supplied and the draft looks incomplete, up to
        *max_rounds* clarification rounds are performed before returning.
        """
        logger.info("Intake started for request: %s", user_request[:100])
        if self._llm is None:
            logger.debug("No LLM configured — returning raw draft")
            return self._build_initial_draft(user_request)

        # LLM-assisted refinement
        logger.info("Refining quest draft via LLM")
        draft = await self._refine_draft(user_request)
        logger.debug(
            "Refined draft: title=%r talents=%s criteria=%d",
            draft.title,
            draft.required_talents,
            len(draft.acceptance_criteria),
        )

        # Clarification loop
        if clarify is not None:
            for round_num in range(self._max_rounds):
                questions = await self._gap_questions(draft)
                if not questions:
                    logger.debug("No gaps found — skipping clarification")
                    break
                logger.info("Clarification round %d: %d questions", round_num + 1, len(questions))
                answers = await clarify(questions)
                draft = self._apply_clarifications(draft, answers)

        logger.info("Intake complete: %r", draft.title)
        return draft

    async def present_result(self, result: QuestResult) -> str:
        """Format a QuestResult for user display."""
        status = "completed successfully" if result.success else "failed"
        lines = [
            f"Quest {status}.",
            f"Summary: {result.summary}",
        ]
        if result.failure_reason:
            lines.append(f"Reason: {result.failure_reason}")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_initial_draft(user_request: str) -> QuestDraft:
        """Build a minimal draft from raw user text."""
        return QuestDraft(
            title=user_request[:80],
            description=user_request,
        )

    async def _refine_draft(self, user_request: str) -> QuestDraft:
        """Use the LLM to extract a structured draft from the user request."""
        if self._llm is None:
            return self._build_initial_draft(user_request)

        content = await guild_complete(
            self._llm,
            system=(
                "You are a guild receptionist. Your job is to understand "
                "the user's request and produce a clear, well-structured quest. "
                "Respond with ONLY a JSON object containing: "
                '"title" (concise, max 80 chars) and '
                '"acceptance_criteria" (list of strings defining done). '
                "Capture EVERY output constraint the user states — format, "
                "length, tone, or style (e.g. 'reply in one word', 'as a table', "
                "'formal tone') — as its own acceptance criterion, using the "
                "user's own wording. Do NOT rewrite or paraphrase the request "
                "itself, and do NOT assign talents or skills — that is the "
                "Guildmaster's job."
                f"{charter_block(self._charter)}"
            ),
            user=user_request,
        )
        return self._parse_draft_response(content, user_request)

    @staticmethod
    def _parse_draft_response(response_content: str, fallback_text: str) -> QuestDraft:
        """Try to parse LLM JSON into a QuestDraft, falling back gracefully.

        The ``description`` always stays the user's original request verbatim —
        it is the instruction the adventurer executes, so paraphrasing it would
        silently drop output constraints (e.g. "reply in one word"). The LLM
        only contributes structure: a clean ``title`` and ``acceptance_criteria``
        (which is where such constraints are captured).
        """
        data = safe_parse_llm_json(response_content, context="draft_response")
        if data is not None:
            return QuestDraft(
                title=data.get("title", fallback_text[:80]),
                description=fallback_text,
                acceptance_criteria=data.get("acceptance_criteria", []),
            )
        return QuestDraft(title=fallback_text[:80], description=fallback_text)

    async def _gap_questions(self, draft: QuestDraft) -> list[str]:
        """Collect all clarification questions for *draft*.

        Combines the static completeness heuristics with the LLM-driven
        missing-weapon-input check (which only runs when both an LLM and an
        armoury provider are available).
        """
        questions = self._identify_gaps(draft)
        questions.extend(await self._identify_missing_inputs(draft))
        return questions

    async def _identify_missing_inputs(self, draft: QuestDraft) -> list[str]:
        """Ask the LLM which required weapon inputs the request doesn't provide.

        Compares the quest against the input schemas of the weapons available
        in the guild (via the connected armoury) and returns one question per
        missing required input. Returns ``[]`` when no LLM or armoury is
        configured, when there are no weapon schemas, or when nothing is
        missing — so the extra LLM call only happens where it can pay off.
        """
        if self._llm is None or self._armoury is None:
            return []
        schemas = self._armoury()
        if not schemas:
            return []

        raw = await guild_complete(
            self._llm,
            system=(
                "You are a guild receptionist checking whether a quest request "
                "provides the inputs the guild's tools need. Given the quest and "
                "the tool input schemas, decide whether any REQUIRED input a "
                "tool clearly needed for this quest is missing from the request "
                "(e.g. a file path, a URL, a search topic). Respond with ONLY a "
                "JSON array of short questions to ask the user — one per "
                "missing input, at most 3. Respond with [] when the request is "
                "self-sufficient or no tool is relevant. Do NOT ask about "
                "optional inputs or details the adventurer can decide itself."
                f"{charter_block(self._charter)}"
            ),
            user=(
                f"Quest: {draft.title}\n"
                f"Description: {draft.description}\n\n"
                f"Tool input schemas:\n{json.dumps(schemas, indent=2)}"
            ),
        )

        try:
            data = parse_llm_json(raw)
        except (ValueError, KeyError):
            logger.warning("Could not parse missing-input check: %.200s", raw)
            return []
        if not isinstance(data, list):
            return []
        questions = [str(q) for q in data if q][:3]
        if questions:
            logger.info("Missing weapon inputs detected: %d question(s)", len(questions))
        return questions

    @staticmethod
    def _identify_gaps(draft: QuestDraft) -> list[str]:
        """Return clarification questions for any incomplete fields."""
        questions: list[str] = []
        if len(draft.description) < 20:
            questions.append(
                "The quest description is very brief. "
                "Could you provide more detail about what you need?"
            )
        if not draft.acceptance_criteria:
            questions.append(
                "What criteria should be met for this quest to be considered complete?"
            )
        return questions

    @staticmethod
    def _apply_clarifications(draft: QuestDraft, answers: dict[str, str]) -> QuestDraft:
        """Merge clarification answers back into the draft."""
        extra_text = " ".join(answers.values())
        return QuestDraft(
            title=draft.title,
            description=f"{draft.description}\n\nAdditional details: {extra_text}"
            if extra_text
            else draft.description,
            required_talents=draft.required_talents,
            acceptance_criteria=draft.acceptance_criteria,
        )
