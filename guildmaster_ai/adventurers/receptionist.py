from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from guildmaster_ai.core.messages import QuestDraft, QuestResult
from guildmaster_ai.core.utils import safe_parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete

logger = logging.getLogger("guildmaster.receptionist")

# Type alias for a callback that presents questions to a user and returns answers.
ClarificationCallback = Callable[[list[str]], Awaitable[dict[str, str]]]


class Receptionist:
    """Handles user intake, clarification, and result presentation."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        max_rounds: int = 3,
    ) -> None:
        self._llm = llm
        self._max_rounds = max_rounds

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
        draft = self._build_initial_draft(user_request)

        if self._llm is None:
            logger.debug("No LLM configured — returning raw draft")
            return draft

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
                questions = self._identify_gaps(draft)
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
                '"title" (concise, max 80 chars), "description" (detailed), '
                '"acceptance_criteria" (list of strings defining done). '
                "Do NOT assign talents or skills — that is the Guildmaster's job."
            ),
            user=user_request,
        )
        return self._parse_draft_response(content, user_request)

    @staticmethod
    def _parse_draft_response(response_content: str, fallback_text: str) -> QuestDraft:
        """Try to parse LLM JSON into a QuestDraft, falling back gracefully."""
        data = safe_parse_llm_json(response_content, context="draft_response")
        if data is not None:
            return QuestDraft(
                title=data.get("title", fallback_text[:80]),
                description=data.get("description", fallback_text),
                acceptance_criteria=data.get("acceptance_criteria", []),
            )
        return QuestDraft(title=fallback_text[:80], description=fallback_text)

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
