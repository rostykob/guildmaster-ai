from __future__ import annotations

import logging

from guildmaster_ai.adventurers.base_guard import BaseGuard
from guildmaster_ai.core.messages import GuardMetrics, GuardVerdict
from guildmaster_ai.core.utils import safe_parse_llm_json
from guildmaster_ai.llm.types import GuildLLM, guild_complete

logger = logging.getLogger("guildmaster.guard")

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial LLM-as-judge. Evaluate the provided content and return \
ONLY a JSON object with the following fields:

- "verdict": one of "pass", "warn", "block"
- "reason": a brief explanation of your verdict
- "hallucination": float 0.0-1.0 (0 = no hallucination, 1 = entirely fabricated)
- "accuracy": float 0.0-1.0 (1 = fully accurate, 0 = completely wrong)
- "relevance": float 0.0-1.0 (1 = perfectly relevant, 0 = completely off-topic)
- "toxicity": float 0.0-1.0 (0 = harmless, 1 = highly toxic)

Guidelines for verdict:
- "pass": all metrics are acceptable (hallucination < 0.3, accuracy > 0.5, \
relevance > 0.5, toxicity < 0.3)
- "warn": at least one metric is marginal
- "block": content is toxic (> 0.7), heavily hallucinated (> 0.7), \
or completely inaccurate (< 0.2)

Respond with ONLY the JSON object, no markdown fences."""


class Guard(BaseGuard):
    """Default LLM-as-judge guard with hallucination, accuracy, relevance, and toxicity metrics.

    When no LLM is configured, falls back to a permissive pass-through.
    """

    def __init__(
        self,
        llm: GuildLLM | None = None,
    ) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "guard"

    async def evaluate(
        self,
        content: str,
        criteria: list[str] | None = None,
        context: str | None = None,
    ) -> GuardVerdict:
        """Evaluate content using the LLM as judge."""
        if self._llm is None:
            logger.debug("No LLM configured for guard — auto-passing")
            return GuardVerdict(
                sender=self.name,
                verdict="pass",
                reason="No LLM configured — auto-pass.",
            )

        user_parts = [f"Content to evaluate:\n{content}"]
        if criteria:
            user_parts.append("\nAcceptance criteria:\n" + "\n".join(f"- {c}" for c in criteria))
        if context:
            user_parts.append(f"\nOriginal request/context:\n{context}")

        user_prompt = "\n".join(user_parts)

        logger.info("Evaluating content with LLM judge")
        logger.debug("Guard prompt: %s", user_prompt[:200])

        try:
            raw = await guild_complete(
                self._llm,
                system=_JUDGE_SYSTEM_PROMPT,
                user=user_prompt,
            )
            return self._parse_response(raw)
        except Exception:
            # Guard fails open: LLM errors should not block quest execution.
            # Use warning (not exception) since this is an expected degradation path.
            logger.warning("Guard LLM call failed — defaulting to pass", exc_info=True)
            return GuardVerdict(
                sender=self.name,
                verdict="pass",
                reason="Guard evaluation failed — defaulting to pass.",
            )

    def _parse_response(self, raw: str) -> GuardVerdict:
        """Parse LLM JSON response into a GuardVerdict."""
        data = safe_parse_llm_json(raw, context="guard_evaluate")
        if data is None:
            return GuardVerdict(
                sender=self.name,
                verdict="pass",
                reason="Could not parse guard evaluation — defaulting to pass.",
            )

        metrics = GuardMetrics(
            hallucination=float(data.get("hallucination", 0.0)),
            accuracy=float(data.get("accuracy", 1.0)),
            relevance=float(data.get("relevance", 1.0)),
            toxicity=float(data.get("toxicity", 0.0)),
        )

        verdict = data.get("verdict", "pass")
        if verdict not in ("pass", "warn", "block"):
            verdict = "pass"

        logger.info(
            "Guard verdict=%s hallucination=%.2f accuracy=%.2f relevance=%.2f toxicity=%.2f",
            verdict,
            metrics.hallucination,
            metrics.accuracy,
            metrics.relevance,
            metrics.toxicity,
        )

        return GuardVerdict(
            sender=self.name,
            verdict=verdict,
            reason=data.get("reason", ""),
            metrics=metrics,
        )
