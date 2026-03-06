from __future__ import annotations

from guildmaster_ai.core.messages import GuardVerdict
from guildmaster_ai.llm.base_provider import BaseLLMProvider


class Guard:
    """Evaluates content against safety and policy criteria."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm_provider
        self._model = model

    async def evaluate(
        self,
        content: str,
        criteria: list[str] | None = None,
    ) -> GuardVerdict:
        """Evaluate content. Placeholder — returns pass verdict."""
        return GuardVerdict(
            sender="guard",
            verdict="pass",
            reason="Content passed all checks.",
        )
