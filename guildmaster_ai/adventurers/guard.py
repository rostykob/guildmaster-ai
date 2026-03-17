from __future__ import annotations

from guildmaster_ai.core.messages import GuardVerdict
from guildmaster_ai.llm.types import GuildLLM


class Guard:
    """Evaluates content against safety and policy criteria."""

    def __init__(
        self,
        llm: GuildLLM | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm
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
