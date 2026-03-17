from __future__ import annotations

from abc import ABC, abstractmethod

from guildmaster_ai.core.messages import GuardVerdict


# TODO No logger here add logging
class BaseGuard(ABC):
    """Abstract base class for all guard (LLM-as-judge) implementations.

    Subclass this to create a custom guard with your own evaluation logic.
    The only required method is :meth:`evaluate`.

    Example::

        class MyGuard(BaseGuard):
            @property
            def name(self) -> str:
                return "my_guard"

            async def evaluate(self, content, criteria=None, context=None):
                # your custom logic
                return GuardVerdict(sender=self.name, verdict="pass", reason="ok")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this guard."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        content: str,
        criteria: list[str] | None = None,
        context: str | None = None,
    ) -> GuardVerdict:
        """Evaluate *content* and return a verdict with metrics.

        Parameters
        ----------
        content:
            The text to evaluate (typically the quest result summary).
        criteria:
            Optional acceptance criteria the content should satisfy.
        context:
            Optional original quest description for relevance/accuracy checks.
        """
        ...
