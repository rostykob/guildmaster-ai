from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

ArmorVerdict = Literal["pass", "warn", "block"]


class ArmorResult(BaseModel):
    """Result returned by an armor check."""

    verdict: ArmorVerdict
    message: str | None = None
    modified_content: str | None = None


class BaseArmor(ABC):
    """Abstract base class for all armor (guardrails)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    async def pre_process(self, content: str) -> ArmorResult:
        """Run before the LLM processes the content. Defaults to pass."""
        return ArmorResult(verdict="pass")

    async def post_process(self, content: str) -> ArmorResult:
        """Run after the LLM processes the content. Defaults to pass."""
        return ArmorResult(verdict="pass")
