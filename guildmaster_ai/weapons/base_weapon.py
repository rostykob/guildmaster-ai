from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class WeaponSchema(BaseModel):
    """Schema describing a weapon's interface."""

    name: str
    description: str
    parameters: dict[str, Any]


class BaseWeapon(ABC):
    """Abstract base class for all weapons (tools)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema dict describing the weapon's input parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]: ...

    def schema(self) -> WeaponSchema:
        """Return a ``WeaponSchema`` built from this weapon's properties."""
        return WeaponSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def to_tool_spec(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool specification."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
