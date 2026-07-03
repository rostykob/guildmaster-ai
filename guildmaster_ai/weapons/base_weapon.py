from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel


class WeaponSchema(BaseModel):
    """Schema describing a weapon's interface."""

    name: str
    description: str
    parameters: dict[str, Any]


class BaseWeapon(BaseTool):
    """Abstract base class for all weapons (tools).

    Extends LangChain ``BaseTool`` so weapons are natively compatible with
    LangChain agents, ``bind_tools()``, and ``create_agent()``.

    Subclasses must set ``name``, ``description``, ``args_schema``
    (a Pydantic model) and implement :meth:`execute`.
    """

    def _run(self, **kwargs: Any) -> str:
        """Sync execution — raises because this framework is async-first."""
        raise NotImplementedError("BaseWeapon is async-only. Use ainvoke().")

    async def _arun(self, **kwargs: Any) -> str:
        """Bridge to :meth:`execute` for LangChain async invocation."""
        kwargs.pop("run_manager", None)
        result = await self.execute(**kwargs)
        return json.dumps(result) if isinstance(result, dict) else str(result)

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the weapon and return a dict result.

        Override this in subclasses.
        """
        raise NotImplementedError

    def schema(self) -> WeaponSchema:  # type: ignore[override]
        """Return a ``WeaponSchema`` built from this weapon's properties."""
        if isinstance(self.args_schema, dict):
            params: dict[str, Any] = self.args_schema
        elif self.args_schema is not None:
            params = self.args_schema.model_json_schema()
        else:
            params = {"type": "object", "properties": {}}
        return WeaponSchema(
            name=self.name,
            description=self.description,
            parameters=params,
        )

    def to_tool_spec(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool specification."""
        return convert_to_openai_tool(self)
