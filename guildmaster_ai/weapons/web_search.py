from __future__ import annotations

from typing import Any

from guildmaster_ai.weapons.base_weapon import BaseWeapon


class WebSearchWeapon(BaseWeapon):
    """Placeholder web-search weapon."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query: str = kwargs["query"]
        return {
            "results": [],
            "query": query,
            "note": "Web search not configured. Provide a search backend.",
        }
