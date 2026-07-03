from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from guildmaster_ai.weapons.base_weapon import BaseWeapon


class WebSearchInput(BaseModel):
    """Input schema for the web search weapon."""

    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return")


class WebSearchWeapon(BaseWeapon):
    """Placeholder web-search weapon."""

    name: str = "web_search"
    description: str = "Search the web for information"
    args_schema: type[BaseModel] = WebSearchInput

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query: str = kwargs["query"]
        return {
            "results": [],
            "query": query,
            "note": "Web search not configured. Provide a search backend.",
        }
