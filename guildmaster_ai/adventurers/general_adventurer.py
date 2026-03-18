from __future__ import annotations

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer


class GeneralAdventurer(BaseAdventurer):
    """A general-purpose adventurer capable of handling a wide range of tasks."""

    system_prompt = (
        "You are a versatile adventurer capable of handling general tasks. "
        "Use any tools at your disposal to accomplish the quest objective. "
        "Be thorough, accurate, and concise in your responses."
    )
