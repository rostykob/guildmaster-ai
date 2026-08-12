"""GeneralHero — a general-purpose hero capable of leading a party."""

from __future__ import annotations

from guildmaster_ai.adventurers.base_hero import BaseHero


class GeneralHero(BaseHero):
    """A general-purpose hero capable of leading a party and coordinating members.

    Use ``GeneralHero`` when you need a ready-made party leader with a sensible
    default system prompt.  Subclass ``BaseHero`` directly when you need a
    custom system prompt or specialised coordination logic.

    When members are recruited, the hero delegates subtasks to the most
    appropriate member via the deep agent's ``task`` tool.  Without members
    it behaves like a regular deep agent adventurer with scroll support.
    """

    system_prompt = (
        "You are a versatile hero and party leader. "
        "Coordinate your party members to accomplish the quest objective. "
        "Delegate subtasks to the most appropriate member using the task tool. "
        "Synthesize their results into a coherent final answer."
    )
