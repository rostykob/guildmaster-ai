from __future__ import annotations

import hashlib
import logging
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from guildmaster_ai.adventurers.vitality import VitalityMiddleware
from guildmaster_ai.armor.base_armor import BaseArmor
from guildmaster_ai.core.exceptions import AdventurerDefeatedError, ArmorBlockedError
from guildmaster_ai.core.messages import AdventurerProfile, QuestResult
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.llm.types import GuildLLM
from guildmaster_ai.weapons.base_weapon import BaseWeapon


class BaseAdventurer:
    """Abstract base class for all adventurer agents.

    Uses weapons (tools) and armor (middleware) only.  For scroll support,
    use :class:`DeepAdventurer` instead.

    Subclasses **must** provide a ``system_prompt``.  This can be either a
    plain class variable or a ``@property`` — both patterns are supported::

        # Option 1 — class variable (simplest)
        class MyAdventurer(BaseAdventurer):
            system_prompt = "You are a helpful pirate."

        # Option 2 — property (dynamic prompts)
        class MyAdventurer(BaseAdventurer):
            @property
            def system_prompt(self) -> str:
                return f"You are {self.name}."
    """

    system_prompt: str = ""
    """The system prompt sent to the LLM.  Set as a class variable or property."""

    DEFAULT_AP: int = 20
    """Default action points — max tool calls per quest before failing."""

    DEFAULT_HP: int = 3
    """Default hit points — max tool errors (retries) per quest before failing."""

    def __init__(
        self,
        adventurer_id: str | None = None,
        name: str = "",
        llm: GuildLLM | None = None,
        ap: int | None = None,
        hp: int | None = None,
    ) -> None:
        # Check for system_prompt — allow property to be resolved after __init__
        has_prompt = isinstance(type(self).__dict__.get("system_prompt"), property) or bool(
            self.system_prompt
        )
        if not has_prompt:
            raise TypeError(
                f"{type(self).__name__} must set a 'system_prompt' class variable or property."
            )
        self.id = adventurer_id or str(uuid4())
        self.name = name
        self.ap = ap if ap is not None else type(self).DEFAULT_AP
        self.hp = hp if hp is not None else type(self).DEFAULT_HP
        self._llm = llm
        self._weapons: dict[str, BaseWeapon] = {}
        self._armor: list[BaseArmor] = []
        self._talents: list[str] = []
        self._logger = logging.getLogger(f"guildmaster.adventurer.{type(self).__name__}")

    # ── Public accessors ────────────────────────────────────────────────

    @property
    def weapons(self) -> dict[str, BaseWeapon]:
        """Return the equipped weapons keyed by name."""
        return dict(self._weapons)

    @property
    def weapon_names(self) -> list[str]:
        """Return a list of equipped weapon names."""
        return list(self._weapons.keys())

    @property
    def armor(self) -> list[BaseArmor]:
        """Return the worn armor pieces."""
        return list(self._armor)

    @property
    def llm(self) -> GuildLLM | None:
        """Return the configured LLM, or ``None``."""
        return self._llm

    @llm.setter
    def llm(self, value: GuildLLM | None) -> None:
        self._llm = value

    # ── Talents ───────────────────────────────────────────────────────

    @property
    def talents(self) -> list[str]:
        """Return the list of talents assigned to this adventurer."""
        return list(self._talents)

    def grant_talents(self, talents: list[str]) -> None:
        """Add talents to this adventurer (called by guildmaster on registration)."""
        added: list[str] = []
        for t in talents:
            if t not in self._talents:
                self._talents.append(t)
                added.append(t)
        if added:
            self._logger.info("Granted talents: %s", added)

    @property
    def config_hash(self) -> str:
        """Deterministic hash of this adventurer's configuration."""
        parts = [
            type(self).__name__,
            ",".join(sorted(self.weapon_names)),
            ",".join(sorted(a.name for a in self.armor)),
            hashlib.md5(self.system_prompt.encode()).hexdigest()[:8],
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Spawning ──────────────────────────────────────────────────────

    def spawn(self) -> BaseAdventurer:
        """Create a fresh instance with the same configuration.

        The new instance shares the same class, name, LLM, weapons, armor,
        and talents — but gets its own identity and clean state.  The Guild
        calls this before every quest execution so no mutable state leaks
        between quests.
        """
        clone = self.__class__(name=self.name, llm=self._llm, ap=self.ap, hp=self.hp)
        for weapon in self._weapons.values():
            clone.equip_weapon(weapon)
        for armor_piece in self._armor:
            clone.wear_armor(armor_piece)
        clone.grant_talents(self._talents)
        return clone

    # ── Execute ────────────────────────────────────────────────────────

    def _build_agent(self) -> Any:
        """Build a LangChain agent graph for quest execution.

        Uses ``create_agent`` from ``langchain`` which creates a tool-calling
        ReAct graph.  Weapons become LangChain tools; armor pieces are injected
        as middleware that can intercept model calls and tool invocations.
        """
        if self._llm is None:
            raise RuntimeError("No LLM configured for this adventurer.")

        tools: list[BaseWeapon] = list(self._weapons.values())

        return create_agent(
            model=self._llm,
            tools=tools or None,
            system_prompt=self.system_prompt,
            middleware=[*self._armor, self._vitality()],
            name=self.name or self.id,
        )

    def _vitality(self) -> VitalityMiddleware:
        """Build a fresh AP/HP budget middleware for one quest execution."""
        return VitalityMiddleware(ap=self.ap, hp=self.hp, adventurer=self.name or self.id)

    def _extract_final_text(self, messages: list[Any]) -> str:
        """Extract the final AI text content from agent output messages."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                return content
        return ""

    @staticmethod
    def _serialize_transcript(messages: list[Any]) -> list[dict[str, Any]]:
        """Serialize LangChain agent messages into role/content dicts.

        Produces a durable, provider-agnostic transcript the guild can persist
        to memory. LangChain message ``type`` (``human``/``ai``/``tool``/
        ``system``) becomes the ``role``.
        """
        transcript: list[dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg, "type", None) or type(msg).__name__
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            transcript.append({"role": role, "content": content})
        return transcript

    async def execute(self, quest: Quest) -> QuestResult:
        """Execute a quest using a LangChain agent.

        Builds an agent via :func:`langchain.agents.create_agent` with
        weapons as tools and armor pieces as middleware.
        """
        self._logger.info("Starting quest %s: %r", quest.id[:8], quest.title)

        agent = self._build_agent()
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=quest.description)]},
            )
        except ArmorBlockedError as exc:
            self._logger.warning(
                "Quest %s blocked by armor %r: %s",
                quest.id[:8],
                exc.armor_name,
                exc.message,
            )
            return QuestResult(
                sender=self.name or self.id,
                quest_id=quest.id,
                success=False,
                summary=f"Blocked by armor {exc.armor_name}: {exc.message}",
                failure_reason="armor_blocked",
            )
        except AdventurerDefeatedError as exc:
            self._logger.warning("Quest %s failed — %s", quest.id[:8], exc)
            return QuestResult(
                sender=self.name or self.id,
                quest_id=quest.id,
                success=False,
                summary=str(exc),
                failure_reason=f"{exc.stat}_depleted",
            )

        summary = self._extract_final_text(result["messages"])

        self._logger.info("Quest %s completed", quest.id[:8])
        return QuestResult(
            sender=self.name or self.id,
            quest_id=quest.id,
            success=True,
            summary=summary,
            transcript=self._serialize_transcript(result["messages"]),
        )

    # ── Convenience helpers ─────────────────────────────────────────────

    async def _llm_complete(self, user_message: str, *, system: str | None = None) -> str:
        """Direct LLM call without agent graph. For custom ``execute()`` overrides.

        Sends a single system + user message pair to the LLM and returns the
        text reply.  Does not create an agent, use tools, or track conversation
        history.
        """
        if self._llm is None:
            raise RuntimeError("No LLM configured for this adventurer.")
        from guildmaster_ai.llm.types import guild_complete

        return await guild_complete(
            self._llm,
            system=system or self.system_prompt,
            user=user_message,
        )

    # ── Equipment ─────────────────────────────────────────────────────

    def equip_weapon(self, weapon: BaseWeapon) -> None:
        """Add a weapon to this adventurer's loadout."""
        self._weapons[weapon.name] = weapon
        self._logger.info("Equipped weapon: %s", weapon.name)

    def wear_armor(self, armor: BaseArmor) -> None:
        """Add armor to this adventurer's defenses."""
        self._armor.append(armor)
        self._logger.info("Wore armor: %s", armor.name)

    def profile(self) -> AdventurerProfile:
        """Return the public profile for this adventurer."""
        return AdventurerProfile(
            id=self.id,
            name=self.name,
            talents=self.talents,
            weapons=list(self._weapons.keys()),
            armor=[a.name for a in self._armor],
            scrolls=[],
            ap=self.ap,
            hp=self.hp,
        )
