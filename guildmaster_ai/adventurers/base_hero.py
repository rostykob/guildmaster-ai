"""BaseHero — a deep agent adventurer with scroll (skill) and member support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.messages import AdventurerProfile
from guildmaster_ai.llm.types import GuildLLM
from guildmaster_ai.scrolls.catalog import ScrollCatalog
from guildmaster_ai.scrolls.scroll import Scroll
from guildmaster_ai.weapons.base_weapon import BaseWeapon


class BaseHero(BaseAdventurer):
    """A deep agent adventurer with scroll, member, and subagent support.

    Extends :class:`BaseAdventurer` with three additional capabilities:

    - **Scrolls** — SKILL.md-based skills loaded from a :class:`ScrollCatalog`.
      Scroll directories are passed as native skills to the deep agent, which
      handles progressive disclosure and file operations internally.
    - **Members** — other :class:`BaseAdventurer` instances recruited as party
      members via :meth:`recruit`.  Each member is compiled into a
      ``CompiledSubAgent`` so the deep agent can delegate work via its
      ``task`` tool.
    - **Deep agent backend** — uses ``create_deep_agent`` from ``deepagents``
      instead of ``create_agent``, providing subagent orchestration and a
      filesystem backend.

    Use this class (or its concrete subclass :class:`GeneralHero`) when an
    adventurer needs to coordinate other agents or use structured skill
    instructions.  For simple tool-only agents, use :class:`BaseAdventurer`.
    """

    DEFAULT_AP: int = 60
    """Heroes coordinate members and skills — they get a larger tool budget."""

    DEFAULT_HP: int = 5

    def __init__(
        self,
        adventurer_id: str | None = None,
        name: str = "",
        llm: GuildLLM | None = None,
        scroll_catalog: ScrollCatalog | None = None,
        ap: int | None = None,
        hp: int | None = None,
    ) -> None:
        super().__init__(adventurer_id=adventurer_id, name=name, llm=llm, ap=ap, hp=hp)
        self._scroll_catalog = scroll_catalog
        self._scrolls: dict[str, Scroll] = {}
        self._pending_scroll_names: list[str] = []
        self._members: dict[str, BaseAdventurer] = {}

    # ── Scroll catalog ────────────────────────────────────────────────

    @property
    def scroll_catalog(self) -> ScrollCatalog | None:
        """Return the configured scroll catalog, or ``None``."""
        return self._scroll_catalog

    @scroll_catalog.setter
    def scroll_catalog(self, value: ScrollCatalog | None) -> None:
        self._scroll_catalog = value
        if value is not None:
            self._resolve_pending_scrolls()

    # ── Scrolls ───────────────────────────────────────────────────────

    @property
    def scrolls(self) -> dict[str, Scroll]:
        """Return the picked scrolls keyed by name."""
        return dict(self._scrolls)

    @property
    def scroll_names(self) -> list[str]:
        """Return a list of picked scroll names."""
        return list(self._scrolls.keys())

    def _resolve_scroll(self, name: str) -> Scroll:
        """Look up a scroll by name from the catalog."""
        if self._scroll_catalog is None:
            raise ValueError(
                f"Cannot resolve scroll {name!r} — "
                "no scroll catalog configured. Pass a ScrollCatalog to the "
                "BaseHero or use GuildBuilder.with_scrolls_dir()."
            )
        return self._scroll_catalog.get(name)

    def pick_scroll(self, name_or_path: str | Path) -> None:
        """Pick up a scroll by name or by direct path to a skill directory.

        Passing a path to a directory containing ``SKILL.md`` loads the
        scroll immediately — no catalog needed. Passing a name resolves it
        from the scroll catalog; if no catalog is set yet (e.g. before
        ``GuildBuilder.build()`` injects one), the name is stored as pending
        and resolved when the catalog arrives. Either way the scroll only
        needs to be registered once.
        """
        path = Path(name_or_path)
        if isinstance(name_or_path, Path) or (path / "SKILL.md").is_file():
            scroll = Scroll(path)
            self._scrolls[scroll.name] = scroll
            self._logger.info("Picked scroll from path: %s", scroll.name)
            return
        name = str(name_or_path)
        if self._scroll_catalog is not None:
            scroll = self._resolve_scroll(name)
            self._scrolls[scroll.name] = scroll
            self._logger.info("Picked scroll: %s", scroll.name)
        else:
            self._pending_scroll_names.append(name)
            self._logger.info("Deferred scroll pick: %s (no catalog yet)", name)

    def _resolve_pending_scrolls(self) -> None:
        """Resolve any scroll names that were picked before the catalog was set."""
        for name in self._pending_scroll_names:
            scroll = self._resolve_scroll(name)
            self._scrolls[scroll.name] = scroll
            self._logger.info("Resolved deferred scroll: %s", scroll.name)
        self._pending_scroll_names.clear()

    # ── Members (sub-adventurers as subagents) ────────────────────────

    def recruit(self, adventurer: BaseAdventurer) -> None:
        """Add an adventurer to this hero's party as a subagent.

        Recruited members are compiled into ``CompiledSubAgent`` instances at
        agent build time.  The deep agent can then delegate work to them via
        its ``task`` tool.  Call :meth:`dismiss` to remove a member.
        """
        self._members[adventurer.id] = adventurer
        self._logger.info("Recruited member: %s", adventurer.label)

    def dismiss(self, adventurer_id: str) -> None:
        """Remove an adventurer from this hero's party.

        No-op if the adventurer is not a current member.
        """
        removed = self._members.pop(adventurer_id, None)
        if removed:
            self._logger.info("Dismissed member: %s", removed.label)

    @property
    def members(self) -> dict[str, BaseAdventurer]:
        """Return recruited members keyed by adventurer id."""
        return dict(self._members)

    @property
    def member_names(self) -> list[str]:
        """Return a list of recruited member names."""
        return [m.label for m in self._members.values()]

    # ── Spawning ──────────────────────────────────────────────────────

    def spawn(self) -> BaseHero:
        """Create a fresh instance with the same configuration."""
        clone: BaseHero = super().spawn()  # type: ignore[assignment]
        clone._scroll_catalog = self._scroll_catalog
        clone._scrolls = dict(self._scrolls)
        clone._pending_scroll_names = list(self._pending_scroll_names)
        clone._members = dict(self._members)
        return clone

    # ── Execute (deep agent with native skills) ───────────────────────

    def _build_agent(self) -> Any:
        """Build a deep agent with weapons, armor, scrolls, and member subagents.

        Pipeline:
        1. Collect weapons as LangChain-compatible tools.
        2. Resolve scroll directories → skill paths for native skill injection.
        3. Compile each recruited member into a ``CompiledSubAgent`` by building
           their own agent graph (recursive — members can themselves be heroes).
        4. Pass everything to ``create_deep_agent`` with a filesystem backend
           for sandbox-safe file operations.
        """
        if self._llm is None:
            raise RuntimeError("No LLM configured for this adventurer.")

        if self._pending_scroll_names:
            raise RuntimeError(
                f"Unresolved scrolls: {self._pending_scroll_names}. "
                "Set a scroll_catalog before building the agent."
            )

        tools: list[BaseWeapon] = list(self._weapons.values())

        # Collect scroll directories as skill paths for the deep agent
        skill_paths: list[str] = [
            str(scroll.resource_dir) for scroll in self._scrolls.values()
        ]

        # Convert recruited members to CompiledSubAgent instances.
        # Each member's agent graph is built here so the deep agent can
        # invoke them as subagents via the ``task`` tool.
        subagents: list[SubAgent | CompiledSubAgent] = [
            CompiledSubAgent(
                name=member.label,
                description=member.system_prompt[:200],
                runnable=member._build_agent(),
            )
            for member in self._members.values()
        ]

        return create_deep_agent(
            model=self._llm,
            tools=tools or None,
            system_prompt=self.system_prompt,
            skills=skill_paths or None,
            subagents=subagents or None,
            middleware=[*self._armor, self._vitality()],
            backend=FilesystemBackend(),
            name=self.label,
        )

    # ── Profile ───────────────────────────────────────────────────────

    def profile(self) -> AdventurerProfile:
        """Return the public profile including scrolls and members."""
        return AdventurerProfile(
            id=self.id,
            name=self.name,
            talents=self.talents,
            weapons=list(self._weapons.keys()),
            armor=[a.name for a in self._armor],
            scrolls=list(self._scrolls.keys()),
            members=self.member_names,
        )
