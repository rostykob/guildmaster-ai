"""Scroll catalog — name-based lookup from a directory of skill packages."""

from __future__ import annotations

import logging
from pathlib import Path

from guildmaster_ai.core.exceptions import ScrollValidationError
from guildmaster_ai.scrolls.scroll import Scroll

logger = logging.getLogger("guildmaster.scrolls")


class ScrollCatalog:
    """Registry that discovers and serves scrolls from a directory.

    The catalog scans a root directory for subdirectories containing
    ``SKILL.md`` files.  Scrolls are loaded lazily on first access
    and cached thereafter.

    Parameters
    ----------
    root:
        Path to the scrolls directory (e.g. ``./scrolls``).
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._cache: dict[str, Scroll] = {}
        self._scanned = False

    @property
    def root(self) -> Path:
        """The root directory this catalog scans."""
        return self._root

    def _scan(self) -> None:
        """Scan the root directory and populate the name index."""
        if self._scanned:
            return
        if not self._root.is_dir():
            logger.warning("Scrolls directory does not exist: %s", self._root)
            self._scanned = True
            return
        for child in sorted(self._root.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                try:
                    scroll = Scroll(child)
                    self._cache[scroll.name] = scroll
                    logger.debug("Cataloged scroll: %s", scroll.name)
                except ScrollValidationError as exc:
                    logger.warning("Skipping invalid scroll %s: %s", child.name, exc)
        self._scanned = True

    def get(self, name: str) -> Scroll:
        """Look up a scroll by name.

        Raises :class:`KeyError` if the scroll is not found.
        """
        self._scan()
        if name not in self._cache:
            raise KeyError(
                f"Scroll {name!r} not found in {self._root}. "
                f"Available: {list(self._cache.keys())}"
            )
        return self._cache[name]

    def list_names(self) -> list[str]:
        """Return the names of all available scrolls."""
        self._scan()
        return list(self._cache.keys())

    def list_scrolls(self) -> list[Scroll]:
        """Return all available scrolls."""
        self._scan()
        return list(self._cache.values())

    def __contains__(self, name: str) -> bool:
        self._scan()
        return name in self._cache

    def __len__(self) -> int:
        self._scan()
        return len(self._cache)

    def __repr__(self) -> str:
        return f"ScrollCatalog(root={self._root!r}, scrolls={self.list_names()})"
