"""Scroll — a SKILL.md-based agent skill package."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from guildmaster_ai.core.exceptions import ScrollValidationError
from guildmaster_ai.scrolls.skill_parser import ScrollFrontmatter, parse_skill_md


class Scroll:
    """A scroll loaded from a directory containing a ``SKILL.md`` file.

    Aligned with the `Agent Skills specification <https://agentskills.io/specification>`_.

    Directory layout::

        skill-name/
            SKILL.md        # required — YAML frontmatter + instructions
            scripts/        # optional — executable code
            references/     # optional — additional documentation
            assets/         # optional — templates, resources

    Parameters
    ----------
    path:
        Path to the skill directory (must contain ``SKILL.md``).
    """

    def __init__(self, path: Path) -> None:
        path = Path(path).resolve()
        skill_md = path / "SKILL.md"
        if not skill_md.is_file():
            raise ScrollValidationError(f"No SKILL.md found in {path}")

        content = skill_md.read_text(encoding="utf-8")
        self._frontmatter, self._instructions = parse_skill_md(content)

        # Validate directory name matches skill name
        if path.name != self._frontmatter.name:
            raise ScrollValidationError(
                f"Directory name {path.name!r} does not match "
                f"skill name {self._frontmatter.name!r}."
            )

        self._path = path

    # ── Spec properties ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Skill name (matches directory name)."""
        return self._frontmatter.name

    @property
    def description(self) -> str:
        """Short description — used for discovery."""
        return self._frontmatter.description

    @property
    def instructions(self) -> str:
        """Full instruction body from SKILL.md — loaded on activation."""
        return self._instructions

    @property
    def discovery_text(self) -> str:
        """Lightweight discovery representation (~100 tokens)."""
        return f"{self.name}: {self.description}"

    @property
    def license(self) -> str | None:
        """SPDX license identifier (optional)."""
        return self._frontmatter.license

    @property
    def compatibility(self) -> str | None:
        """Environment requirements (optional)."""
        return self._frontmatter.compatibility

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata (optional)."""
        return self._frontmatter.metadata

    @property
    def frontmatter(self) -> ScrollFrontmatter:
        """The parsed SKILL.md frontmatter."""
        return self._frontmatter

    # ── Resource access ──────────────────────────────────────────────

    @property
    def resource_dir(self) -> Path:
        """The skill directory path."""
        return self._path

    def has_resource(self, relative_path: str) -> bool:
        """Check whether a resource file exists within the skill directory."""
        target = (self._path / relative_path).resolve()
        if not str(target).startswith(str(self._path)):
            return False
        return target.is_file()

    async def read_resource(self, relative_path: str) -> str:
        """Read a resource file from the skill directory.

        Raises :class:`ScrollValidationError` on path traversal attempts
        or missing files.
        """
        target = (self._path / relative_path).resolve()
        if not str(target).startswith(str(self._path)):
            raise ScrollValidationError(
                f"Path traversal detected: {relative_path!r}"
            )
        if not target.is_file():
            raise ScrollValidationError(
                f"Resource not found: {relative_path!r}"
            )
        return await asyncio.to_thread(target.read_text, "utf-8")

    def __repr__(self) -> str:
        return f"Scroll(name={self.name!r}, path={self._path})"
