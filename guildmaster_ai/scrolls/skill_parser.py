"""SKILL.md parser aligned with the Agent Skills specification."""

from __future__ import annotations

import re
from typing import Any

import yaml
from pydantic import BaseModel, Field

from guildmaster_ai.core.exceptions import ScrollValidationError

# Spec: 1-64 chars, lowercase alphanumeric + hyphens, no consecutive/leading/trailing hyphens.
SCROLL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){0,62}[a-z0-9]?$")


def validate_scroll_name(name: str) -> str:
    """Validate a scroll name against the Agent Skills spec.

    Raises :class:`ScrollValidationError` if the name is invalid.
    Returns the name unchanged on success.
    """
    if not name:
        raise ScrollValidationError("Scroll name must not be empty.")
    if len(name) > 64:
        raise ScrollValidationError(
            f"Scroll name must be at most 64 characters, got {len(name)}."
        )
    if not SCROLL_NAME_RE.match(name):
        raise ScrollValidationError(
            f"Invalid scroll name {name!r}. Must be lowercase alphanumeric + hyphens, "
            "no consecutive/leading/trailing hyphens."
        )
    return name


class ScrollFrontmatter(BaseModel):
    """Pydantic model for SKILL.md YAML frontmatter."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_skill_md(content: str) -> tuple[ScrollFrontmatter, str]:
    """Parse a SKILL.md file into frontmatter and markdown body.

    Returns a ``(ScrollFrontmatter, body)`` tuple.
    Raises :class:`ScrollValidationError` on parse or validation failure.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ScrollValidationError("SKILL.md must start with YAML frontmatter (--- delimiters).")

    yaml_text = match.group(1)
    body = content[match.end():]

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ScrollValidationError(f"Invalid YAML in SKILL.md frontmatter: {exc}") from exc

    if not isinstance(data, dict):
        raise ScrollValidationError("SKILL.md frontmatter must be a YAML mapping.")

    try:
        frontmatter = ScrollFrontmatter(**data)
    except Exception as exc:
        raise ScrollValidationError(f"Invalid SKILL.md frontmatter: {exc}") from exc

    validate_scroll_name(frontmatter.name)

    return frontmatter, body.strip()
