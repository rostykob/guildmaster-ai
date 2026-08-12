"""Shared helpers used across the core and adventurer packages."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("guildmaster.utils")


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(UTC)


def charter_block(charter: str) -> str:
    """Format the guild charter for prompt injection (empty when absent).

    The charter is owner-provided domain context. It is always appended to the
    built-in system prompts — never substituted for them — so the JSON output
    contracts those prompts define stay intact.
    """
    if not charter:
        return ""
    return f"\n\nGuild charter (domain context from the guild owner):\n{charter}"


# Pattern: ```json ... ``` or ``` ... ``` (full-string match)
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n(.*?)```\s*$", re.DOTALL)
# Same pattern but matches anywhere in the string (prose before/after)
_FENCE_SEARCH_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _find_outermost_json_object(text: str) -> str | None:
    """Find the first balanced ``{...}`` JSON object in *text*."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_json(raw: str) -> Any:
    """Parse a JSON value from LLM output, stripping markdown fences.

    Tries four strategies in order:
    1. Exact fenced match (entire string is a fenced code block)
    2. Search for a fenced block anywhere in the string
    3. Find the first balanced ``{...}`` JSON object
    4. Fall back to ``json.loads`` on the raw text

    Returns the parsed Python object on success; raises ``ValueError``
    when the content cannot be decoded.
    """
    text = raw.strip()

    # Strategy 1: exact fenced match
    m = _FENCE_RE.match(text)
    if m:
        return json.loads(m.group(1).strip())

    # Strategy 2: fenced block anywhere in prose
    m = _FENCE_SEARCH_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (ValueError, json.JSONDecodeError):
            pass

    # Strategy 3: first balanced {…} object
    obj_str = _find_outermost_json_object(text)
    if obj_str is not None:
        try:
            return json.loads(obj_str)
        except (ValueError, json.JSONDecodeError):
            pass

    # Strategy 4: raw text as-is
    return json.loads(text)


def safe_parse_llm_json(raw: str, *, context: str = "") -> dict[str, Any] | None:
    """Parse JSON from LLM output, returning ``None`` on failure with a warning.

    Wraps :func:`parse_llm_json` with error handling so callers don't need
    their own try/except blocks.  Logs a WARNING with the context label and
    a truncated snippet of the raw response on failure.
    """
    try:
        data = parse_llm_json(raw)
        if isinstance(data, dict):
            return data
        # LLM returned valid JSON but not a dict (e.g. a list) — caller
        # expected a dict so treat as parse failure.
        logger.warning(
            "LLM returned non-dict JSON%s: %s",
            f" ({context})" if context else "",
            repr(data)[:200],
        )
        return None
    except (ValueError, KeyError):
        logger.warning(
            "Failed to parse LLM JSON%s: %.200s",
            f" ({context})" if context else "",
            raw,
        )
        return None
