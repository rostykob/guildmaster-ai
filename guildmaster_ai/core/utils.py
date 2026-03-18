"""Shared helpers used across the core and adventurer packages."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(UTC)


# Pattern: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n(.*?)```\s*$", re.DOTALL)


def parse_llm_json(raw: str) -> Any:
    """Parse a JSON value from LLM output, stripping markdown fences.

    Returns the parsed Python object on success; raises ``ValueError``
    when the content cannot be decoded.
    """
    text = raw.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)
