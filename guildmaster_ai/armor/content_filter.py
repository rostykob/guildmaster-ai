from __future__ import annotations

from guildmaster_ai.armor.base_armor import ArmorResult, BaseArmor


class ContentFilterArmor(BaseArmor):
    """Simple content filter that blocks messages containing forbidden patterns."""

    def __init__(self, blocked_patterns: list[str]) -> None:
        self._blocked_patterns = blocked_patterns

    @property
    def name(self) -> str:
        return "content_filter"

    async def pre_process(self, content: str) -> ArmorResult:
        return self._check(content)

    async def post_process(self, content: str) -> ArmorResult:
        return self._check(content)

    def _check(self, content: str) -> ArmorResult:
        lowered = content.lower()
        for pattern in self._blocked_patterns:
            if pattern.lower() in lowered:
                return ArmorResult(
                    verdict="block",
                    message=f"Content blocked: matched pattern '{pattern}'",
                )
        return ArmorResult(verdict="pass")
