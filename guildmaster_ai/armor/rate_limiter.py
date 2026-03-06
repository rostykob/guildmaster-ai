from __future__ import annotations

import time

from guildmaster_ai.armor.base_armor import ArmorResult, BaseArmor


class RateLimiterArmor(BaseArmor):
    """Simple sliding-window rate limiter."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []

    @property
    def name(self) -> str:
        return "rate_limiter"

    async def pre_process(self, content: str) -> ArmorResult:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self._max_calls:
            return ArmorResult(
                verdict="block",
                message=(
                    f"Rate limit exceeded: {self._max_calls} calls "
                    f"per {self._window_seconds}s window"
                ),
            )

        self._timestamps.append(now)
        return ArmorResult(verdict="pass")
