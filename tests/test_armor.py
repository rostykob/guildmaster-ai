"""Tests for Armor abstractions."""

from __future__ import annotations

import pytest

from guildmaster_ai.armor.content_filter import ContentFilterArmor
from guildmaster_ai.armor.rate_limiter import RateLimiterArmor


class TestContentFilter:
    @pytest.mark.asyncio
    async def test_pass_clean_content(self) -> None:
        armor = ContentFilterArmor(blocked_patterns=["bad_word"])
        result = await armor.pre_process("This is clean content")
        assert result.verdict == "pass"

    @pytest.mark.asyncio
    async def test_block_bad_content(self) -> None:
        armor = ContentFilterArmor(blocked_patterns=["bad_word"])
        result = await armor.pre_process("This contains bad_word in it")
        assert result.verdict == "block"

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        armor = ContentFilterArmor(blocked_patterns=["secret"])
        result = await armor.post_process("This has SECRET data")
        assert result.verdict == "block"


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_within_limit(self) -> None:
        armor = RateLimiterArmor(max_calls=5, window_seconds=60.0)
        result = await armor.pre_process("request 1")
        assert result.verdict == "pass"

    @pytest.mark.asyncio
    async def test_exceed_limit(self) -> None:
        armor = RateLimiterArmor(max_calls=2, window_seconds=60.0)
        await armor.pre_process("request 1")
        await armor.pre_process("request 2")
        result = await armor.pre_process("request 3")
        assert result.verdict == "block"
