"""Tests for Weapon abstractions."""

from __future__ import annotations

import pytest

from guildmaster_ai.weapons.file_read import FileReadWeapon
from guildmaster_ai.weapons.web_search import WebSearchWeapon


class TestWebSearchWeapon:
    @pytest.mark.asyncio
    async def test_execute_placeholder(self) -> None:
        weapon = WebSearchWeapon()
        result = await weapon.execute(query="test query")
        assert result["query"] == "test query"
        assert "results" in result

    def test_tool_spec(self) -> None:
        weapon = WebSearchWeapon()
        spec = weapon.to_tool_spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "web_search"


class TestFileReadWeapon:
    @pytest.mark.asyncio
    async def test_read_nonexistent(self) -> None:
        weapon = FileReadWeapon()
        result = await weapon.execute(path="/nonexistent/file.txt")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_read_existing(self, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        weapon = FileReadWeapon()
        result = await weapon.execute(path=str(test_file))
        assert result["content"] == "hello world"
