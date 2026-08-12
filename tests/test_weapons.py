"""Tests for Weapon abstractions."""

from __future__ import annotations

import pytest

from guildmaster_ai.weapons.file_read import FileReadWeapon
from guildmaster_ai.weapons.script_run import ScriptRunWeapon
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


class TestScriptRunWeapon:
    @pytest.mark.asyncio
    async def test_run_python_script(self, tmp_path) -> None:
        script = tmp_path / "hello.py"
        script.write_text("print('hello from script')")
        weapon = ScriptRunWeapon()
        result = await weapon.execute(path=str(script))
        assert result["exit_code"] == 0
        assert result["stdout"] == "hello from script"

    @pytest.mark.asyncio
    async def test_run_with_args(self, tmp_path) -> None:
        script = tmp_path / "echo.py"
        script.write_text("import sys; print(' '.join(sys.argv[1:]))")
        weapon = ScriptRunWeapon()
        result = await weapon.execute(path=str(script), args="foo bar")
        assert result["exit_code"] == 0
        assert result["stdout"] == "foo bar"

    @pytest.mark.asyncio
    async def test_run_nonexistent(self) -> None:
        weapon = ScriptRunWeapon()
        result = await weapon.execute(path="/nonexistent/script.py")
        assert result["exit_code"] != 0
        assert "No such file" in result.get("stderr", "")

    @pytest.mark.asyncio
    async def test_run_script_with_error(self, tmp_path) -> None:
        script = tmp_path / "fail.py"
        script.write_text("import sys; print('oops', file=sys.stderr); sys.exit(1)")
        weapon = ScriptRunWeapon()
        result = await weapon.execute(path=str(script))
        assert result["exit_code"] == 1
        assert "oops" in result["stderr"]

    def test_tool_spec(self) -> None:
        weapon = ScriptRunWeapon()
        spec = weapon.to_tool_spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "run_script"


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
