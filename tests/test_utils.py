"""Tests for guildmaster_ai.core.utils — JSON parsing strategies."""

from __future__ import annotations

import pytest

from guildmaster_ai.core.utils import parse_llm_json


class TestParseLLMJson:
    def test_exact_fenced_json(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        assert parse_llm_json(raw) == {"key": "value"}

    def test_fenced_json_in_prose(self) -> None:
        raw = 'Here is the answer:\n```json\n{"key": "value"}\n```\nHope that helps!'
        assert parse_llm_json(raw) == {"key": "value"}

    def test_bare_json_in_prose(self) -> None:
        raw = 'Sure, here you go: {"key": "value"} — let me know if you need more.'
        assert parse_llm_json(raw) == {"key": "value"}

    def test_plain_json(self) -> None:
        raw = '{"a": 1}'
        assert parse_llm_json(raw) == {"a": 1}

    def test_json_array(self) -> None:
        raw = '["a", "b"]'
        assert parse_llm_json(raw) == ["a", "b"]

    def test_nested_json_in_prose(self) -> None:
        raw = 'Result: {"outer": {"inner": true}} done.'
        result = parse_llm_json(raw)
        assert result == {"outer": {"inner": True}}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_llm_json("This is not JSON at all")

    def test_fenced_no_lang_tag(self) -> None:
        raw = '```\n{"x": 1}\n```'
        assert parse_llm_json(raw) == {"x": 1}
