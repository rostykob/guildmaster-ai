"""Tests for GuildBuilder."""

from __future__ import annotations

import pytest

from guildmaster_ai.adventurers.base_guard import BaseGuard
from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.core.messages import GuardVerdict
from guildmaster_ai.sdk.builder import GuildBuilder

from .conftest import MockChatModel


class _CustomGuard(BaseGuard):
    @property
    def name(self) -> str:
        return "custom_guard"

    async def evaluate(self, content, criteria=None, context=None):
        return GuardVerdict(sender=self.name, verdict="pass", reason="ok")


class TestResolveApiKey:
    def test_explicit_key_takes_precedence(self) -> None:
        builder = GuildBuilder()
        key = builder._resolve_api_key("openrouter", "my-key")
        assert key == "my-key"

    def test_fallback_to_settings(self) -> None:
        builder = GuildBuilder()
        # When no explicit key, falls back to settings (which reads from env).
        # Just verify it returns a string and doesn't raise.
        key = builder._resolve_api_key("openrouter", None)
        assert isinstance(key, str)

    def test_unknown_provider_returns_empty(self) -> None:
        builder = GuildBuilder()
        key = builder._resolve_api_key("unknown_provider", None)
        assert key == ""


class TestWithLLMProvider:
    def test_instance_passthrough(self) -> None:
        mock_llm = MockChatModel()
        builder = GuildBuilder().with_llm_provider(mock_llm)
        assert builder._llm is mock_llm

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected str or GuildLLM"):
            GuildBuilder().with_llm_provider(42)  # type: ignore[arg-type]


class TestBuild:
    def test_build_without_llm_raises(self) -> None:
        with pytest.raises(ValueError, match="LLM provider is required"):
            GuildBuilder().build()

    def test_build_basic_guild(self) -> None:
        mock_llm = MockChatModel()
        guild = GuildBuilder().with_llm_provider(mock_llm).build()
        assert guild is not None

    def test_register_adventurer_class(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )
        assert len(guild.roster) == 1

    def test_register_adventurer_instance(self) -> None:
        mock_llm = MockChatModel()
        adv = GeneralAdventurer(name="Custom")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(adv)
            .build()
        )
        assert len(guild.roster) == 1
        assert guild.roster[0].name == "Custom"

    def test_register_adventurer_count(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer, count=3)
            .build()
        )
        assert len(guild.roster) == 3

    def test_with_guard_default(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .with_guard()
            .build()
        )
        assert guild.info.guard_enabled is True

    def test_with_guard_custom(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .with_guard(_CustomGuard())
            .build()
        )
        assert guild.info.guard_enabled is True

    def test_with_settings(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .with_settings(log_level="DEBUG")
            .build()
        )
        assert guild.settings.log_level == "DEBUG"

    def test_fluent_chaining(self) -> None:
        mock_llm = MockChatModel()
        builder = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .with_guard()
            .with_settings(log_level="WARNING")
        )
        assert isinstance(builder, GuildBuilder)
        guild = builder.build()
        assert len(guild.roster) == 1
        assert guild.info.guard_enabled is True
