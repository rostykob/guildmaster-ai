"""Integration test for the Guild SDK."""

from __future__ import annotations

import pytest

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.sdk.builder import GuildBuilder
from guildmaster_ai.sdk.guild import Guild

from .conftest import MockChatModel


class TestGuildIntegration:
    @pytest.mark.asyncio
    async def test_full_quest_lifecycle(self) -> None:
        mock_llm = MockChatModel(response_content="Here is the result of the quest.")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )

        result = await guild.post_quest("Tell me about Python")
        assert result.quest_id != ""
        assert result.summary != ""

    @pytest.mark.asyncio
    async def test_no_adventurers_fails(self) -> None:
        mock_llm = MockChatModel()
        guild = Guild(llm=mock_llm)
        result = await guild.post_quest("Do something impossible")
        assert result.success is False

    def test_builder_requires_provider(self) -> None:
        with pytest.raises(ValueError, match="LLM provider is required"):
            GuildBuilder().build()

    def test_builder_fluent_api(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .register_adventurer(GeneralAdventurer, count=2)
            .with_guard()
            .build()
        )
        assert len(guild.roster) == 3

    # ── Guild.info ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_info_counts(self) -> None:
        mock_llm = MockChatModel(response_content="Done.")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )

        info_before = guild.info
        assert info_before.total_quests == 0
        assert info_before.completed == 0
        assert len(info_before.adventurers) == 1

        await guild.post_quest("Test quest")

        info_after = guild.info
        assert info_after.total_quests == 1
        # Quest should be archived after lifecycle
        assert info_after.archived == 1

    def test_info_guard_enabled_flag(self) -> None:
        mock_llm = MockChatModel()
        guild_no_guard = GuildBuilder().with_llm_provider(mock_llm).build()
        assert guild_no_guard.info.guard_enabled is False

        guild_with_guard = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .with_guard()
            .build()
        )
        assert guild_with_guard.info.guard_enabled is True

    # ── Quest lookup ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_quest_lookup_by_id(self) -> None:
        mock_llm = MockChatModel(response_content="Result.")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )
        result = await guild.post_quest("Lookup test")
        quest = guild.get_quest(result.quest_id)
        assert quest.title  # non-empty
        assert quest.id == result.quest_id

    @pytest.mark.asyncio
    async def test_quest_result_lookup(self) -> None:
        mock_llm = MockChatModel(response_content="Result.")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )
        result = await guild.post_quest("Result test")
        stored = guild.get_result(result.quest_id)
        assert stored is not None
        assert stored.quest_id == result.quest_id

    def test_quest_lookup_missing_raises(self) -> None:
        mock_llm = MockChatModel()
        guild = Guild(llm=mock_llm)
        with pytest.raises(KeyError, match="not found"):
            guild.get_quest("nonexistent-id")

    def test_get_result_missing_returns_none(self) -> None:
        mock_llm = MockChatModel()
        guild = Guild(llm=mock_llm)
        assert guild.get_result("nonexistent") is None

    @pytest.mark.asyncio
    async def test_quests_list(self) -> None:
        mock_llm = MockChatModel(response_content="Done.")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )
        await guild.post_quest("Quest A")
        await guild.post_quest("Quest B")
        assert len(guild.quests) == 2
