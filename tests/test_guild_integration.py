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
