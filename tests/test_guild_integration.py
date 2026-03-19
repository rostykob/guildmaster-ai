"""Integration test for the Guild SDK."""

from __future__ import annotations

import json

import pytest

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.core.quest import QuestStatus
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

        guild_with_guard = GuildBuilder().with_llm_provider(mock_llm).with_guard().build()
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

    # ── Complex quest decomposition ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_complex_quest_decomposition_lifecycle(self) -> None:
        """A complex quest gets decomposed, subtasks run, and results combine."""
        decompose_response = json.dumps(
            {
                "decompose": True,
                "strategy": "Split into parts",
                "subtasks": [
                    {
                        "title": "Part 1",
                        "description": "Do part 1",
                        "required_talents": ["general"],
                        "acceptance_criteria": ["Part 1 done"],
                    },
                    {
                        "title": "Part 2",
                        "description": "Do part 2",
                        "required_talents": ["general"],
                        "acceptance_criteria": ["Part 2 done"],
                    },
                ],
            }
        )
        # LLM calls: 1-2) refine_all_talents (2 adventurers),
        # 3) receptionist intake, 4) assess_quest_talents,
        # 5) plan_quest, 6) subtask 1 exec, 7) subtask 2 exec,
        # 8) evaluate completion, 9) verify result
        mock_llm = MockChatModel(
            responses=[
                '["general"]',  # refine talents for adventurer 1
                '["general"]',  # refine talents for adventurer 2
                '{"title": "Complex quest", "description": "Do complex things", '
                '"acceptance_criteria": ["Done"]}',
                '["general"]',
                decompose_response,
                "Part 1 result",
                "Part 2 result",
                '{"decision": "done", "reason": "All done", "combined_summary": "Both parts done"}',
                '{"accepted": true}',
            ]
        )
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer, count=2)
            .build()
        )
        result = await guild.post_quest("Do something complex")
        assert result.success is True
        assert result.data.get("subtask_count") == 2

        # Parent quest should be archived
        parent = guild.get_quest(result.quest_id)
        assert parent.is_composite is True
        assert parent.status == QuestStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_complex_quest_with_retry(self) -> None:
        """A complex quest retries failed subtasks."""
        decompose_response = json.dumps(
            {
                "decompose": True,
                "strategy": "Two parts",
                "subtasks": [
                    {"title": "Good part", "description": "Works"},
                    {"title": "Bad part", "description": "Fails then works"},
                ],
            }
        )
        mock_llm = MockChatModel(
            responses=[
                # refine talents (1 adventurer)
                '["general"]',
                # receptionist
                '{"title": "Retry quest", "description": "Test retry", '
                '"acceptance_criteria": ["Done"]}',
                # assess_quest_talents
                '["general"]',
                # plan_quest
                decompose_response,
                # subtask 1 exec - success
                "Good result",
                # subtask 2 exec - failure
                "Bad result",
                # evaluate completion - retry
                '{"decision": "retry", "reason": "Sub 2 failed", "retry_subtask_indices": [1]}',
                # subtask 2 retry - success
                "Fixed result",
                # evaluate completion - done
                '{"decision": "done", "reason": "All done", "combined_summary": "Complete"}',
                # verify result
                '{"accepted": true}',
            ]
        )
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )

        # Need to make the second subtask fail initially
        # The mock returns responses in order, so "Bad result" for subtask 2
        # will be treated as success by the default execute() (no tool calls = success)
        # We need to handle this differently - the evaluate step handles the logic
        result = await guild.post_quest("Do something with retry")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_simple_quest_unchanged(self) -> None:
        """A simple quest follows the original path when plan_quest returns None."""
        mock_llm = MockChatModel(
            responses=[
                # refine talents (1 adventurer)
                '["general"]',
                # receptionist
                '{"title": "Simple quest", "description": "Just do it", '
                '"acceptance_criteria": ["Done"]}',
                # assess_quest_talents
                '["general"]',
                # plan_quest - no decomposition
                '{"decompose": false}',
                # adventurer execute
                "Simple result",
                # verify result
                '{"accepted": true}',
            ]
        )
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )
        result = await guild.post_quest("Do something simple")
        assert result.success is True
        quest = guild.get_quest(result.quest_id)
        assert quest.is_composite is False
