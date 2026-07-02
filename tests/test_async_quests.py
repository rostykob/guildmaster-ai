"""Tests for background quest submission, status checks, and waiting."""

from __future__ import annotations

import pytest

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.core.quest import QuestStatus
from guildmaster_ai.sdk.builder import GuildBuilder

from .conftest import MockChatModel


def _guild(mock_llm: MockChatModel):  # type: ignore[no-untyped-def]
    return (
        GuildBuilder()
        .with_llm_provider(mock_llm)
        .register_adventurer(GeneralAdventurer)
        .build()
    )


class TestQuestSubmission:
    async def test_post_quest_returns_ticket_immediately(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        ticket = await guild.post_quest("Tell me about Python")
        assert ticket.status == "submitted"
        assert ticket.quest_id
        assert ticket.title
        # The quest is registered before any execution happens.
        quest = guild.get_quest(ticket.quest_id)
        assert quest.id == ticket.quest_id
        await guild.close()

    async def test_wait_for_quest_returns_result(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        ticket = await guild.post_quest("Do something")
        result = await guild.wait_for_quest(ticket.quest_id)
        assert result.quest_id == ticket.quest_id
        assert result.summary != ""
        await guild.close()

    async def test_wait_for_unknown_quest_raises(self) -> None:
        guild = _guild(MockChatModel())
        with pytest.raises(KeyError, match="not found"):
            await guild.wait_for_quest("nonexistent")
        await guild.close()

    async def test_run_quest_is_post_plus_wait(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        result = await guild.run_quest("Quick task")
        assert result.quest_id != ""
        assert guild.get_quest(result.quest_id).status == QuestStatus.ARCHIVED
        await guild.close()

    async def test_close_finishes_pending_quests(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        ticket = await guild.post_quest("Pending at close")
        await guild.close()
        # Worker drains the stream before shutdown — result is recorded.
        assert guild.get_result(ticket.quest_id) is not None


class TestReceptionistStatus:
    async def test_check_status_after_completion(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        ticket = await guild.post_quest("Status test")
        await guild.wait_for_quest(ticket.quest_id)

        report = guild.receptionist.check_status(ticket.quest_id)
        assert report.quest_id == ticket.quest_id
        assert report.status == QuestStatus.ARCHIVED.value
        assert report.finished is True
        assert report.success is True
        assert report.summary
        await guild.close()

    async def test_check_status_pending_quest(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        ticket = await guild.post_quest("Pending status")
        # Immediately after submission the quest exists but has no result yet
        # (the worker may not have started it).
        report = guild.check_quest_status(ticket.quest_id)
        assert report.quest_id == ticket.quest_id
        if not report.finished:
            assert report.success is None
        await guild.close()

    async def test_check_status_unknown_quest_raises(self) -> None:
        guild = _guild(MockChatModel())
        with pytest.raises(KeyError, match="not found"):
            guild.check_quest_status("nonexistent")
        await guild.close()

    async def test_unbound_receptionist_raises(self) -> None:
        from guildmaster_ai.adventurers.receptionist import Receptionist

        receptionist = Receptionist()
        with pytest.raises(RuntimeError, match="not connected"):
            receptionist.check_status("any-id")


class TestConcurrentQuests:
    async def test_multiple_quests_complete(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        tickets = [await guild.post_quest(f"Quest {i}") for i in range(3)]
        results = [await guild.wait_for_quest(t.quest_id) for t in tickets]
        assert len(results) == 3
        assert all(r.summary for r in results)
        assert len(guild.quests) == 3
        await guild.close()
