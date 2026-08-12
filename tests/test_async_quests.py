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


class TestQuestCompletionHandlers:
    async def test_async_handler_receives_quest_and_result(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        seen: list[tuple[str, bool]] = []

        async def handler(quest, result) -> None:  # type: ignore[no-untyped-def]
            seen.append((quest.id, result.success))

        guild.on_quest_complete(handler)
        result = await guild.run_quest("Do something")
        assert seen == [(result.quest_id, result.success)]
        await guild.close()

    async def test_sync_handler_and_unsubscribe(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))
        seen: list[str] = []
        unsubscribe = guild.on_quest_complete(lambda q, r: seen.append(q.id))

        first = await guild.run_quest("First quest")
        unsubscribe()
        unsubscribe()  # idempotent
        await guild.run_quest("Second quest")

        assert seen == [first.quest_id]
        await guild.close()

    async def test_failing_handler_does_not_break_quest(self) -> None:
        guild = _guild(MockChatModel(response_content="Done."))

        def bad_handler(quest, result) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("subscriber exploded")

        guild.on_quest_complete(bad_handler)
        result = await guild.run_quest("Do something")
        assert result.summary != ""
        await guild.close()

    async def test_handler_fires_for_failed_quests_too(self) -> None:
        from guildmaster_ai.sdk.guild import Guild

        # No adventurers registered → quest resolves as infeasible/failed,
        # but subscribers are still notified.
        guild = Guild(llm=MockChatModel())
        seen: list[bool] = []
        guild.on_quest_complete(lambda q, r: seen.append(r.success))

        result = await guild.run_quest("Anything at all")
        assert result.success is False
        assert seen == [False]
        await guild.close()

    async def test_builder_with_quest_listener(self) -> None:
        seen: list[str] = []
        guild = (
            GuildBuilder()
            .with_llm_provider(MockChatModel(response_content="Done."))
            .register_adventurer(GeneralAdventurer)
            .with_quest_listener(lambda q, r: seen.append(q.id))
            .build()
        )
        result = await guild.run_quest("Go")
        assert seen == [result.quest_id]
        await guild.close()
