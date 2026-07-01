"""Tests for persistent memory: conversations, findings, and stage history."""

from __future__ import annotations

from pathlib import Path

import pytest

from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.core.quest import Quest, QuestRank, QuestStatus
from guildmaster_ai.memory.sqlite_store import SQLiteStore
from guildmaster_ai.sdk.builder import GuildBuilder

from .conftest import MockChatModel


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(db_path=str(tmp_path / "test.db"))
    await s.initialize()
    return s


def _quest() -> Quest:
    return Quest(
        title="Scrape a site",
        description="Build a scraper",
        required_talents=["general"],
        rank=QuestRank.C,
        acceptance_criteria=["Works"],
    )


class TestConversationStorage:
    async def test_save_and_get_conversation(self, store: SQLiteStore) -> None:
        quest = _quest()
        await store.save_quest(quest)
        await store.save_conversation(
            quest.id,
            "adventurer-1",
            [
                {"role": "human", "content": "Build a scraper"},
                {"role": "ai", "content": "Done, here it is."},
            ],
        )

        convo = await store.get_conversation(quest.id)
        assert [m["role"] for m in convo] == ["human", "ai"]
        assert convo[0]["content"] == "Build a scraper"
        assert convo[0]["adventurer"] == "adventurer-1"
        assert convo[0]["seq"] == 0

    async def test_empty_transcript_is_noop(self, store: SQLiteStore) -> None:
        quest = _quest()
        await store.save_quest(quest)
        await store.save_conversation(quest.id, "adventurer-1", [])
        assert await store.get_conversation(quest.id) == []

    async def test_conversations_are_append_only(self, store: SQLiteStore) -> None:
        quest = _quest()
        await store.save_quest(quest)
        await store.save_conversation(
            quest.id, "adv", [{"role": "ai", "content": "first attempt"}]
        )
        await store.save_conversation(
            quest.id, "adv", [{"role": "ai", "content": "retry attempt"}]
        )
        convo = await store.get_conversation(quest.id)
        assert [m["content"] for m in convo] == ["first attempt", "retry attempt"]


class TestFindingStorage:
    async def test_save_and_get_findings_ordered(self, store: SQLiteStore) -> None:
        quest = _quest()
        await store.save_quest(quest)
        await store.save_finding(quest.id, 0, "subtask_result", "Part 1 ok", {"success": True})
        await store.save_finding(
            quest.id, 0, "leader_decision", "Retry part 2", {"decision": "retry"}
        )
        await store.save_finding(quest.id, 1, "subtask_result", "Part 2 fixed", {"success": True})

        findings = await store.get_findings(quest.id)
        assert [f["iteration"] for f in findings] == [0, 0, 1]
        assert findings[1]["kind"] == "leader_decision"
        assert findings[1]["data"]["decision"] == "retry"


class TestHistoryStorage:
    async def test_save_quest_persists_stage_history(self, store: SQLiteStore) -> None:
        quest = _quest()
        quest.transition(QuestStatus.POSTED, "quest_board")
        quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        await store.save_quest(quest)

        history = await store.get_history(quest.id)
        transitions = [(h.payload.get("from"), h.payload.get("to")) for h in history]
        assert ("draft", "posted") in transitions
        assert ("posted", "assigned") in transitions

    async def test_history_stays_in_sync_on_resave(self, store: SQLiteStore) -> None:
        quest = _quest()
        quest.transition(QuestStatus.POSTED, "quest_board")
        await store.save_quest(quest)
        assert len(await store.get_history(quest.id)) == 1

        # A later transition + resave must not duplicate earlier entries.
        quest.transition(QuestStatus.ASSIGNED, "guildmaster")
        await store.save_quest(quest)
        history = await store.get_history(quest.id)
        assert len(history) == 2


class TestAdventurerTranscriptCapture:
    async def test_execute_captures_transcript(self) -> None:
        llm = MockChatModel(response_content="All done.")
        adv = GeneralAdventurer(llm=llm)
        adv.grant_talents(["general"])
        result = await adv.execute(_quest())

        assert result.transcript, "expected a captured conversation transcript"
        roles = {m["role"] for m in result.transcript}
        assert "human" in roles and "ai" in roles
        assert any("Build a scraper" in m["content"] for m in result.transcript)

    async def test_transcript_excluded_from_serialization(self) -> None:
        llm = MockChatModel(response_content="All done.")
        adv = GeneralAdventurer(llm=llm)
        adv.grant_talents(["general"])
        result = await adv.execute(_quest())

        # Transcript is carried on the object but not leaked into inter-agent JSON.
        assert "transcript" not in result.model_dump()


class TestGuildPersistsMemory:
    async def test_post_quest_persists_conversation_history_and_result(self) -> None:
        mock_llm = MockChatModel(
            responses=[
                '["general"]',  # refine talents
                '{"title": "Scrape", "description": "Build a scraper", '
                '"acceptance_criteria": ["Works"]}',  # receptionist intake
                '["general"]',  # assess talents
                "no",  # plan_quest — not composite
                "Scraper built and tested.",  # adventurer execution
                '{"accepted": true}',  # verify result
            ]
        )
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .build()
        )
        result = await guild.post_quest("Build a web scraper")
        assert result.success is True

        store = guild._store
        # Conversation transcript from the adventurer was persisted.
        convo = await store.get_conversation(result.quest_id)
        assert convo, "expected persisted conversation"
        assert any(m["role"] == "ai" for m in convo)

        # Stage history captures the full lifecycle including archival.
        history = await store.get_history(result.quest_id)
        stages = [h.payload.get("to") for h in history if h.event_type == "transition"]
        assert "archived" in stages

        # Final result is retrievable from the store.
        persisted = await store.get_result(result.quest_id)
        assert persisted is not None
        assert persisted["success"] is True

        await guild.close()
