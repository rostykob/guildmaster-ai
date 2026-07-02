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
                '{"rank": "E", "adventurers": []}',  # triage → simple path
                "Scraper built and tested.",  # adventurer execution
                '{"accepted": true}',  # verify result
                '{"summary": "Scraper quest went well", "tags": ["success"], '
                '"lessons_learned": ["Keep scraping simple"]}',  # librarian analyze
                "- Keep scraping simple",  # chronicle update
            ]
        )
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .with_settings(enable_observations=True)
            .build()
        )
        result = await guild.run_quest("Build a web scraper")
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

        # Librarian observation was persisted to the store (on-demand queryable).
        observations = await store.get_observations(quest_id=result.quest_id)
        assert observations, "expected a persisted librarian observation"

        await guild.close()


class TestObservationStorage:
    async def test_save_and_get_observations(self, store: SQLiteStore) -> None:
        quest = _quest()
        await store.save_quest(quest)
        await store.save_observation(
            obs_id="obs-1",
            quest_id=quest.id,
            summary="Quest succeeded cleanly.",
            tags=["success", "talent:general"],
            lessons_learned=["Kept it simple."],
        )
        obs = await store.get_observations(quest_id=quest.id)
        assert len(obs) == 1
        assert obs[0]["summary"] == "Quest succeeded cleanly."
        assert "success" in obs[0]["tags"]
        assert obs[0]["lessons_learned"] == ["Kept it simple."]

    async def test_librarian_persists_and_reads_on_demand(self, store: SQLiteStore) -> None:
        from guildmaster_ai.adventurers.librarian import Librarian
        from guildmaster_ai.core.messages import QuestResult

        quest = _quest()
        await store.save_quest(quest)
        librarian = Librarian(llm=None, store=store)  # rule-based analysis
        result = QuestResult(sender="adv", quest_id=quest.id, success=True, summary="ok")
        await librarian.archive(quest, result)

        # recent_observations reads back from the store, not the in-memory list.
        fetched = await librarian.recent_observations(quest_id=quest.id)
        assert fetched
        assert fetched[0]["quest_id"] == quest.id


class TestChronicle:
    async def test_archive_updates_chronicle(self, store: SQLiteStore) -> None:
        """Archiving folds lessons into the rolling chronicle summary."""
        from guildmaster_ai.adventurers.librarian import Librarian
        from guildmaster_ai.core.messages import QuestResult

        quest = _quest()
        await store.save_quest(quest)
        librarian = Librarian(llm=None, store=store)  # rule-based chronicle merge
        result = QuestResult(
            sender="adv",
            quest_id=quest.id,
            success=False,
            summary="failed",
            failure_reason="timeout",
        )
        await librarian.archive(quest, result)

        chronicle = await librarian.get_chronicle()
        assert chronicle, "expected a non-empty chronicle after archival"

        # The chronicle survives a fresh librarian instance (persisted state).
        fresh = Librarian(llm=None, store=store)
        assert await fresh.get_chronicle() == chronicle

    async def test_chronicle_empty_without_history(self, store: SQLiteStore) -> None:
        from guildmaster_ai.adventurers.librarian import Librarian

        librarian = Librarian(llm=None, store=store)
        assert await librarian.get_chronicle() == ""

    async def test_chronicle_is_capped(self, store: SQLiteStore) -> None:
        from guildmaster_ai.adventurers.librarian import Librarian
        from guildmaster_ai.core.messages import QuestObservation

        librarian = Librarian(llm=None, store=store)
        obs = QuestObservation(
            sender="librarian",
            quest_id="q",
            summary="s",
            lessons_learned=["x" * 500 for _ in range(10)],
        )
        await librarian._update_chronicle(obs)
        assert len(await librarian.get_chronicle()) <= 2000


class TestBackgroundArchivalAndWiring:
    async def test_background_archival_completes_on_close(self) -> None:
        mock_llm = MockChatModel(response_content="Done.")
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .with_settings(
                background_archival=True,
                enable_vector_store=False,
                enable_observations=True,
            )
            .build()
        )
        result = await guild.run_quest("A backgrounded quest")
        assert result.success is True

        # Archival was scheduled off the request path; draining completes it
        # while the store is still open so we can observe the persisted result.
        await guild._drain_background()
        observations = await guild._store.get_observations(quest_id=result.quest_id)
        assert observations, "background archival should have persisted an observation"

        await guild.close()

    def test_vector_store_wired_when_enabled(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .with_settings(enable_vector_store=True, enable_observations=True)
            .build()
        )
        # The librarian has full access to both stores when the VS is enabled.
        assert guild._vector_store is not None
        assert guild._librarian._vector_store is not None
        assert guild._librarian._store is guild._store

    def test_vector_store_absent_when_disabled(self) -> None:
        mock_llm = MockChatModel()
        guild = (
            GuildBuilder()
            .with_llm_provider(mock_llm)
            .register_adventurer(GeneralAdventurer)
            .with_settings(enable_vector_store=False)
            .build()
        )
        assert guild._vector_store is None
        assert guild._librarian._vector_store is None
