"""Tests for Receptionist agent."""

from __future__ import annotations

import pytest
from pydantic import Field

from guildmaster_ai.adventurers.receptionist import Receptionist

from .conftest import MockChatModel


class TestReceptionist:
    @pytest.mark.asyncio
    async def test_intake_no_llm(self) -> None:
        r = Receptionist()
        draft = await r.intake("Build a web scraper")
        assert draft.title == "Build a web scraper"
        assert draft.description == "Build a web scraper"

    @pytest.mark.asyncio
    async def test_intake_with_llm(self) -> None:
        llm = MockChatModel(
            response_content='{"title": "Web Scraper", "description": "Build a scraper", '
            '"acceptance_criteria": ["Works"]}'
        )
        r = Receptionist(llm=llm)
        draft = await r.intake("Build a web scraper")
        assert draft.title == "Web Scraper"
        assert draft.acceptance_criteria == ["Works"]
        # Receptionist no longer assigns talents — that's the Guildmaster's job
        assert draft.required_talents == []

    @pytest.mark.asyncio
    async def test_intake_preserves_request_and_captures_constraint(self) -> None:
        # Even if the LLM paraphrases the request in "description", the draft
        # must keep the user's original wording (it's what the adventurer runs)
        # and surface output constraints as acceptance criteria.
        llm = MockChatModel(
            response_content='{"title": "Paris, the Capital", '
            '"description": "Find the capital city of France.", '
            '"acceptance_criteria": ["Reply in one word"]}'
        )
        r = Receptionist(llm=llm)
        draft = await r.intake("Capital of France? One word.")
        assert draft.description == "Capital of France? One word."
        assert draft.acceptance_criteria == ["Reply in one word"]
        assert draft.title == "Paris, the Capital"

    @pytest.mark.asyncio
    async def test_intake_with_clarification(self) -> None:
        r = Receptionist()  # no LLM — uses basic draft

        async def fake_clarify(questions: list[str]) -> dict[str, str]:
            return {q: "Yes, please do it well." for q in questions}

        draft = await r.intake("Do X", clarify=fake_clarify)
        # Without LLM the draft stays simple, but clarifications should
        # not crash even though the LLM path is skipped
        assert draft.title == "Do X"

    @pytest.mark.asyncio
    async def test_present_result_success(self) -> None:
        from guildmaster_ai.core.messages import QuestResult

        r = Receptionist()
        result = QuestResult(
            sender="adventurer",
            quest_id="q1",
            success=True,
            summary="All done.",
        )
        text = await r.present_result(result)
        assert "completed successfully" in text

    @pytest.mark.asyncio
    async def test_present_result_failure(self) -> None:
        from guildmaster_ai.core.messages import QuestResult

        r = Receptionist()
        result = QuestResult(
            sender="adventurer",
            quest_id="q1",
            success=False,
            summary="Oops",
            failure_reason="timeout",
        )
        text = await r.present_result(result)
        assert "failed" in text
        assert "timeout" in text

    @pytest.mark.asyncio
    async def test_identify_gaps_empty_draft(self) -> None:
        questions = Receptionist._identify_gaps(
            __import__("guildmaster_ai.core.messages", fromlist=["QuestDraft"]).QuestDraft(
                title="X",
                description="X",
            )
        )
        # Brief description + no criteria → 2 questions (talents are Guildmaster's job)
        assert len(questions) == 2


class _RecordingMock(MockChatModel):
    """MockChatModel that records the system prompt of every call."""

    recorded_systems: list[str] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        self.recorded_systems.append(str(messages[0].content) if messages else "")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class TestCharter:
    @pytest.mark.asyncio
    async def test_charter_appended_to_refine_prompt(self) -> None:
        llm = _RecordingMock(
            response_content='{"title": "T", "acceptance_criteria": []}',
            recorded_systems=[],
        )
        r = Receptionist(llm=llm, charter="Only tutor English exam students.")
        await r.intake("Help me practice for the exam")
        assert any("Only tutor English exam students." in s for s in llm.recorded_systems)


class TestMissingInputs:
    @pytest.mark.asyncio
    async def test_no_armoury_no_llm_call(self) -> None:
        llm = MockChatModel(response_content="should never be used")
        r = Receptionist(llm=llm)
        questions = await r._identify_missing_inputs(
            __import__("guildmaster_ai.core.messages", fromlist=["QuestDraft"]).QuestDraft(
                title="X", description="X"
            )
        )
        assert questions == []
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_empty_armoury_no_llm_call(self) -> None:
        llm = MockChatModel(response_content="should never be used")
        r = Receptionist(llm=llm)
        r.connect_armoury(lambda: [])
        questions = await r._identify_missing_inputs(
            __import__("guildmaster_ai.core.messages", fromlist=["QuestDraft"]).QuestDraft(
                title="X", description="X"
            )
        )
        assert questions == []
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_missing_input_question_surfaces_in_intake(self) -> None:
        # Call order: 1) refine draft, 2) missing-input check (one question),
        # 3) missing-input check after clarification (nothing missing).
        llm = MockChatModel(
            responses=[
                '{"title": "Read report", '
                '"acceptance_criteria": ["Summary of the report"]}',
                '["Which file should be read?"]',
                "[]",
            ]
        )
        r = Receptionist(llm=llm)
        r.connect_armoury(
            lambda: [
                {
                    "name": "file_read",
                    "description": "Read a file from disk",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        )

        asked: list[str] = []

        async def clarify(questions: list[str]) -> dict[str, str]:
            asked.extend(questions)
            return {q: "/tmp/report.txt" for q in questions}

        draft = await r.intake("Summarise the quarterly report file for me please", clarify=clarify)
        assert "Which file should be read?" in asked
        assert "/tmp/report.txt" in draft.description

    @pytest.mark.asyncio
    async def test_unparseable_response_yields_no_questions(self) -> None:
        llm = MockChatModel(response_content="not json at all")
        r = Receptionist(llm=llm)
        r.connect_armoury(lambda: [{"name": "w", "description": "d", "parameters": {}}])
        questions = await r._identify_missing_inputs(
            __import__("guildmaster_ai.core.messages", fromlist=["QuestDraft"]).QuestDraft(
                title="X", description="X"
            )
        )
        assert questions == []
