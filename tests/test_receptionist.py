"""Tests for Receptionist agent."""

from __future__ import annotations

import pytest

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
