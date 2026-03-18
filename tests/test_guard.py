"""Tests for the Guard (LLM-as-judge) agent."""

from __future__ import annotations

import pytest

from guildmaster_ai.adventurers.guard import Guard

from .conftest import MockChatModel


class TestGuardEvaluate:
    @pytest.mark.asyncio
    async def test_no_llm_auto_pass(self) -> None:
        guard = Guard()
        verdict = await guard.evaluate("Hello world")
        assert verdict.verdict == "pass"
        assert "auto-pass" in verdict.reason.lower()

    @pytest.mark.asyncio
    async def test_valid_json_response(self) -> None:
        llm = MockChatModel(
            response_content=(
                '{"verdict": "pass", "reason": "Looks good", '
                '"hallucination": 0.1, "accuracy": 0.9, '
                '"relevance": 0.95, "toxicity": 0.0}'
            )
        )
        guard = Guard(llm=llm)
        verdict = await guard.evaluate("Some content")
        assert verdict.verdict == "pass"
        assert verdict.reason == "Looks good"
        assert verdict.metrics.hallucination == pytest.approx(0.1)
        assert verdict.metrics.accuracy == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self) -> None:
        llm = MockChatModel(
            response_content=(
                '```json\n{"verdict": "warn", "reason": "Marginal accuracy", '
                '"hallucination": 0.2, "accuracy": 0.55, '
                '"relevance": 0.8, "toxicity": 0.1}\n```'
            )
        )
        guard = Guard(llm=llm)
        verdict = await guard.evaluate("Some content")
        assert verdict.verdict == "warn"
        assert verdict.metrics.accuracy == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_unparseable_response_defaults_to_pass(self) -> None:
        llm = MockChatModel(response_content="I can't produce JSON right now.")
        guard = Guard(llm=llm)
        verdict = await guard.evaluate("Some content")
        assert verdict.verdict == "pass"
        assert "could not parse" in verdict.reason.lower()

    @pytest.mark.asyncio
    async def test_exception_fallback_to_pass(self) -> None:
        """When the LLM call itself raises, guard should still return pass."""

        class FailingLLM(MockChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        guard = Guard(llm=FailingLLM())
        verdict = await guard.evaluate("Some content")
        assert verdict.verdict == "pass"
        assert "failed" in verdict.reason.lower()

    @pytest.mark.asyncio
    async def test_criteria_and_context_passed(self) -> None:
        """Ensure criteria and context are included in the user prompt."""
        llm = MockChatModel(
            response_content='{"verdict": "pass", "reason": "ok"}'
        )
        guard = Guard(llm=llm)
        verdict = await guard.evaluate(
            "Result text",
            criteria=["Must be accurate"],
            context="Original quest",
        )
        assert verdict.verdict == "pass"

    @pytest.mark.asyncio
    async def test_block_verdict(self) -> None:
        llm = MockChatModel(
            response_content=(
                '{"verdict": "block", "reason": "Toxic content", '
                '"hallucination": 0.0, "accuracy": 0.8, '
                '"relevance": 0.9, "toxicity": 0.9}'
            )
        )
        guard = Guard(llm=llm)
        verdict = await guard.evaluate("Bad content")
        assert verdict.verdict == "block"
        assert verdict.metrics.toxicity == pytest.approx(0.9)


class TestGuardParseResponse:
    def test_pass_verdict(self) -> None:
        guard = Guard()
        verdict = guard._parse_response(
            '{"verdict": "pass", "reason": "All good", '
            '"hallucination": 0.0, "accuracy": 1.0, '
            '"relevance": 1.0, "toxicity": 0.0}'
        )
        assert verdict.verdict == "pass"
        assert verdict.reason == "All good"

    def test_warn_verdict(self) -> None:
        guard = Guard()
        verdict = guard._parse_response(
            '{"verdict": "warn", "reason": "Marginal"}'
        )
        assert verdict.verdict == "warn"

    def test_block_verdict(self) -> None:
        guard = Guard()
        verdict = guard._parse_response(
            '{"verdict": "block", "reason": "Toxic"}'
        )
        assert verdict.verdict == "block"

    def test_invalid_verdict_defaults_to_pass(self) -> None:
        guard = Guard()
        verdict = guard._parse_response(
            '{"verdict": "unknown", "reason": "Huh"}'
        )
        assert verdict.verdict == "pass"

    def test_missing_fields_use_defaults(self) -> None:
        guard = Guard()
        verdict = guard._parse_response('{"verdict": "pass"}')
        assert verdict.reason == ""
        assert verdict.metrics.hallucination == 0.0
        assert verdict.metrics.accuracy == 1.0
        assert verdict.metrics.relevance == 1.0
        assert verdict.metrics.toxicity == 0.0

    def test_garbage_input_defaults_to_pass(self) -> None:
        guard = Guard()
        verdict = guard._parse_response("not json at all!!!")
        assert verdict.verdict == "pass"
        assert "could not parse" in verdict.reason.lower()

    def test_name_property(self) -> None:
        guard = Guard()
        assert guard.name == "guard"
