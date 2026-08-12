"""Tests for BaseHero member management and hero-led execution."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.adventurers.base_hero import BaseHero
from guildmaster_ai.adventurers.general_hero import GeneralHero
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.scrolls.catalog import ScrollCatalog

# ── Test helpers ──────────────────────────────────────────────────────


class _TestHero(BaseHero):
    """Concrete BaseHero subclass for testing."""

    system_prompt = "You are a test hero."


class _TestAdventurer(BaseAdventurer):
    """Concrete BaseAdventurer subclass for testing."""

    system_prompt = "You are a test adventurer."


# ── Member management ─────────────────────────────────────────────────


class TestMemberManagement:
    def test_recruit_member(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        member = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(member)
        assert member.id in hero.members
        assert hero.members[member.id] is member

    def test_recruit_multiple_members(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        m1 = _TestAdventurer(name="Fighter", llm=mock_llm)
        m2 = _TestAdventurer(name="Mage", llm=mock_llm)
        hero.recruit(m1)
        hero.recruit(m2)
        assert len(hero.members) == 2

    def test_dismiss_member(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        member = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(member)
        hero.dismiss(member.id)
        assert member.id not in hero.members

    def test_dismiss_nonexistent_is_noop(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        hero.dismiss("nonexistent-id")
        assert len(hero.members) == 0

    def test_member_names(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        m1 = _TestAdventurer(name="Fighter", llm=mock_llm)
        m2 = _TestAdventurer(name="Mage", llm=mock_llm)
        hero.recruit(m1)
        hero.recruit(m2)
        assert hero.member_names == ["Fighter", "Mage"]

    def test_member_names_falls_back_to_id(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        member = _TestAdventurer(llm=mock_llm)  # no name
        hero.recruit(member)
        assert hero.member_names == [member.id]

    def test_members_dict_is_copy(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        member = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(member)
        copy = hero.members
        copy.pop(member.id)
        assert member.id in hero.members


# ── Spawn propagation ─────────────────────────────────────────────────


class TestSpawnWithMembers:
    def test_spawn_propagates_members(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        m1 = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(m1)
        clone = hero.spawn()
        assert m1.id in clone.members

    def test_spawn_members_independent(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        m1 = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(m1)
        clone = hero.spawn()
        clone._members.pop(m1.id)
        assert m1.id in hero.members


# ── Profile ───────────────────────────────────────────────────────────


class TestProfileWithMembers:
    def test_profile_includes_members(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        m1 = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(m1)
        profile = hero.profile()
        assert profile.members == ["Fighter"]

    def test_profile_no_members(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        profile = hero.profile()
        assert profile.members == []


# ── _build_agent with subagents ───────────────────────────────────────


class TestBuildAgentWithSubagents:
    def test_build_agent_passes_subagents(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        member = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(member)

        # Mock the member's _build_agent to return a fake runnable
        mock_runnable = MagicMock()
        with (
            patch.object(member, "_build_agent", return_value=mock_runnable),
            patch(
                "guildmaster_ai.adventurers.base_hero.create_deep_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()
            hero._build_agent()

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["subagents"] is not None
            assert len(call_kwargs["subagents"]) == 1
            assert call_kwargs["subagents"][0]["name"] == "Fighter"
            assert call_kwargs["subagents"][0]["runnable"] is mock_runnable

    def test_build_agent_no_subagents_when_no_members(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)

        with patch(
            "guildmaster_ai.adventurers.base_hero.create_deep_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            hero._build_agent()

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["subagents"] is None


# ── Hero-led execution ────────────────────────────────────────────────


class TestHeroExecution:
    async def test_hero_execute_with_members(self, mock_llm: Any) -> None:
        hero = _TestHero(name="Leader", llm=mock_llm)
        member = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(member)

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Quest completed by party.")]
        }

        quest = Quest(
            title="Party quest",
            description="Do something as a team",
            required_talents=["general"],
            rank=QuestRank.E,
        )

        with patch.object(hero, "_build_agent", return_value=mock_agent):
            result = await hero.execute(quest)

        assert result.success
        assert "Quest completed by party" in result.summary


# ── GeneralHero ───────────────────────────────────────────────────────


class TestGeneralHero:
    def test_has_system_prompt(self) -> None:
        hero = GeneralHero(name="General")
        assert "party leader" in hero.system_prompt

    def test_can_recruit_members(self, mock_llm: Any) -> None:
        hero = GeneralHero(name="General", llm=mock_llm)
        member = _TestAdventurer(name="Fighter", llm=mock_llm)
        hero.recruit(member)
        assert len(hero.members) == 1


# ── Lazy scroll resolution ───────────────────────────────────────────


def _make_catalog() -> ScrollCatalog:
    """Create a temporary ScrollCatalog with one skill."""
    d = Path(tempfile.mkdtemp()) / "scrolls"
    d.mkdir()
    (d / "test-skill").mkdir()
    (d / "test-skill" / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test skill.\n---\nDo the thing.\n"
    )
    return ScrollCatalog(d)


class TestLazyScrollResolution:
    def test_pick_scroll_deferred_resolution(self, mock_llm: Any) -> None:
        """pick_scroll without catalog defers; setting catalog resolves."""
        hero = _TestHero(name="Leader", llm=mock_llm)
        # No catalog — should defer
        hero.pick_scroll("test-skill")
        assert hero.scroll_names == []
        assert hero._pending_scroll_names == ["test-skill"]

        # Set catalog — should resolve
        catalog = _make_catalog()
        hero.scroll_catalog = catalog
        assert "test-skill" in hero.scroll_names
        assert hero._pending_scroll_names == []

    def test_build_agent_raises_on_unresolved(self, mock_llm: Any) -> None:
        """_build_agent raises if scrolls are still pending."""
        hero = _TestHero(name="Leader", llm=mock_llm)
        hero.pick_scroll("test-skill")
        with pytest.raises(RuntimeError, match="Unresolved scrolls"):
            hero._build_agent()

    def test_spawn_copies_pending_scrolls(self, mock_llm: Any) -> None:
        """spawn() propagates pending scroll names."""
        hero = _TestHero(name="Leader", llm=mock_llm)
        hero.pick_scroll("test-skill")
        clone = hero.spawn()
        assert clone._pending_scroll_names == ["test-skill"]
