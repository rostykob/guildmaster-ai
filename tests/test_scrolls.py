"""Tests for scrolls (SKILL.md-based agent skills)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from guildmaster_ai.adventurers.base_hero import BaseHero
from guildmaster_ai.core.exceptions import ScrollValidationError
from guildmaster_ai.core.quest import Quest, QuestRank
from guildmaster_ai.scrolls.catalog import ScrollCatalog
from guildmaster_ai.scrolls.scroll import Scroll
from guildmaster_ai.scrolls.skill_parser import validate_scroll_name

# ── Test helpers ──────────────────────────────────────────────────────


class _TestHero(BaseHero):
    """Concrete BaseHero subclass for testing."""

    system_prompt = "You are a test hero."


def _make_skill_dir(tmp_path: Path, name: str, desc: str, body: str, **extra: Any) -> Path:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(exist_ok=True)
    fm_lines = [f"name: {name}", f"description: {desc}"]
    for k, v in extra.items():
        fm_lines.append(f"{k}: {v}")
    content = "---\n" + "\n".join(fm_lines) + "\n---\n" + body
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


# ── Scroll name validation ──────────────────────────────────────────


class TestScrollNameValidation:
    def test_valid_names(self) -> None:
        for name in ["research", "code-review", "a", "a1", "my-skill-v2"]:
            assert validate_scroll_name(name) == name

    def test_empty_name(self) -> None:
        with pytest.raises(ScrollValidationError, match="must not be empty"):
            validate_scroll_name("")

    def test_too_long(self) -> None:
        with pytest.raises(ScrollValidationError, match="at most 64"):
            validate_scroll_name("a" * 65)

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ScrollValidationError, match="Invalid scroll name"):
            validate_scroll_name("Research")

    def test_leading_hyphen(self) -> None:
        with pytest.raises(ScrollValidationError, match="Invalid scroll name"):
            validate_scroll_name("-research")

    def test_trailing_hyphen(self) -> None:
        with pytest.raises(ScrollValidationError, match="Invalid scroll name"):
            validate_scroll_name("research-")

    def test_consecutive_hyphens(self) -> None:
        with pytest.raises(ScrollValidationError, match="Invalid scroll name"):
            validate_scroll_name("code--review")

    def test_spaces_rejected(self) -> None:
        with pytest.raises(ScrollValidationError, match="Invalid scroll name"):
            validate_scroll_name("code review")

    def test_underscores_rejected(self) -> None:
        with pytest.raises(ScrollValidationError, match="Invalid scroll name"):
            validate_scroll_name("code_review")


# ── Scroll (loading from SKILL.md) ─────────────────────────────────


class TestScroll:
    def test_load_valid_skill(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "my-skill", "A test skill.", "Do the thing.")
        scroll = Scroll(d)
        assert scroll.name == "my-skill"
        assert scroll.description == "A test skill."
        assert scroll.instructions == "Do the thing."

    def test_discovery_text(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "disco", "Discover me.", "Full text.")
        scroll = Scroll(d)
        assert scroll.discovery_text == "disco: Discover me."

    def test_frontmatter_optional_fields(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "licensed-skill", "Licensed.", "Instructions.", license="MIT")
        scroll = Scroll(d)
        assert scroll.license == "MIT"
        assert scroll.compatibility is None
        assert scroll.metadata == {}

    def test_dir_name_mismatch_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "wrong-name"
        skill_dir.mkdir()
        content = "---\nname: correct-name\ndescription: X\n---\nBody."
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        with pytest.raises(ScrollValidationError, match="does not match"):
            Scroll(skill_dir)

    def test_missing_skill_md_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ScrollValidationError, match=r"No SKILL\.md"):
            Scroll(empty_dir)

    def test_has_resource(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "res-skill", "Resources.", "Body.")
        refs = d / "references"
        refs.mkdir()
        (refs / "data.txt").write_text("hello", encoding="utf-8")
        scroll = Scroll(d)
        assert scroll.has_resource("references/data.txt")
        assert not scroll.has_resource("references/missing.txt")

    def test_has_resource_rejects_traversal(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "sec-skill", "Secure.", "Body.")
        scroll = Scroll(d)
        assert not scroll.has_resource("../../etc/passwd")

    async def test_read_resource(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "read-skill", "Readable.", "Body.")
        refs = d / "references"
        refs.mkdir()
        (refs / "info.txt").write_text("some data", encoding="utf-8")
        scroll = Scroll(d)
        content = await scroll.read_resource("references/info.txt")
        assert content == "some data"

    async def test_read_resource_traversal_raises(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "trav-skill", "Traversal.", "Body.")
        scroll = Scroll(d)
        with pytest.raises(ScrollValidationError, match="traversal"):
            await scroll.read_resource("../../etc/passwd")

    async def test_read_resource_missing_raises(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "miss-skill", "Missing.", "Body.")
        scroll = Scroll(d)
        with pytest.raises(ScrollValidationError, match="not found"):
            await scroll.read_resource("nope.txt")

    def test_resource_dir(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "dir-skill", "Dir.", "Body.")
        scroll = Scroll(d)
        assert scroll.resource_dir == d.resolve()

    def test_skill_with_scripts(self, tmp_path: Path) -> None:
        d = _make_skill_dir(
            tmp_path,
            "scripted",
            "Has scripts.",
            "Run scripts/extract.py to extract data.",
        )
        scripts = d / "scripts"
        scripts.mkdir()
        (scripts / "extract.py").write_text("print('hello')", encoding="utf-8")
        scroll = Scroll(d)
        assert scroll.has_resource("scripts/extract.py")
        assert "scripts/extract.py" in scroll.instructions

    def test_repr(self, tmp_path: Path) -> None:
        d = _make_skill_dir(tmp_path, "repr-skill", "Repr.", "Body.")
        scroll = Scroll(d)
        assert "repr-skill" in repr(scroll)


# ── ScrollCatalog ───────────────────────────────────────────────────


class TestScrollCatalog:
    def test_get_by_name(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Alpha instructions.")
        catalog = ScrollCatalog(tmp_path)
        scroll = catalog.get("alpha")
        assert scroll.name == "alpha"
        assert scroll.instructions == "Alpha instructions."

    def test_get_not_found(self, tmp_path: Path) -> None:
        catalog = ScrollCatalog(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            catalog.get("nonexistent")

    def test_list_names(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Body.")
        _make_skill_dir(tmp_path, "beta", "Second.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        assert catalog.list_names() == ["alpha", "beta"]

    def test_list_scrolls(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Body.")
        _make_skill_dir(tmp_path, "beta", "Second.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        scrolls = catalog.list_scrolls()
        assert len(scrolls) == 2
        assert scrolls[0].name == "alpha"

    def test_contains(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        assert "alpha" in catalog
        assert "beta" not in catalog

    def test_len(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Body.")
        _make_skill_dir(tmp_path, "beta", "Second.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        assert len(catalog) == 2

    def test_skips_dirs_without_skill_md(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "valid", "Valid.", "Body.")
        (tmp_path / "no-skill").mkdir()
        catalog = ScrollCatalog(tmp_path)
        assert catalog.list_names() == ["valid"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        catalog = ScrollCatalog(tmp_path)
        assert len(catalog) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        catalog = ScrollCatalog(tmp_path / "nonexistent")
        assert len(catalog) == 0

    def test_skips_invalid_scrolls(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "good", "Valid.", "Body.")
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
        catalog = ScrollCatalog(tmp_path)
        assert catalog.list_names() == ["good"]


# ── Built-in skills ─────────────────────────────────────────────────


def _builtin_skills_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "guildmaster_ai" / "scrolls" / "skills"


class TestBuiltInSkills:
    def test_research_loads(self) -> None:
        catalog = ScrollCatalog(_builtin_skills_dir())
        assert "research" in catalog
        scroll = catalog.get("research")
        assert scroll.name == "research"
        assert "research" in scroll.description.lower()
        assert len(scroll.instructions) > 0

    def test_data_profiler_loads(self) -> None:
        catalog = ScrollCatalog(_builtin_skills_dir())
        assert "data-profiler" in catalog
        scroll = catalog.get("data-profiler")
        assert scroll.name == "data-profiler"
        assert scroll.has_resource("scripts/profile_csv.py")

    async def test_data_profiler_script_runs(self, tmp_path: Path) -> None:
        """The bundled profile_csv.py script produces valid JSON output."""
        import asyncio
        import json

        # Create a test CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "name,age,score\nAlice,30,95\nBob,,80\nCharlie,25,\n",
            encoding="utf-8",
        )

        script = _builtin_skills_dir() / "data-profiler" / "scripts" / "profile_csv.py"
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(script),
            str(csv_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0

        report = json.loads(stdout.decode())
        assert report["row_count"] == 3
        assert report["column_count"] == 3
        assert report["columns"]["age"]["missing"] == 1
        assert report["columns"]["score"]["dtype"] == "numeric"


# ── pick_scroll (by name from catalog) ──────────────────────────────


class TestPickScroll:
    def test_pick_by_name(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "my-skill", "A skill.", "Do the thing.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        adv.pick_scroll("my-skill")
        assert "my-skill" in adv.scroll_names
        assert adv.scrolls["my-skill"].instructions == "Do the thing."

    def test_pick_not_found(self, mock_llm: Any, tmp_path: Path) -> None:
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        with pytest.raises(KeyError, match="not found"):
            adv.pick_scroll("nonexistent")

    def test_pick_no_catalog_defers(self, mock_llm: Any) -> None:
        adv = _TestHero(name="Tester", llm=mock_llm)
        # Without a catalog, pick_scroll defers resolution
        adv.pick_scroll("some-skill")
        assert adv.scroll_names == []
        assert adv._pending_scroll_names == ["some-skill"]

    def test_pick_multiple_scrolls(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Alpha body.")
        _make_skill_dir(tmp_path, "beta", "Second.", "Beta body.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        adv.pick_scroll("alpha")
        adv.pick_scroll("beta")
        assert adv.scroll_names == ["alpha", "beta"]


# ── spawn ────────────────────────────────────────────────────────────


class TestSpawnWithScrolls:
    def test_spawn_propagates_scrolls(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "alpha", "First.", "Body.")
        _make_skill_dir(tmp_path, "beta", "Second.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        adv.pick_scroll("alpha")
        adv.pick_scroll("beta")
        clone = adv.spawn()
        assert clone.scroll_names == ["alpha", "beta"]

    def test_spawn_propagates_catalog(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "cat-skill", "Cataloged.", "Instructions.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        clone = adv.spawn()
        assert clone.scroll_catalog is catalog
        clone.pick_scroll("cat-skill")
        assert "cat-skill" in clone.scroll_names

    def test_spawn_scrolls_independent(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "test-scroll", "Test.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        adv.pick_scroll("test-scroll")
        clone = adv.spawn()
        clone._scrolls.pop("test-scroll")
        assert "test-scroll" in adv.scroll_names


# ── profile ──────────────────────────────────────────────────────────


class TestProfileWithScrolls:
    def test_profile_includes_scrolls(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "my-scroll", "Mine.", "Body.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        adv.pick_scroll("my-scroll")
        profile = adv.profile()
        assert profile.scrolls == ["my-scroll"]

    def test_profile_no_scrolls(self, mock_llm: Any) -> None:
        adv = _TestHero(name="Tester", llm=mock_llm)
        profile = adv.profile()
        assert profile.scrolls == []


# ── execute uses deep agent ──────────────────────────────────────────


class TestExecuteWithScrolls:
    async def test_execute_with_scroll(self, mock_llm: Any, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "exec-scroll", "Exec.", "Execute instructions.")
        catalog = ScrollCatalog(tmp_path)
        adv = _TestHero(name="Tester", llm=mock_llm, scroll_catalog=catalog)
        adv.pick_scroll("exec-scroll")

        # Mock _build_agent to return a fake agent with ainvoke
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="Done with scroll.")]}

        quest = Quest(
            title="Scroll test",
            description="Do something",
            required_talents=["general"],
            rank=QuestRank.E,
        )
        with patch.object(adv, "_build_agent", return_value=mock_agent):
            result = await adv.execute(quest)
        assert result.success
        assert "Done with scroll" in result.summary
