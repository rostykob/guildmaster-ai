# CLAUDE.md — Guildmaster-AI

## Project Overview

Guildmaster-AI is a Python agentic framework using a fantasy guild metaphor. It orchestrates multiple LLM agents (Adventurers) to plan, execute, verify, and archive tasks (Quests).

## Tech Stack

- **Python 3.11+** with async/await throughout
- **Pydantic v2** for all models and settings
- **uv** for package management
- **aiosqlite** for metadata storage
- **chromadb** for vector/embedding storage
- **LangChain 1.x** (`langchain`, `langchain-core`, `langchain-openai`) for LLM connections and tool calling
- **pytest + pytest-asyncio** for testing
- **ruff** for linting, **mypy** for type checking

## Key Commands

```bash
# Install
uv sync --all-extras

# Test
uv run pytest
uv run pytest tests/test_quest.py -v          # specific test file
uv run pytest -k "test_transition" -v         # specific test pattern

# Lint & type check
uv tool run ruff check .
uv tool run ruff check --fix .
```

## Architecture

```
guildmaster_ai/
  core/           — Domain models, no LLM logic.
                    Quest, QuestBoard, Party, Messages, Exceptions.
  adventurers/    — All agent implementations.
                    BaseAdventurer is the abstract base (uses create_agent).
                    BaseHero extends BaseAdventurer with scroll + member support
                      (uses create_deep_agent with subagents).
                    GeneralHero is the concrete hero for party leadership.
                    Concrete agents: GeneralAdventurer, Guildmaster, Receptionist,
                              Librarian, Guard.
  weapons/        — Tool abstractions. BaseWeapon extends LangChain BaseTool.
                    Implementations: FileReadWeapon, WebSearchWeapon.
  armor/          — Guardrail abstractions. BaseArmor defines pre/post hooks.
                    Implementations: ContentFilterArmor, RateLimiterArmor.
  scrolls/        — SKILL.md-based agent skills (agentskills.io spec).
                    Scroll loads from a skill-name/SKILL.md directory.
                    ScrollCatalog scans a directory for name-based lookup.
                    Built-in skills in scrolls/skills/ (e.g. research).
                    pick_scroll("name") resolves from catalog; progressive disclosure.
  memory/         — SQLiteStore for metadata, ChromaStore for embeddings.
  llm/            — LangChain chat model factory and provider wrappers.
                    types.py (GuildLLM, GuildResponse), base_provider.py (factory),
                    openrouter.py, anthropic.py, openai.py, google.py, azure.py, bedrock.py.
  sdk/            — Guild (runtime) and GuildBuilder (fluent config API).
  config/         — GuildSettings via pydantic-settings (env vars, .env file).
  cli/            — CLI layer (planned, not yet implemented).
```

## Conventions

- **Async-first**: All agent methods and I/O operations are async
- **Pydantic models**: All data structures use Pydantic v2 BaseModel
- **`from __future__ import annotations`** in every Python file
- **Type annotations**: Full typing on all public APIs
- **Env prefix**: Framework settings use the `GUILD_` prefix (e.g., `GUILD_LLM_PROVIDER`).
  Provider API keys use their **official env var names** (e.g., `OPENROUTER_API_KEY`) —
  no `GUILD_` prefix — so existing credentials work out of the box.
- **Domain language**: Use the guild metaphor consistently (Quest not Task, Weapon not Tool, Armor not Guardrail, Adventurer not Agent)
- **State transitions**: Quest status changes must go through `Quest.transition()` which validates against `VALID_TRANSITIONS`
- **Inter-agent comms**: Agents exchange Pydantic message objects, not raw strings
- **LangChain isolation**: All LangChain imports are confined to `llm/types.py`, `llm/openrouter.py`, and `weapons/base_weapon.py`. Other modules use guild-native wrappers (`GuildLLM`, `GuildResponse`, `guild_complete`).

## Testing

- Tests live in `tests/`
- Use `pytest-asyncio` with `asyncio_mode = "auto"`
- Mock LLM providers for unit tests — never call real APIs in tests
- Test fixtures go in `tests/conftest.py`
- `MockChatModel` in `conftest.py` extends LangChain `BaseChatModel` with configurable responses

## Quest Status Flow

```
DRAFT -> POSTED -> ASSIGNED -> IN_PROGRESS -> COMPLETED -> ARCHIVED
                                           -> FAILED -> ARCHIVED
```

## Complex Quest Execution

Complex quests (multiple subtasks) support two execution paths:

1. **Hero-led** (preferred): If a `BaseHero` is registered and matches the quest, it becomes party leader. Matched adventurers are recruited as subagents via `CompiledSubAgent`. The deep agent coordinates delegation via its `task` tool.
2. **Manual party** (fallback): Without a hero, the guild distributes subtasks across adventurers manually with concurrent execution and retry logic.

Simple quests always use a single adventurer regardless.

## Adding New Components

- **New Adventurer**: Subclass `BaseAdventurer`, implement `system_prompt`
- **New Hero**: Subclass `BaseHero` (or `GeneralHero`), implement `system_prompt`. Use `recruit()` to add member adventurers as subagents. Heroes use `create_deep_agent` and support scrolls + members.
- **New Weapon**: Subclass `BaseWeapon` (extends LangChain `BaseTool`), set `name`, `description`, `args_schema`, implement `execute()`
- **New Armor**: Subclass `BaseArmor`, implement `name`, override `pre_process()` and/or `post_process()`
- **New Scroll**: Create a directory `scrolls/<skill-name>/SKILL.md` with YAML frontmatter (`name`, `description`) and markdown instructions. Add optional `scripts/`, `references/`, `assets/` subdirectories. Load via `ScrollCatalog` and `pick_scroll("skill-name")`.
- **New LLM Provider**: Pass any LangChain `BaseChatModel` to `GuildBuilder.with_llm_provider()`, or add a new provider to `create_chat_model()` factory


## Workflow Orchestration

**1. Plan Mode Default**
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

**2. Subagent Strategy**
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

**3. Self-Improvement Loop**
- After ANY correction from the user: update `.claude/tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

**4. Verification Before Done**
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

**5. Demand Elegance (Balanced)**
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

**6. Autonomous Bug Fixing**
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

**Task Management**
- Plan First: Write plan to `.claude/tasks/todo.md` with checkable items
- Verify Plan: Check in before starting implementation
- Track Progress: Mark items complete as you go
- Explain Changes: High-level summary at each step
- Document Results: Add review section to `.claude/tasks/todo.md`
- Capture Lessons: Update `.claude/tasks/lessons.md` after corrections

**Core Principles**
- Simplicity First: Make every change as simple as possible. Impact minimal code.
- No Laziness: Find root causes. No temporary fixes. Senior developer standards.
- Minimal Impact: Changes should only touch what's necessary. Avoid introducing bugs.


