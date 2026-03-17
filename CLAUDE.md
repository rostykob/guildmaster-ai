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
pytest
pytest tests/test_quest.py -v          # specific test file
pytest -k "test_transition" -v         # specific test pattern

# Lint & type check
ruff check .
ruff check --fix .
mypy guildmaster_ai
```

## Architecture

```
guildmaster_ai/
  core/       — Domain models, no LLM logic. Quest, QuestBoard, Party, Messages, Exceptions.
  adventurers/ — All agent implementations. BaseAdventurer is the abstract base.
  weapons/    — Tool abstractions. BaseWeapon extends LangChain BaseTool.
  armor/      — Guardrail abstractions. BaseArmor defines pre/post hooks.
  memory/     — SQLiteStore for metadata, ChromaStore for embeddings.
  llm/        — LangChain chat model factory, ChatOpenRouter wrapper.
  sdk/        — Guild (runtime) and GuildBuilder (fluent config API).
  config/     — GuildSettings via pydantic-settings (env vars, .env file).
```

## Conventions

- **Async-first**: All agent methods and I/O operations are async
- **Pydantic models**: All data structures use Pydantic v2 BaseModel
- **`from __future__ import annotations`** in every Python file
- **Type annotations**: Full typing on all public APIs
- **Env prefix**: All settings use `GUILD_` prefix (e.g., `GUILD_LLM_API_KEY`)
- **Domain language**: Use the guild metaphor consistently (Quest not Task, Weapon not Tool, Armor not Guardrail, Adventurer not Agent)
- **State transitions**: Quest status changes must go through `Quest.transition()` which validates against `VALID_TRANSITIONS`
- **Inter-agent comms**: Agents exchange Pydantic message objects, not raw strings

## Testing

- Tests live in `tests/`
- Use `pytest-asyncio` with `asyncio_mode = "auto"`
- Mock LLM providers for unit tests — never call real APIs in tests
- Test fixtures go in `tests/conftest.py`

## Quest Status Flow

```
DRAFT -> POSTED -> ASSIGNED -> IN_PROGRESS -> COMPLETED -> ARCHIVED
                                           -> FAILED -> ARCHIVED
```

## Adding New Components

- **New Adventurer**: Subclass `BaseAdventurer`, implement `talents`, `system_prompt`, `execute()`
- **New Weapon**: Subclass `BaseWeapon` (extends LangChain `BaseTool`), set `name`, `description`, `args_schema`, implement `execute()`
- **New Armor**: Subclass `BaseArmor`, implement `name`, override `pre_process()` and/or `post_process()`
- **New LLM Provider**: Pass any LangChain `BaseChatModel` to `GuildBuilder.with_llm_provider()`, or add a new provider to `create_chat_model()` factory
