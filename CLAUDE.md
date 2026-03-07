# CLAUDE.md — Guildmaster-AI

## Project Overview

Guildmaster-AI is a Python agentic framework using a fantasy guild metaphor. It orchestrates multiple LLM agents (Adventurers) to plan, execute, verify, and archive tasks (Quests).

## Tech Stack

- **Python 3.11+** with async/await throughout
- **Pydantic v2** for all models and settings
- **uv** for package management
- **aiosqlite** for metadata storage
- **chromadb** for vector/embedding storage
- **httpx** for HTTP (transitive via openrouter SDK)
- **openrouter** (optional extra) — official OpenRouter Python SDK
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
  core/           — Domain models, no LLM logic.
                    Quest, QuestBoard, Party, Messages, Exceptions.
  adventurers/    — All agent implementations.
                    BaseAdventurer is the abstract base.
                    Concrete: GeneralAdventurer, Guildmaster, Receptionist,
                              Librarian, Guard.
  weapons/        — Tool abstractions. BaseWeapon defines the interface.
                    Implementations: FileReadWeapon, WebSearchWeapon.
  armor/          — Guardrail abstractions. BaseArmor defines pre/post hooks.
                    Implementations: ContentFilterArmor, RateLimiterArmor.
  memory/         — SQLiteStore for metadata, ChromaStore for embeddings.
  llm/            — BaseLLMProvider ABC, OpenRouterProvider implementation
                    (uses the official openrouter SDK).
  sdk/            — Guild (runtime) and GuildBuilder (fluent config API).
  config/         — GuildSettings via pydantic-settings (env vars, .env file).
  cli/            — CLI layer (planned, not yet implemented).
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
- `MockLLMProvider` in `conftest.py` is configurable: set `.response_content` and `.tool_calls_to_return`

## Quest Status Flow

```
DRAFT -> POSTED -> ASSIGNED -> IN_PROGRESS -> COMPLETED -> ARCHIVED
                                           -> FAILED -> ARCHIVED
```

## LLM Provider

`OpenRouterProvider` (`guildmaster_ai/llm/openrouter.py`) is the only production implementation.

- Requires the `openrouter` optional extra: `pip install 'guildmaster-ai[openrouter]'`
- Uses the official `openrouter` SDK (`OpenRouter` class, `chat.send_async()`)
- Supports both non-streaming (`complete()`) and streaming (`stream()`)
- Can be used as an async context manager
- Default model: `anthropic/claude-sonnet-4-20250514`

## Adding New Components

- **New Adventurer**: Subclass `BaseAdventurer` (in `guildmaster_ai/adventurers/`), implement `talents`, `system_prompt`, `execute()`
- **New Weapon**: Subclass `BaseWeapon` (in `guildmaster_ai/weapons/`), implement `name`, `description`, `parameters`, `execute()`
- **New Armor**: Subclass `BaseArmor` (in `guildmaster_ai/armor/`), implement `name`, override `pre_process()` and/or `post_process()`
- **New LLM Provider**: Subclass `BaseLLMProvider` (in `guildmaster_ai/llm/`), implement `complete()` and `stream()`
