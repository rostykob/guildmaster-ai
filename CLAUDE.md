# CLAUDE.md — Guildmaster-AI

## Project Overview

Guildmaster-AI is a Python agentic framework using a fantasy guild metaphor. It orchestrates multiple LLM agents (Adventurers) to plan, execute, verify, and archive tasks (Quests).

## Tech Stack

- **Python 3.11+** with async/await throughout
- **Pydantic v2** for all models and settings
- **uv** for package management
- **aiosqlite** for metadata storage
- **chromadb** for vector/embedding storage
- **httpx** for HTTP (transitive via provider SDKs)
- **LLM provider extras** (all optional — install only what you need):
  - `openrouter` — official OpenRouter SDK (`OPENROUTER_API_KEY`)
  - `anthropic` — Anthropic SDK (`ANTHROPIC_API_KEY`)
  - `openai` — OpenAI SDK (`OPENAI_API_KEY`); also used by the `azure` extra
  - `google` — Google Generative AI SDK (`GEMINI_API_KEY` / `GOOGLE_API_KEY`)
  - `azure` — Azure OpenAI via `openai` SDK (`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`)
  - `bedrock` — AWS Bedrock via `boto3` (`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`)
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
  llm/            — BaseLLMProvider ABC + one concrete file per provider:
                    openrouter.py, anthropic.py, openai.py,
                    google.py, azure.py, bedrock.py.
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
  Provider API keys use their **official env var names** (e.g., `OPENROUTER_API_KEY`,
  `ANTHROPIC_API_KEY`) — no `GUILD_` prefix — so existing credentials work out of the box.
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

## LLM Providers

All providers live in `guildmaster_ai/llm/` and are optional extras.  Each raises a clear
`ImportError` at construction time if its SDK is missing.  All support `complete()`,
`stream()`, and async context-manager usage.

| File | Class | Extra | SDK | Default model |
|---|---|---|---|---|
| `openrouter.py` | `OpenRouterProvider` | `openrouter` | `openrouter` SDK | `anthropic/claude-sonnet-4-20250514` |
| `anthropic.py` | `AnthropicProvider` | `anthropic` | `anthropic` SDK | `claude-sonnet-4-5` |
| `openai.py` | `OpenAIProvider` | `openai` | `openai` SDK | `gpt-4o` |
| `google.py` | `GoogleProvider` | `google` | `google-genai` SDK | `gemini-2.0-flash` |
| `azure.py` | `AzureOpenAIProvider` | `azure` | `openai` SDK | `gpt-4o` |
| `bedrock.py` | `BedrockProvider` | `bedrock` | `boto3` | `anthropic.claude-3-5-sonnet-20241022-v2:0` |

Key format-conversion notes:
- **Anthropic**: system prompt extracted to top-level `system`; tools use `input_schema`; tool results are `tool_result` content blocks.
- **Google**: system prompt → `system_instruction`; tool calls use `function_call`/`function_response` parts.
- **Bedrock**: uses the Converse API; sync boto3 calls wrapped in `asyncio.to_thread`.
- **OpenAI / Azure / OpenRouter**: native OpenAI format, no conversion needed.

## Adding New Components

- **New Adventurer**: Subclass `BaseAdventurer` (in `guildmaster_ai/adventurers/`), implement `talents`, `system_prompt`, `execute()`
- **New Weapon**: Subclass `BaseWeapon` (in `guildmaster_ai/weapons/`), implement `name`, `description`, `parameters`, `execute()`
- **New Armor**: Subclass `BaseArmor` (in `guildmaster_ai/armor/`), implement `name`, override `pre_process()` and/or `post_process()`
- **New LLM Provider**: Subclass `BaseLLMProvider` (in `guildmaster_ai/llm/`), implement `complete()` and `stream()`
