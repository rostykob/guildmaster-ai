# Guildmaster-AI

An agentic framework with a fantasy guild metaphor.

Guildmaster-AI is an open-source Python framework for orchestrating multiple LLM agents to collaboratively plan, execute, verify, and archive complex tasks — called **Quests**.

## Concepts

| Concept | Description |
|---|---|
| **Guild** | The runtime — contains a Guildmaster, Receptionist, Librarian, and a roster of Adventurers |
| **Quest** | A structured unit of work with a title, description, required talents, rank, and acceptance criteria |
| **Adventurer** | An agent that executes quests using Weapons (tools) and wearing Armor (guardrails) |
| **Guildmaster** | The orchestration brain — matches quests to adventurers, forms parties, verifies results |
| **Receptionist** | User-facing intake agent — clarifies requests and produces well-formed quest specs |
| **Librarian** | Memory agent — archives quest results and retrieves past context |
| **Guard** | Optional LLM-as-judge — evaluates outputs for correctness and policy compliance |
| **Weapon** | A tool abstraction (e.g. web search, file read, code execution) |
| **Armor** | A guardrail abstraction (e.g. content filter, rate limiter, PII scrubber) |
| **Party** | A temporary coalition of adventurers for complex quests |
| **Quest Board** | Priority queue of posted quests awaiting assignment |

## Quest Lifecycle

```
User Request -> Receptionist (clarify) -> Guildmaster (feasibility)
  -> Quest Board (post) -> Adventurer(s) (execute)
  -> Guard (optional verify) -> Guildmaster (verify) -> Librarian (archive) -> User Result
```

Quest status transitions:
```
DRAFT -> POSTED -> ASSIGNED -> IN_PROGRESS -> COMPLETED -> ARCHIVED
                                           -> FAILED    -> ARCHIVED
```

## Quick Start

### Installation

```bash
# Clone and install with uv
git clone https://github.com/rostykob/guildmaster-ai.git
cd guildmaster-ai

# Core dependencies + a specific LLM provider
uv sync --extra openrouter        # OpenRouter (default)
uv sync --extra anthropic         # Anthropic direct
uv sync --extra openai            # OpenAI
uv sync --extra google            # Google Gemini
uv sync --extra azure             # Azure OpenAI
uv sync --extra bedrock           # AWS Bedrock

# All providers + dev tools
uv sync --all-extras
```

### Supported Providers

| Provider string | Class | Package extra | API key env var |
|---|---|---|---|
| `openrouter` | `OpenRouterProvider` | `openrouter` | `OPENROUTER_API_KEY` |
| `anthropic` | `AnthropicProvider` | `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OpenAIProvider` | `openai` | `OPENAI_API_KEY` |
| `google` / `gemini` | `GoogleProvider` | `google` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `azure` | `AzureOpenAIProvider` | `azure` | `AZURE_OPENAI_API_KEY` |
| `bedrock` | `BedrockProvider` | `bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |

### Usage

```python
import asyncio
from guildmaster_ai import GuildBuilder
from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer

# OpenRouter (reads OPENROUTER_API_KEY from env automatically)
guild = (
    GuildBuilder()
    .with_llm_provider("openrouter")
    .register_adventurer(GeneralAdventurer)
    .build()
)

result = asyncio.run(guild.post_quest("Summarise the key differences between Python and Rust"))
print(result.summary)
```

### Switching Providers

```python
# Anthropic — reads ANTHROPIC_API_KEY from env
guild = GuildBuilder().with_llm_provider("anthropic").register_adventurer(GeneralAdventurer).build()

# OpenAI — reads OPENAI_API_KEY from env
guild = GuildBuilder().with_llm_provider("openai", model="gpt-4o").register_adventurer(GeneralAdventurer).build()

# Google Gemini — reads GEMINI_API_KEY or GOOGLE_API_KEY from env
guild = GuildBuilder().with_llm_provider("google", model="gemini-2.0-flash").register_adventurer(GeneralAdventurer).build()

# Azure OpenAI — reads AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT from env
guild = GuildBuilder().with_llm_provider("azure", model="gpt-4o").register_adventurer(GeneralAdventurer).build()

# AWS Bedrock — reads AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION from env
guild = GuildBuilder().with_llm_provider(
    "bedrock", model="anthropic.claude-3-5-sonnet-20241022-v2:0"
).register_adventurer(GeneralAdventurer).build()
```

### With the Guard enabled

```python
guild = (
    GuildBuilder()
    .with_llm_provider("openrouter")
    .register_adventurer(GeneralAdventurer)
    .with_guard()
    .build()
)
```

### Equipping Weapons (Tools)

```python
from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer
from guildmaster_ai.weapons.file_read import FileReadWeapon
from guildmaster_ai.weapons.web_search import WebSearchWeapon

adventurer = GeneralAdventurer()
adventurer.grant_talents(["research"])
adventurer.equip_weapon(FileReadWeapon())
adventurer.equip_weapon(WebSearchWeapon())

guild = (
    GuildBuilder()
    .with_llm_provider("openrouter")
    .register_adventurer(adventurer)
    .build()
)
```

### Configuration

Provider API keys are read from their **official env var names** — no renaming needed if you already have them set.  Framework settings use the `GUILD_` prefix.

```bash
# --- Provider credentials (official names) ---
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...            # or GOOGLE_API_KEY
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# --- Framework settings (GUILD_ prefix) ---
GUILD_LLM_PROVIDER=openrouter
GUILD_LLM_DEFAULT_MODEL=anthropic/claude-sonnet-4-20250514
GUILD_LLM_TEMPERATURE=0.7
GUILD_LLM_MAX_TOKENS=4096
GUILD_MAX_QUEST_RETRIES=3
GUILD_MAX_CLARIFICATION_ROUNDS=3
# Azure only
GUILD_AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

## Project Structure

```
guildmaster_ai/
  core/           # Domain models — Quest, QuestBoard, Party, Messages, Exceptions
  adventurers/    # All agent implementations
    base_adventurer.py        # Abstract base with LLM call loop & tool dispatch
    general_adventurer.py     # Versatile general-purpose adventurer
    guildmaster.py            # Orchestration brain
    receptionist.py           # User-facing intake agent
    librarian.py              # Memory & archival agent
    guard.py                  # Optional LLM-as-judge
  weapons/        # Tool abstractions and implementations
    base_weapon.py            # BaseWeapon ABC
    file_read.py              # Read files from the filesystem
    web_search.py             # Web search (stub — wire up your own backend)
  armor/          # Guardrail abstractions and implementations
    base_armor.py             # BaseArmor ABC with pre/post hooks
    content_filter.py         # Blocks messages matching forbidden patterns
    rate_limiter.py           # Sliding-window request rate limiter
  memory/         # Storage backends
    sqlite_store.py           # aiosqlite — quest metadata & history
    chroma_store.py           # ChromaDB — vector embeddings for retrieval
  llm/            # LLM provider abstraction
    base_provider.py          # BaseLLMProvider ABC, LLMMessage, LLMResponse
    openrouter.py             # OpenRouterProvider  (extra: openrouter)
    anthropic.py              # AnthropicProvider   (extra: anthropic)
    openai.py                 # OpenAIProvider      (extra: openai)
    google.py                 # GoogleProvider      (extra: google)
    azure.py                  # AzureOpenAIProvider (extra: azure)
    bedrock.py                # BedrockProvider     (extra: bedrock)
  sdk/            # SDK entry points
    guild.py                  # Guild runtime — full quest lifecycle orchestration
    builder.py                # GuildBuilder — fluent configuration API
  config/         # Settings management
    settings.py               # GuildSettings via pydantic-settings (GUILD_ prefix)
  cli/            # CLI layer (planned)
```

## Quest Ranks

Quests are ranked by difficulty: `F -> E -> D -> C -> B -> A -> S`

Higher-ranked quests require more capable adventurers and may need a full party.

## Development

```bash
# Install all extras (all providers + dev tools)
uv sync --all-extras

# Run tests
pytest

# Run a specific test file
pytest tests/test_quest.py -v

# Lint
ruff check .
ruff check --fix .

# Type check
mypy guildmaster_ai
```

## Extending the Framework

### New Adventurer

Subclass `BaseAdventurer` and implement `talents`, `system_prompt`, and `execute()`. See `AGENTS.md` for a complete example.

### New Weapon

```python
from guildmaster_ai.weapons.base_weapon import BaseWeapon

class MyWeapon(BaseWeapon):
    @property
    def name(self) -> str:
        return "my_weapon"

    @property
    def description(self) -> str:
        return "Does something useful."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "The input value"}},
            "required": ["input"],
        }

    async def execute(self, **kwargs) -> str:
        return f"Result for: {kwargs['input']}"
```

### New Armor

```python
from guildmaster_ai.armor.base_armor import BaseArmor, ArmorResult

class MyArmor(BaseArmor):
    @property
    def name(self) -> str:
        return "my_armor"

    async def pre_process(self, messages):
        # inspect / modify messages before LLM call
        return ArmorResult(verdict="pass")

    async def post_process(self, response):
        # inspect / modify LLM response
        return ArmorResult(verdict="pass")
```

### New LLM Provider

Subclass `BaseLLMProvider` and implement `complete()` and `stream()`.

## License

See [LICENSE](LICENSE) for details.
