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

# Core dependencies only
uv sync

# Include the OpenRouter SDK and dev tools
uv sync --all-extras
```

### Usage

```python
import asyncio
from guildmaster_ai import GuildBuilder
from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer

guild = (
    GuildBuilder()
    .with_llm_provider("openrouter", api_key="sk-or-your-key")
    .register_adventurer(GeneralAdventurer)
    .build()
)

result = asyncio.run(guild.post_quest("Summarise the key differences between Python and Rust"))
print(result.summary)
```

### With the Guard enabled

```python
guild = (
    GuildBuilder()
    .with_llm_provider("openrouter", api_key="sk-or-your-key")
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
    .with_llm_provider("openrouter", api_key="sk-or-your-key")
    .register_adventurer(adventurer)
    .build()
)
```

### Configuration

Set environment variables (prefix `GUILD_`) or use a `.env` file:

```bash
GUILD_LLM_API_KEY=sk-or-your-openrouter-key
GUILD_LLM_DEFAULT_MODEL=anthropic/claude-sonnet-4-20250514
GUILD_LLM_TEMPERATURE=0.7
GUILD_MAX_QUEST_RETRIES=3
GUILD_MAX_CLARIFICATION_ROUNDS=3
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
    openrouter.py             # OpenRouterProvider using the official openrouter SDK
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
# Install all extras (includes openrouter SDK + dev tools)
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
