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
  -> Guildmaster (verify) -> Librarian (archive) -> User Result
```

## Quick Start

### Installation

```bash
# Clone and install with uv
git clone https://github.com/rostykob/guildmaster-ai.git
cd guildmaster-ai
uv sync --all-extras
```

### Usage

```python
import asyncio
from guildmaster_ai import GuildBuilder
from guildmaster_ai.agents.adventurers.general_adventurer import GeneralAdventurer

guild = (
    GuildBuilder()
    .with_llm_provider("openrouter", api_key="your-api-key")
    .register_adventurer(GeneralAdventurer)
    .build()
)

result = asyncio.run(guild.post_quest("Summarise the key differences between Python and Rust"))
print(result.summary)
```

### Configuration

Set environment variables (prefix `GUILD_`) or use a `.env` file:

```bash
GUILD_LLM_API_KEY=your-openrouter-key
GUILD_LLM_DEFAULT_MODEL=anthropic/claude-sonnet-4-20250514
GUILD_LLM_TEMPERATURE=0.7
GUILD_MAX_QUEST_RETRIES=3
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy guildmaster_ai
```

## Project Structure

```
guildmaster_ai/
  core/           # Domain models (Quest, QuestBoard, Party, Messages)
  agents/         # Agent implementations (Guildmaster, Receptionist, Adventurers)
  weapons/        # Tool abstractions and implementations
  armor/          # Guardrail abstractions and implementations
  memory/         # Storage backends (SQLite, ChromaDB)
  llm/            # LLM provider abstractions (OpenRouter)
  sdk/            # SDK entry points (Guild, GuildBuilder)
  config/         # Settings management
  cli/            # CLI layer (Phase 2)
```

## Quest Ranks

Quests are ranked by difficulty: `F -> E -> D -> C -> B -> A -> S`

Higher-ranked quests require more capable adventurers and may need a full party.

## License

See [LICENSE](LICENSE) for details.
