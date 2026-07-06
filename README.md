# Guildmaster-AI

An agentic framework with a fantasy guild metaphor.

Guildmaster-AI is an open-source Python framework for orchestrating multiple LLM agents to collaboratively plan, execute, verify, and archive complex tasks — called **Quests**.

## Concepts

| Concept | Description |
|---|---|
| **Guild** | The runtime — contains a Guildmaster, Receptionist, Librarian, and a roster of Adventurers |
| **Quest** | A structured unit of work with a title, description, rank, and acceptance criteria |
| **Adventurer** | An agent that executes quests using Weapons (tools) and wearing Armor (guardrails) |
| **Hero** | An adventurer built on a deep agent — supports Scrolls and leads parties by delegating to recruited members as subagents |
| **Guildmaster** | The orchestration brain — triages quests (rank + candidates in one LLM call), forms parties, verifies results |
| **Receptionist** | User-facing intake agent — refines requests, asks for missing weapon inputs, answers status queries |
| **Librarian** | Memory agent — archives quest results and maintains the guild chronicle |
| **Guard** | Optional LLM-as-judge — evaluates outputs for correctness and policy compliance |
| **Weapon** | A tool abstraction (e.g. web search, file read, code execution) |
| **Armor** | A guardrail abstraction (e.g. content filter, rate limiter, PII scrubber) |
| **Scroll** | A SKILL.md-based skill package heroes can pick up (agentskills.io spec) |
| **Party** | A temporary coalition of adventurers for complex quests |
| **Quest Board** | Priority queue of posted quests awaiting assignment |

## Quest Lifecycle

```
User Request -> Receptionist (refine) -> Guildmaster (triage: rank + candidates)
  -> route by rank:
       F-C  single adventurer executes
       B    manual party (decompose into subtasks, concurrent waves, retries)
       A/S  hero-led (candidates recruited as subagents, hero plans internally)
  -> Guard (optional) -> Guildmaster (verify) -> Librarian (archive) -> Result
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

# Core + dev tools
uv sync --all-extras
```

### Supported Providers

| Provider string | Class | Package extra | API key env var |
|---|---|---|---|
| `openrouter` | `ChatOpenRouter` | `openrouter` | `OPENROUTER_API_KEY` |
| `anthropic` | `ChatAnthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `ChatOpenAIProvider` | `openai` | `OPENAI_API_KEY` |
| `google` / `gemini` | `ChatGoogle` | `google` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `azure` | `ChatAzureOpenAI` | `azure` | `AZURE_OPENAI_API_KEY` |
| `bedrock` | `ChatBedrock` | `bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |

All providers are thin wrappers around their respective LangChain chat model classes.

### Usage

```python
import asyncio
from guildmaster_ai.sdk.builder import GuildBuilder
from guildmaster_ai.adventurers.general_adventurer import GeneralAdventurer

# OpenRouter (reads OPENROUTER_API_KEY from env automatically)
guild = (
    GuildBuilder()
    .with_llm_provider("openrouter")
    .register_adventurer(GeneralAdventurer)
    .build()
)

# run_quest submits and waits for the result
result = asyncio.run(guild.run_quest("Summarise the key differences between Python and Rust"))
print(result.summary)
```

### Background execution

Quests run in a background worker (up to `GUILD_MAX_CONCURRENT_QUESTS` at a
time). `post_quest` returns immediately with a ticket; `run_quest` is the
submit-and-wait convenience:

```python
async def main():
    async with guild:  # close() drains in-flight quests on exit
        ticket = await guild.post_quest("Research topic A")   # returns QuestTicket
        report = guild.check_quest_status(ticket.quest_id)    # point-in-time status
        result = await guild.wait_for_quest(ticket.quest_id)  # await completion
```

Subscribe to completions instead of polling (sync or async handlers; handler
errors are logged, never raised):

```python
async def notify(quest, result):
    print(f"{quest.title}: success={result.success}")

unsubscribe = guild.on_quest_complete(notify)
# or fluently at build time: GuildBuilder().with_quest_listener(notify)...
```

Because the SDK is fully async, it embeds directly in an async framework such
as FastAPI — create the `Guild` at app startup, call `post_quest` in request
handlers, push updates from an `on_quest_complete` handler (webhooks/SSE),
and `await guild.close()` on shutdown.

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

### Using Any LangChain Chat Model

You can pass any LangChain `BaseChatModel` directly:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", api_key="sk-...")
guild = GuildBuilder().with_llm_provider(llm).register_adventurer(GeneralAdventurer).build()
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

### Custom Adventurers with Descriptions

Give adventurers a `description` — the guildmaster uses it (alongside talents
and weapons) when triaging quests and assigning subtasks:

```python
from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer

class GrammarCoach(BaseAdventurer):
    description = "A grammar specialist: explains rules and corrects sentences."
    system_prompt = "You are a friendly English grammar coach..."
```

### Scoping a Guild with a Charter

The charter is owner-provided domain context, appended to the guildmaster's
triage/planning/verification prompts and the receptionist's intake prompt:

```python
guild = (
    GuildBuilder()
    .with_llm_provider("openrouter")
    .with_guild_charter("This guild is an English exam tutoring service...")
    .register_adventurer(GrammarCoach)
    .build()
)
```

### Heroes and Scrolls

Heroes are deep-agent adventurers that can pick up scrolls (SKILL.md skill
packages) and lead parties — on rank A/S quests the other triage candidates
are recruited as their subagents automatically:

```python
from guildmaster_ai.adventurers.general_hero import GeneralHero

hero = GeneralHero(name="Leader")
hero.pick_scroll("research")  # resolved from with_scrolls_dir(...) catalog

guild = (
    GuildBuilder()
    .with_llm_provider("openrouter")
    .with_scrolls_dir("guildmaster_ai/scrolls/skills")
    .register_adventurer(hero)
    .build()
)
```

See `examples/` for runnable notebooks and `examples/english_tutor/` for a
complete sub-project (custom adventurers, weapons, a scroll, and a charter).

### Configuration

Provider API keys are read from their **official env var names** — no renaming needed if you already have them set.  Framework settings use the `GUILD_` prefix.

```bash
# --- Provider credentials (official names) ---
OPENROUTER_API_KEY=sk-or-...

# --- Framework settings (GUILD_ prefix) ---
GUILD_LLM_PROVIDER=openrouter
GUILD_LLM_DEFAULT_MODEL=anthropic/claude-sonnet-4-20250514
GUILD_LLM_TEMPERATURE=0.7
GUILD_LLM_MAX_TOKENS=4096
GUILD_MAX_QUEST_RETRIES=3
GUILD_MAX_CLARIFICATION_ROUNDS=3
GUILD_MAX_CONCURRENT_QUESTS=3       # background worker concurrency
GUILD_GUILD_HOME=./.guildmaster     # SQLite + Chroma storage location
GUILD_REFINE_REQUESTS=true          # false = skip intake refinement (one less LLM call)
GUILD_VERIFY_MIN_RANK=F             # D = skip result verification for trivial F/E quests
GUILD_GUILD_CHARTER=                # owner-provided domain context for coordinating agents
GUILD_ENABLE_OBSERVATIONS=false     # librarian quest analysis + chronicle (off the hot path)
```

## Project Structure

```
guildmaster_ai/
  core/           # Domain models — Quest, QuestBoard, Party, Messages, Exceptions
  adventurers/    # All agent implementations
    base_adventurer.py        # Abstract base (LangChain create_agent, weapons + armor)
    base_hero.py              # Deep-agent base with scrolls + member subagents
    general_adventurer.py     # Versatile general-purpose adventurer
    general_hero.py           # Ready-made party-leader hero
    vitality.py               # AP/HP budget middleware (tool calls / tool errors)
    guildmaster.py            # Orchestration brain (triage, planning, verification)
    receptionist.py           # User-facing intake, clarification & status agent
    librarian.py              # Memory & archival agent (chronicle)
    guard.py                  # Optional LLM-as-judge
  weapons/        # Tool abstractions (extend LangChain BaseTool)
    base_weapon.py            # BaseWeapon — extends LangChain BaseTool
    file_read.py              # Read files from the filesystem
    script_run.py             # Run scripts
    web_search.py             # Web search (stub — wire up your own backend)
  armor/          # Guardrail abstractions and implementations
    base_armor.py             # BaseArmor ABC with pre/post hooks
    content_filter.py         # Blocks messages matching forbidden patterns
    rate_limiter.py           # Sliding-window request rate limiter
  scrolls/        # SKILL.md-based skills (agentskills.io spec)
    scroll.py                 # Scroll — loads a skill-name/SKILL.md package
    catalog.py                # ScrollCatalog — name-based lookup over a directory
    skills/                   # Built-in skills (e.g. research, data-profiler)
  memory/         # Storage backends
    sqlite_store.py           # aiosqlite — quest metadata & history
    chroma_store.py           # ChromaDB — vector embeddings for retrieval
  llm/            # LangChain chat model factory & wrapper types
    types.py                  # GuildLLM, GuildResponse, guild_complete()
    base_provider.py          # create_chat_model() factory
    openrouter.py             # ChatOpenRouter (+ anthropic/openai/google/azure/bedrock)
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

Subclass `BaseAdventurer` and set a `system_prompt` (class variable or
property). Optionally set a `description` for routing. Talents are assessed
automatically by the guildmaster. See `AGENTS.md` for a complete example.

### New Hero

Subclass `BaseHero` (or use `GeneralHero`) for a deep-agent adventurer with
scroll support. Use `pick_scroll()` to load skills and `recruit()` to add
member adventurers as subagents.

### New Weapon

```python
from guildmaster_ai.weapons.base_weapon import BaseWeapon
from pydantic import BaseModel, Field

class MyWeaponInput(BaseModel):
    input: str = Field(description="The input value")

class MyWeapon(BaseWeapon):
    name: str = "my_weapon"
    description: str = "Does something useful."
    args_schema: type[BaseModel] = MyWeaponInput

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

Pass any LangChain `BaseChatModel` to `GuildBuilder.with_llm_provider()`. No custom provider class needed.

## License

See [LICENSE](LICENSE) for details.
