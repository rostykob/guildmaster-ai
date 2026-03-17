# AGENTS.md — Guildmaster-AI Agent Architecture

## Agent Hierarchy

### Guildmaster (Orchestrator)
- **Role**: Orchestration brain. Does NOT execute tasks directly.
- **Responsibilities**: Validate quest feasibility, match quests to adventurers by talent/rank, assemble parties, verify quest outcomes, trigger replanning on failure.
- **Key methods**: `check_feasibility()`, `match_adventurers()`, `assign_quest()`, `verify_result()`
- **File**: `guildmaster_ai/adventurers/guildmaster.py`

### Receptionist (Intake)
- **Role**: User-facing intake agent.
- **Responsibilities**: Transform vague user requests into well-formed Quest specs via clarification dialogue, consult Guildmaster for feasibility, present results back to user.
- **Key methods**: `intake()`, `present_result()`
- **File**: `guildmaster_ai/adventurers/receptionist.py`

### Librarian (Memory)
- **Role**: Memory and archival agent.
- **Responsibilities**: Archive quest results with structured summaries and lessons-learned entries, store embeddings for retrieval, serve historical context to other agents.
- **Key methods**: `archive()`, `analyze_quest()`, `query_observations()`
- **File**: `guildmaster_ai/adventurers/librarian.py`

### Guard (Judge)
- **Role**: Optional LLM-as-judge.
- **Responsibilities**: Evaluate agent outputs for correctness, safety, and policy compliance at configurable checkpoints.
- **Verdict types**: `pass`, `warn`, `block`
- **Key methods**: `evaluate()`
- **File**: `guildmaster_ai/adventurers/guard.py`

### BaseAdventurer (Abstract Base)
- **Role**: Abstract base for all task-executing agents.
- **Capabilities**: Equip weapons (tools), wear armor (guardrails), declare talents, call LLM, handle tool dispatch.
- **Key methods**: `execute()`, `_call_llm()`, `equip_weapon()`, `wear_armor()`, `grant_talents()`
- **File**: `guildmaster_ai/adventurers/base_adventurer.py`

### Concrete Adventurers

#### GeneralAdventurer
- **Talents**: `general`, `reasoning`
- **Role**: Versatile adventurer for general tasks; runs a tool-call loop (max 10 iterations)
- **File**: `guildmaster_ai/adventurers/general_adventurer.py`

## Communication Protocol

Agents communicate via **structured Pydantic message objects** defined in `guildmaster_ai/core/messages.py`:

- `QuestDraft` — Receptionist -> Guildmaster
- `QuestClarificationRequest/Response` — Receptionist <-> User
- `QuestFeasibilityReport` — Guildmaster -> Receptionist
- `QuestResult` — Adventurer -> Guildmaster -> User
- `QuestObservation` — Librarian -> (stored / queried)
- `GuardVerdict` — Guard -> Guildmaster
- `AdventurerProfile` — Guildmaster roster management

## Quest Execution Flow

1. **Receptionist** receives user request, creates `QuestDraft`
2. **Guildmaster** checks feasibility against roster
3. Quest posted to **QuestBoard**
4. **Guildmaster** assigns adventurer(s), forms **Party** if needed
5. **Adventurer(s)** execute quest using weapons, respecting armor guardrails
6. **Guard** (optional) evaluates output
7. **Guildmaster** verifies result against acceptance criteria
8. **Librarian** archives quest and produces lessons-learned
9. Result returned to user

## Talent System

Talents are free-form strings that connect quests to suitable adventurers:

- Every adventurer exposes a `talents` property returning a list of strings.
- The `Guildmaster` infers additional talents from keyword matching on the adventurer's `system_prompt` and from the names/descriptions of equipped weapons.
- Every adventurer implicitly receives the `"general"` talent.
- Quests can specify `required_talents`; only adventurers possessing all required talents are eligible.

## Weapon (Tool) Dispatch

`BaseAdventurer` converts equipped `BaseWeapon` instances to OpenAI-compatible tool specs via `weapon.to_tool_spec()`. When the LLM emits a `tool_calls` response, `_handle_tool_call()` dispatches to the matching weapon by name and appends the result as a `tool` role message.

## Armor (Guardrail) Hooks

`BaseArmor` subclasses are applied in order:

- `pre_process(messages)` — runs before the LLM call; can block or modify the input.
- `post_process(response)` — runs after the LLM call; can block or modify the output.

A `block` verdict stops execution and raises an exception; a `warn` verdict logs a warning but continues.

## Extending with New Agents

To create a new adventurer type:

```python
from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.core.messages import QuestResult

class MyAdventurer(BaseAdventurer):
    @property
    def talents(self) -> list[str]:
        return ["my_domain", "specific_skill"]

    @property
    def system_prompt(self) -> str:
        return "You are an adventurer specialized in..."

    async def execute(self, quest: Quest) -> QuestResult:
        self._reset_conversation()
        self._add_user_message(quest.description)
        response = await self._call_llm()
        return QuestResult(
            sender=self.name or self.id,
            quest_id=quest.id,
            success=True,
            summary=response.content,
        )
```

Then register it:

```python
from guildmaster_ai import GuildBuilder

guild = (
    GuildBuilder()
    .with_llm_provider("openrouter", api_key="sk-or-...")
    .register_adventurer(MyAdventurer)
    .build()
)
```
