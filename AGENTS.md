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
- **Key methods**: `archive()`, `query()`
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
- **Key methods**: `execute()`, `_call_llm()`, `equip_weapon()`, `wear_armor()`
- **File**: `guildmaster_ai/adventurers/base_adventurer.py`

### Concrete Adventurers

#### GeneralAdventurer
- **Talents**: `general`, `reasoning`
- **Role**: Versatile adventurer for general tasks
- **File**: `guildmaster_ai/adventurers/general_adventurer.py`

## Communication Protocol

Agents communicate via **structured Pydantic message objects** defined in `guildmaster_ai/core/messages.py`:

- `QuestDraft` — Receptionist -> Guildmaster
- `QuestClarificationRequest/Response` — Receptionist <-> User
- `QuestFeasibilityReport` — Guildmaster -> Receptionist
- `QuestResult` — Adventurer -> Guildmaster -> User
- `GuardVerdict` — Guard -> Guildmaster

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

## Extending with New Agents

To create a new adventurer type:

```python
from langchain_core.messages import HumanMessage

from guildmaster_ai.adventurers.base_adventurer import BaseAdventurer
from guildmaster_ai.core.quest import Quest
from guildmaster_ai.core.messages import QuestResult

class MyAdventurer(BaseAdventurer):
    @property
    def system_prompt(self) -> str:
        return "You are an adventurer specialized in..."

    async def execute(self, quest: Quest) -> QuestResult:
        self._reset_conversation()
        self._conversation.append(HumanMessage(content=quest.description))
        response = await self._call_llm()
        return QuestResult(
            sender=self.name,
            quest_id=quest.id,
            success=True,
            summary=response.content,
            data={},
        )
```

Then register it:

```python
guild = (
    GuildBuilder()
    .with_llm_provider("openrouter", api_key="...")
    .register_adventurer(MyAdventurer)
    .build()
)
```
