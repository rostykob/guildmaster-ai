from __future__ import annotations

import json
import logging
from typing import Any

import aiosqlite

from guildmaster_ai.core.quest import Quest, QuestHistoryEntry, QuestRank, QuestStatus
from guildmaster_ai.core.utils import _utcnow

logger = logging.getLogger("guildmaster.memory.sqlite")


class SQLiteStore:
    """Async SQLite storage for quests and history."""

    def __init__(self, db_path: str = "guildmaster.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        """Return the open connection, or raise if the store isn't initialized."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._db

    async def initialize(self) -> None:
        """Open the database and create tables if they don't exist."""
        logger.info("Initializing SQLite store at %s", self._db_path)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS quests (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                rank INTEGER NOT NULL,
                status TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quest_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quest_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (quest_id) REFERENCES quests(id)
            );

            CREATE TABLE IF NOT EXISTS adventurer_roster (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                talents TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS guild_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS adventurer_talents (
                config_hash TEXT PRIMARY KEY,
                class_name TEXT NOT NULL,
                talents TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quest_results (
                quest_id TEXT PRIMARY KEY,
                success INTEGER NOT NULL,
                summary TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                failure_reason TEXT,
                FOREIGN KEY (quest_id) REFERENCES quests(id)
            );

            CREATE TABLE IF NOT EXISTS quest_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quest_id TEXT NOT NULL,
                adventurer TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (quest_id) REFERENCES quests(id)
            );

            CREATE TABLE IF NOT EXISTS quest_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quest_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (quest_id) REFERENCES quests(id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                quest_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                lessons_learned TEXT NOT NULL DEFAULT '[]',
                timestamp TEXT NOT NULL
            );
            """
        )
        await self._db.commit()

    async def save_quest(self, quest: Quest) -> None:
        """Upsert a quest into the database."""
        data = json.dumps(
            {
                "required_talents": quest.required_talents,
                "acceptance_criteria": quest.acceptance_criteria,
                "assigned_party_id": quest.assigned_party_id,
                "result": quest.result,
            }
        )

        await self._conn.execute(
            """
            INSERT INTO quests (id, title, description, rank, status, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                rank = excluded.rank,
                status = excluded.status,
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (
                quest.id,
                quest.title,
                quest.description,
                quest.rank.value,
                quest.status.value,
                data,
                quest.created_at.isoformat(),
                quest.updated_at.isoformat(),
            ),
        )
        # Sync the audit trail (stage transitions + events). History is
        # append-only in memory; mirror the full trail so the DB always
        # reflects the current stage and how the quest reached it.
        await self._conn.execute(
            "DELETE FROM quest_history WHERE quest_id = ?", (quest.id,)
        )
        for entry in quest.history:
            await self._conn.execute(
                """
                INSERT INTO quest_history (quest_id, actor, event_type, payload, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    quest.id,
                    entry.actor,
                    entry.event_type,
                    json.dumps(entry.payload),
                    entry.timestamp.isoformat(),
                ),
            )
        await self._conn.commit()

    async def get_quest(self, quest_id: str) -> Quest | None:
        """Retrieve a quest by ID, or return None if not found."""
        async with self._conn.execute(
            "SELECT id, title, description, rank, status, data, created_at, updated_at"
            " FROM quests WHERE id = ?",
            (quest_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        data = json.loads(row[5])
        history = await self.get_history(quest_id)

        return Quest(
            id=row[0],
            title=row[1],
            description=row[2],
            rank=QuestRank(row[3]),
            status=QuestStatus(row[4]),
            required_talents=data.get("required_talents", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            assigned_party_id=data.get("assigned_party_id"),
            result=data.get("result"),
            history=history,
            created_at=row[6],
            updated_at=row[7],
        )

    async def get_history(self, quest_id: str) -> list[QuestHistoryEntry]:
        """Retrieve all history entries for a quest."""
        async with self._conn.execute(
            "SELECT actor, event_type, payload, timestamp"
            " FROM quest_history WHERE quest_id = ? ORDER BY timestamp",
            (quest_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            QuestHistoryEntry(
                actor=row[0],
                event_type=row[1],
                payload=json.loads(row[2]),
                timestamp=row[3],
            )
            for row in rows
        ]

    # ── Guild state ─────────────────────────────────────────────────────

    async def save_guild_state(self, key: str, value: str) -> None:
        """Save a key-value pair to the guild state table."""
        now = _utcnow().isoformat()
        await self._conn.execute(
            """
            INSERT INTO guild_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, now),
        )
        await self._conn.commit()

    async def get_guild_state(self, key: str) -> str | None:
        """Retrieve a guild state value by key."""
        async with self._conn.execute(
            "SELECT value FROM guild_state WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    # ── Adventurer talents ───────────────────────────────────────────────

    async def save_adventurer_talents(
        self, config_hash: str, class_name: str, talents: list[str]
    ) -> None:
        """Persist adventurer talents keyed by config hash."""
        now = _utcnow().isoformat()
        await self._conn.execute(
            """
            INSERT INTO adventurer_talents (config_hash, class_name, talents, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(config_hash) DO UPDATE SET
                class_name = excluded.class_name,
                talents = excluded.talents,
                updated_at = excluded.updated_at
            """,
            (config_hash, class_name, json.dumps(talents), now),
        )
        await self._conn.commit()

    async def get_adventurer_talents(self, config_hash: str) -> list[str] | None:
        """Retrieve cached talents by config hash, or None if not found."""
        async with self._conn.execute(
            "SELECT talents FROM adventurer_talents WHERE config_hash = ?",
            (config_hash,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        talents: list[str] = json.loads(row[0])
        return talents

    # ── Quest results ────────────────────────────────────────────────────

    async def save_result(self, quest_id: str, result: dict[str, Any]) -> None:
        """Persist a quest result."""
        await self._conn.execute(
            """
            INSERT INTO quest_results (quest_id, success, summary, data, failure_reason)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(quest_id) DO UPDATE SET
                success = excluded.success,
                summary = excluded.summary,
                data = excluded.data,
                failure_reason = excluded.failure_reason
            """,
            (
                quest_id,
                int(result.get("success", False)),
                result.get("summary", ""),
                json.dumps(result.get("data", {})),
                result.get("failure_reason"),
            ),
        )
        await self._conn.commit()

    async def get_result(self, quest_id: str) -> dict[str, Any] | None:
        """Retrieve a persisted quest result."""
        async with self._conn.execute(
            "SELECT success, summary, data, failure_reason FROM quest_results WHERE quest_id = ?",
            (quest_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "success": bool(row[0]),
            "summary": row[1],
            "data": json.loads(row[2]),
            "failure_reason": row[3],
        }

    # ── Conversations ────────────────────────────────────────────────────

    async def save_conversation(
        self,
        quest_id: str,
        adventurer: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Append an adventurer's message transcript for a quest.

        Each *message* is a ``{"role": ..., "content": ...}`` dict. Transcripts
        are append-only: a subtask retry records a fresh conversation rather
        than overwriting the earlier attempt.
        """
        if not messages:
            return

        now = _utcnow().isoformat()
        await self._conn.executemany(
            """
            INSERT INTO quest_conversations
                (quest_id, adventurer, seq, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    quest_id,
                    adventurer,
                    seq,
                    msg.get("role", "unknown"),
                    msg.get("content", ""),
                    now,
                )
                for seq, msg in enumerate(messages)
            ],
        )
        await self._conn.commit()

    async def get_conversation(self, quest_id: str) -> list[dict[str, Any]]:
        """Retrieve the full conversation transcript for a quest, in order."""
        async with self._conn.execute(
            "SELECT adventurer, seq, role, content, timestamp"
            " FROM quest_conversations WHERE quest_id = ?"
            " ORDER BY id, seq",
            (quest_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "adventurer": row[0],
                "seq": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
            }
            for row in rows
        ]

    # ── Findings ─────────────────────────────────────────────────────────

    async def save_finding(
        self,
        quest_id: str,
        iteration: int,
        kind: str,
        summary: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Record an intermediate finding for a quest.

        Findings capture per-iteration progress — individual subtask outcomes
        and party-leader decisions across retry rounds — so a quest's reasoning
        trail survives beyond the final :class:`QuestResult`.
        """
        await self._conn.execute(
            """
            INSERT INTO quest_findings
                (quest_id, iteration, kind, summary, data, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                quest_id,
                iteration,
                kind,
                summary,
                json.dumps(data or {}),
                _utcnow().isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_findings(self, quest_id: str) -> list[dict[str, Any]]:
        """Retrieve all findings for a quest, ordered by iteration."""
        async with self._conn.execute(
            "SELECT iteration, kind, summary, data, timestamp"
            " FROM quest_findings WHERE quest_id = ?"
            " ORDER BY iteration, id",
            (quest_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "iteration": row[0],
                "kind": row[1],
                "summary": row[2],
                "data": json.loads(row[3]),
                "timestamp": row[4],
            }
            for row in rows
        ]

    # ── Observations ─────────────────────────────────────────────────────

    async def save_observation(
        self,
        obs_id: str,
        quest_id: str,
        summary: str,
        tags: list[str],
        lessons_learned: list[str],
    ) -> None:
        """Persist a librarian observation (upsert by observation id)."""
        await self._conn.execute(
            """
            INSERT INTO observations
                (id, quest_id, summary, tags, lessons_learned, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                quest_id = excluded.quest_id,
                summary = excluded.summary,
                tags = excluded.tags,
                lessons_learned = excluded.lessons_learned,
                timestamp = excluded.timestamp
            """,
            (
                obs_id,
                quest_id,
                summary,
                json.dumps(tags),
                json.dumps(lessons_learned),
                _utcnow().isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_observations(
        self,
        quest_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve stored observations (newest first), optionally by quest."""
        if quest_id is not None:
            query = (
                "SELECT id, quest_id, summary, tags, lessons_learned, timestamp"
                " FROM observations WHERE quest_id = ?"
                " ORDER BY timestamp DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (quest_id, limit)
        else:
            query = (
                "SELECT id, quest_id, summary, tags, lessons_learned, timestamp"
                " FROM observations ORDER BY timestamp DESC LIMIT ?"
            )
            params = (limit,)

        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "quest_id": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "lessons_learned": json.loads(row[4]),
                "timestamp": row[5],
            }
            for row in rows
        ]

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.debug("SQLite store closed")
