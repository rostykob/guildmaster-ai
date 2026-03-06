from __future__ import annotations

import json
from typing import Any

import aiosqlite

from guildmaster_ai.core.quest import Quest, QuestHistoryEntry, QuestRank, QuestStatus


class SQLiteStore:
    """Async SQLite storage for quests and history."""

    def __init__(self, db_path: str = "guildmaster.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database and create tables if they don't exist."""
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
            """
        )
        await self._db.commit()

    async def save_quest(self, quest: Quest) -> None:
        """Upsert a quest into the database."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        data = json.dumps({
            "required_talents": quest.required_talents,
            "acceptance_criteria": quest.acceptance_criteria,
            "assigned_party_id": quest.assigned_party_id,
            "result": quest.result,
        })

        await self._db.execute(
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
        await self._db.commit()

    async def get_quest(self, quest_id: str) -> Quest | None:
        """Retrieve a quest by ID, or return None if not found."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        async with self._db.execute(
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

    async def list_quests(self, status: QuestStatus | None = None) -> list[Quest]:
        """List quests, optionally filtered by status."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        if status is not None:
            query = "SELECT id FROM quests WHERE status = ? ORDER BY created_at DESC"
            params: tuple[Any, ...] = (status.value,)
        else:
            query = "SELECT id FROM quests ORDER BY created_at DESC"
            params = ()

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        quests: list[Quest] = []
        for row in rows:
            quest = await self.get_quest(row[0])
            if quest is not None:
                quests.append(quest)
        return quests

    async def save_history_entry(
        self, quest_id: str, entry: QuestHistoryEntry
    ) -> None:
        """Save a single history entry for a quest."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        await self._db.execute(
            """
            INSERT INTO quest_history (quest_id, actor, event_type, payload, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                quest_id,
                entry.actor,
                entry.event_type,
                json.dumps(entry.payload),
                entry.timestamp.isoformat(),
            ),
        )
        await self._db.commit()

    async def get_history(self, quest_id: str) -> list[QuestHistoryEntry]:
        """Retrieve all history entries for a quest."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        async with self._db.execute(
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

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
