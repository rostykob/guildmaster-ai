from __future__ import annotations

import asyncio
import logging
from typing import Any

import chromadb

logger = logging.getLogger("guildmaster.memory.chroma")


class ChromaStore:
    """ChromaDB vector store for guild knowledge retrieval."""

    def __init__(
        self,
        collection_name: str = "guild_knowledge",
        persist_directory: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    async def initialize(self) -> None:
        """Create or retrieve the ChromaDB collection."""
        logger.info("Initializing ChromaDB collection %r", self._collection_name)
        if self._persist_directory:
            settings = chromadb.Settings(
                persist_directory=str(self._persist_directory),
                is_persistent=True,
            )
            self._client = await asyncio.to_thread(chromadb.Client, settings)
        else:
            self._client = await asyncio.to_thread(chromadb.Client)
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self._collection_name,
        )

    async def _ensure(self) -> None:
        """Initialize the collection on first use if not already done."""
        if self._collection is None:
            await self.initialize()

    async def store(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add or update a document in the collection."""
        await self._ensure()
        assert self._collection is not None

        await asyncio.to_thread(
            self._collection.upsert,
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
        )

    async def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Query the collection and return matching documents."""
        await self._ensure()
        assert self._collection is not None

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where is not None:
            kwargs["where"] = where

        raw = await asyncio.to_thread(self._collection.query, **kwargs)

        results: list[dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            results.append(
                {
                    "id": doc_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return results
