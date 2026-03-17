from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from guildmaster_ai.weapons.base_weapon import BaseWeapon


class FileReadInput(BaseModel):
    """Input schema for the file read weapon."""

    path: str = Field(description="Path to the file to read")


class FileReadWeapon(BaseWeapon):
    """Weapon that reads a file from the filesystem."""

    name: str = "file_read"
    description: str = "Read a file from the filesystem"
    args_schema: type[BaseModel] = FileReadInput  # type: ignore[assignment]

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        path_str: str = kwargs["path"]
        file_path = Path(path_str)
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return {"path": path_str, "error": f"File not found: {path_str}"}
        return {"path": path_str, "content": content}
