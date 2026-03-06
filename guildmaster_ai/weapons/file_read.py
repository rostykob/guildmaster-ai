from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from guildmaster_ai.weapons.base_weapon import BaseWeapon


class FileReadWeapon(BaseWeapon):
    """Weapon that reads a file from the filesystem."""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read a file from the filesystem"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        path_str: str = kwargs["path"]
        file_path = Path(path_str)
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return {"path": path_str, "error": f"File not found: {path_str}"}
        return {"path": path_str, "content": content}
