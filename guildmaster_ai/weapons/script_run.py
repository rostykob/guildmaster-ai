"""Weapon that executes a script in a subprocess."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from guildmaster_ai.weapons.base_weapon import BaseWeapon


class ScriptRunInput(BaseModel):
    """Input schema for the script run weapon."""

    path: str = Field(description="Path to the script to execute")
    args: str = Field(default="", description="Space-separated arguments to pass to the script")


class ScriptRunWeapon(BaseWeapon):
    """Weapon that executes a script file and returns its output.

    Runs the script as a subprocess using ``python`` for ``.py`` files
    and ``bash`` for ``.sh`` files.  Captures both stdout and stderr.
    """

    name: str = "run_script"
    description: str = (
        "Execute a script file and return its output. "
        "Provide the path to the script and optional arguments."
    )
    args_schema: type[BaseModel] = ScriptRunInput

    timeout_seconds: int = 30
    """Maximum seconds to wait for the script to complete."""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        path: str = kwargs["path"]
        args_str: str = kwargs.get("args", "")

        if path.endswith(".py"):
            cmd = ["python", path]
        elif path.endswith(".sh"):
            cmd = ["bash", path]
        else:
            cmd = [path]

        if args_str:
            cmd.extend(args_str.split())

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            return {"path": path, "error": f"Script not found: {path}"}
        except TimeoutError:
            proc.kill()
            return {"path": path, "error": f"Script timed out after {self.timeout_seconds}s"}

        result: dict[str, Any] = {
            "path": path,
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace").strip(),
        }
        stderr_text = stderr.decode(errors="replace").strip()
        if stderr_text:
            result["stderr"] = stderr_text
        return result
