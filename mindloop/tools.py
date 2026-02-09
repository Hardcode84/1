"""Agent tool definitions and execution."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Param:
    name: str
    description: str
    type: str = "string"
    required: bool = True


@dataclass
class ToolDef:
    name: str
    description: str
    params: list[Param] = field(default_factory=list)
    func: Callable[..., str] | None = None

    def to_api(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        properties = {
            p.name: {"type": p.type, "description": p.description} for p in self.params
        }
        required = [p.name for p in self.params if p.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def add(
        self,
        name: str,
        description: str,
        params: list[Param],
        func: Callable[..., str],
    ) -> None:
        """Register a tool with its definition and implementation."""
        self._tools[name] = ToolDef(
            name=name, description=description, params=params, func=func
        )

    def definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions in OpenAI API format."""
        return [tool.to_api() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str) -> str:
        """Execute a tool by name with JSON-encoded arguments."""
        tool = self._tools.get(name)
        if tool is None or tool.func is None:
            return f"Error: unknown tool '{name}'."
        try:
            kwargs: dict[str, Any] = json.loads(arguments)
            return tool.func(**kwargs)
        except Exception as e:
            return f"Error: {e}"


# --- Path sanitization ---

_work_dir: Path = Path.cwd().resolve()


class ToolError(Exception):
    """Raised when a tool encounters an expected error."""


def _sanitize_path(path: str) -> Path:
    """Resolve path and verify it stays within the working directory."""
    resolved = (_work_dir / path).resolve()
    if not str(resolved).startswith(str(_work_dir)):
        raise ToolError(f"{path} is outside the working directory.")
    return resolved


# --- Built-in tool implementations ---


def _ls(path: str) -> str:
    """List directory contents."""
    p = _sanitize_path(path)
    if not p.exists():
        raise ToolError(f"{path} does not exist.")
    if not p.is_dir():
        raise ToolError(f"{path} is not a directory.")
    entries = sorted(p.iterdir())
    lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
    return "\n".join(lines) if lines else "(empty directory)"


def _read(path: str) -> str:
    """Read file contents."""
    p = _sanitize_path(path)
    if not p.exists():
        raise ToolError(f"{path} does not exist.")
    if not p.is_file():
        raise ToolError(f"{path} is not a file.")
    return p.read_text()


# --- Default registry with built-in tools ---

default_registry = ToolRegistry()
default_registry.add(
    name="ls",
    description="List files and directories in the given path.",
    params=[Param(name="path", description="Directory path to list.")],
    func=_ls,
)
default_registry.add(
    name="read",
    description="Read the contents of a file.",
    params=[Param(name="path", description="Path to the file to read.")],
    func=_read,
)
