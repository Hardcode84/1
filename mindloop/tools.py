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

    def copy(self) -> "ToolRegistry":
        """Return a shallow copy of this registry."""
        clone = ToolRegistry()
        clone._tools = dict(self._tools)
        return clone

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


_BINARY_CHECK_SIZE = 8192
_MAX_LINES = 100
_MAX_LINE_LENGTH = 200


def _is_binary(p: Path) -> bool:
    """Check if a file is binary by looking for null bytes in the first chunk."""
    with p.open("rb") as f:
        return b"\x00" in f.read(_BINARY_CHECK_SIZE)


def _truncate_line(line: str, max_length: int) -> str:
    """Truncate a line if it exceeds max_length."""
    if len(line) <= max_length:
        return line
    truncated = len(line) - max_length
    # Strip trailing newline before appending indicator, then restore it.
    nl = "\n" if line.endswith("\n") else ""
    return f"{line[:max_length]}... ({truncated} chars truncated){nl}"


def _read(
    path: str,
    offset: int = 0,
    limit: int = _MAX_LINES,
    max_line_length: int = _MAX_LINE_LENGTH,
) -> str:
    """Read file contents with optional line offset, limit, and line truncation."""
    p = _sanitize_path(path)
    if not p.exists():
        raise ToolError(f"{path} does not exist.")
    if not p.is_file():
        raise ToolError(f"{path} is not a file.")
    if _is_binary(p):
        return f"{path} is a binary file."
    all_lines = p.read_text().splitlines(keepends=True)
    total = len(all_lines)
    selected = [
        _truncate_line(line, max_line_length)
        for line in all_lines[offset : offset + limit]
    ]
    result = "".join(selected)
    remaining = total - offset - len(selected)
    if remaining > 0:
        result += f"\n... ({remaining} lines remaining)"
    return result


# --- Default registry with built-in tools ---

default_registry = ToolRegistry()
default_registry.add(
    name="ls",
    description="List files and directories. Paths are relative to the working directory.",
    params=[
        Param(name="path", description="Relative path within the working directory.")
    ],
    func=_ls,
)
default_registry.add(
    name="read",
    description="Read file contents. Paths are relative to the working directory.",
    params=[
        Param(name="path", description="Relative path within the working directory."),
        Param(
            name="offset",
            description="Line number to start from (0-based). Default: 0.",
            type="integer",
            required=False,
        ),
        Param(
            name="limit",
            description=f"Maximum number of lines to return. Default: {_MAX_LINES}.",
            type="integer",
            required=False,
        ),
        Param(
            name="max_line_length",
            description=f"Maximum characters per line before truncation. Default: {_MAX_LINE_LENGTH}.",
            type="integer",
            required=False,
        ),
    ],
    func=_read,
)
