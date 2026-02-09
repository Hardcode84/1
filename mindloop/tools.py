"""Agent tool definitions and execution."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Tool definitions in OpenAI function-calling format.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List files and directories in the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def _ls(path: str) -> str:
    """List directory contents."""
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if not p.is_dir():
        return f"Error: {path} is not a directory."
    entries = sorted(p.iterdir())
    lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
    return "\n".join(lines) if lines else "(empty directory)"


def _read(path: str) -> str:
    """Read file contents."""
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if not p.is_file():
        return f"Error: {path} is not a file."
    try:
        return p.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"


# Registry mapping tool names to implementations.
_REGISTRY: dict[str, Callable[..., str]] = {
    "ls": _ls,
    "read": _read,
}


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool by name with JSON-encoded arguments. Returns the result string."""
    func = _REGISTRY.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'."
    kwargs: dict[str, Any] = json.loads(arguments)
    result: str = func(**kwargs)
    return result
