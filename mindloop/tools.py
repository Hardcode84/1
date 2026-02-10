"""Agent tool definitions and execution."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
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
        self.stats: dict[Any, Any] = {}

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


def _track_file(reg: "ToolRegistry", tool: str, path: str) -> None:
    """Record a file access in registry stats."""
    reg.stats.setdefault(tool, {}).setdefault("files_accessed", []).append(path)


# --- Built-in tool implementations ---


def _ls(reg: "ToolRegistry", path: str) -> str:
    """List directory contents."""
    _track_file(reg, "ls", path)
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


def _edit(
    reg: "ToolRegistry",
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace exact string occurrences in a file."""
    _track_file(reg, "edit", path)
    p = _sanitize_path(path)
    if not p.exists():
        raise ToolError(f"{path} does not exist.")
    if not p.is_file():
        raise ToolError(f"{path} is not a file.")
    if _is_binary(p):
        raise ToolError(f"{path} is a binary file.")
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical.")
    content = p.read_text()
    count = content.count(old_string)
    if count == 0:
        raise ToolError("old_string not found in file.")
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string matches {count} locations. "
            "Provide more context to make it unique, or set replace_all=true."
        )
    if replace_all:
        result = content.replace(old_string, new_string)
    else:
        result = content.replace(old_string, new_string, 1)
    p.write_text(result)
    return f"Replaced {count if replace_all else 1} occurrence(s) in {path}."


def _write(reg: "ToolRegistry", path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    _track_file(reg, "write", path)
    p = _sanitize_path(path)
    if p.exists() and not p.is_file():
        raise ToolError(f"{path} is not a file.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    lines = content.count("\n") + (0 if content.endswith("\n") else 1)
    return f"Wrote {lines} lines to {path}."


def _read(
    reg: "ToolRegistry",
    path: str,
    line_offset: int = 0,
    line_limit: int = _MAX_LINES,
    max_line_length: int = _MAX_LINE_LENGTH,
) -> str:
    """Read file contents with optional line offset, limit, and line truncation."""
    _track_file(reg, "read", path)
    p = _sanitize_path(path)
    if not p.exists():
        raise ToolError(f"{path} does not exist.")
    if not p.is_file():
        raise ToolError(f"{path} is not a file.")
    if _is_binary(p):
        return f"{path} is a binary file."
    all_lines = p.read_text().splitlines(keepends=True)
    total = len(all_lines)
    start = line_offset if line_offset >= 0 else max(0, total + line_offset)
    selected = [
        _truncate_line(line, max_line_length)
        for line in all_lines[start : start + line_limit]
    ]
    result = "".join(selected)
    remaining = total - start - len(selected)
    if remaining > 0:
        result += f"\n... ({remaining} lines remaining)"
    return result


# --- Default registry factory ---


def create_default_registry() -> ToolRegistry:
    """Create a fresh registry populated with the built-in tools."""
    reg = ToolRegistry()
    reg.add(
        name="ls",
        description="List files and directories. Paths are relative to the working directory.",
        params=[
            Param(
                name="path", description="Relative path within the working directory."
            )
        ],
        func=partial(_ls, reg),
    )
    reg.add(
        name="edit",
        description=(
            "Replace exact string in a file. "
            "old_string must be unique unless replace_all is true. "
            "Read the file first to get the exact text. "
            "Paths are relative to the working directory."
        ),
        params=[
            Param(
                name="path", description="Relative path within the working directory."
            ),
            Param(name="old_string", description="Exact text to find in the file."),
            Param(name="new_string", description="Text to replace it with."),
            Param(
                name="replace_all",
                description="Replace all occurrences. Default: false.",
                type="boolean",
                required=False,
            ),
        ],
        func=partial(_edit, reg),
    )
    reg.add(
        name="write",
        description=(
            "Create or overwrite a file. Creates parent directories if needed. "
            "Prefer edit for modifying existing files. "
            "Paths are relative to the working directory."
        ),
        params=[
            Param(
                name="path", description="Relative path within the working directory."
            ),
            Param(name="content", description="Full file content to write."),
        ],
        func=partial(_write, reg),
    )
    reg.add(
        name="read",
        description="Read file contents. Paths are relative to the working directory.",
        params=[
            Param(
                name="path", description="Relative path within the working directory."
            ),
            Param(
                name="line_offset",
                description="Line number to start from (0-based). Negative counts from end. Default: 0.",
                type="integer",
                required=False,
            ),
            Param(
                name="line_limit",
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
        func=partial(_read, reg),
    )
    return reg
