"""Tests for mindloop.tools."""

from pathlib import Path
from unittest.mock import patch

from mindloop.tools import Param, ToolError, ToolRegistry, default_registry


# --- Param / ToolDef / to_api ---


def test_to_api_format() -> None:
    registry = ToolRegistry()
    registry.add(
        name="greet",
        description="Say hello.",
        params=[Param(name="name", description="Who to greet.")],
        func=lambda name: f"hi {name}",
    )
    defs = registry.definitions()
    assert len(defs) == 1
    func_def = defs[0]["function"]
    assert func_def["name"] == "greet"
    assert func_def["description"] == "Say hello."
    assert "name" in func_def["parameters"]["properties"]
    assert func_def["parameters"]["required"] == ["name"]


def test_to_api_optional_param() -> None:
    registry = ToolRegistry()
    registry.add(
        name="test",
        description="Test tool.",
        params=[
            Param(name="a", description="Required."),
            Param(name="b", description="Optional.", required=False),
        ],
        func=lambda a, b="default": f"{a} {b}",
    )
    func_def = registry.definitions()[0]["function"]
    assert func_def["parameters"]["required"] == ["a"]


# --- execute ---


def test_execute_known_tool() -> None:
    registry = ToolRegistry()
    registry.add(
        name="echo",
        description="Echo input.",
        params=[Param(name="text", description="Text to echo.")],
        func=lambda text: text,
    )
    result = registry.execute("echo", '{"text": "hello"}')
    assert result == "hello"


def test_execute_unknown_tool() -> None:
    registry = ToolRegistry()
    result = registry.execute("nonexistent", "{}")
    assert "unknown tool" in result


# --- built-in ls ---


def test_ls_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "subdir").mkdir()

    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("ls", '{"path": "."}')
    assert "f  a.txt" in result
    assert "d  subdir" in result


def test_ls_nonexistent(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("ls", '{"path": "nope"}')
    assert "does not exist" in result


def test_ls_not_a_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hi")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("ls", '{"path": "file.txt"}')
    assert "not a directory" in result


def test_ls_empty_directory(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("ls", '{"path": "."}')
    assert result == "(empty directory)"


# --- built-in read ---


def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "hello.txt"}')
    assert result == "hello world"


def test_read_nonexistent(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "nope.txt"}')
    assert "does not exist" in result


def test_read_not_a_file(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "."}')
    assert "not a file" in result


def test_read_binary_file(tmp_path: Path) -> None:
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "img.png"}')
    assert "binary file" in result


def test_read_large_file_truncated(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(250)]
    (tmp_path / "big.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "big.txt"}')
    assert "line 0" in result
    assert "line 99" in result
    assert "line 100" not in result
    assert "150 lines remaining" in result


def test_read_with_offset(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(250)]
    (tmp_path / "big.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "big.txt", "offset": 200}')
    assert "line 199" not in result
    assert "line 200" in result
    assert "line 249" in result
    assert "remaining" not in result


def test_read_with_limit(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(50)]
    (tmp_path / "f.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "f.txt", "limit": 10}')
    assert "line 0" in result
    assert "line 9" in result
    assert "line 10" not in result
    assert "40 lines remaining" in result


def test_read_with_offset_and_limit(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(100)]
    (tmp_path / "f.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute(
            "read", '{"path": "f.txt", "offset": 10, "limit": 5}'
        )
    assert "line 9" not in result
    assert "line 10" in result
    assert "line 14" in result
    assert "line 15" not in result
    assert "85 lines remaining" in result


# --- path sanitization ---


def test_path_traversal_blocked_ls(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("ls", '{"path": ".."}')
    assert "outside the working directory" in result


def test_path_traversal_blocked_read(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "../etc/passwd"}')
    assert "outside the working directory" in result


def test_nested_traversal_blocked(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "sub/../../etc/passwd"}')
    assert "outside the working directory" in result


def test_absolute_path_outside_blocked(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "/etc/passwd"}')
    assert "outside the working directory" in result


def test_subdir_access_allowed(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "file.txt").write_text("nested")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = default_registry.execute("read", '{"path": "sub/file.txt"}')
    assert result == "nested"


# --- exception handling ---


def test_execute_catches_tool_error() -> None:
    """ToolError raised inside a tool is caught and returned as error string."""

    def _fail(x: str) -> str:
        raise ToolError("something went wrong")

    registry = ToolRegistry()
    registry.add(
        name="fail",
        description="Always fails.",
        params=[Param(name="x", description="Input.")],
        func=_fail,
    )
    result = registry.execute("fail", '{"x": "test"}')
    assert "Error:" in result
    assert "something went wrong" in result


def test_execute_catches_generic_exception() -> None:
    """Unexpected exceptions are also caught and returned as error strings."""

    def _boom(x: str) -> str:
        raise PermissionError("access denied")

    registry = ToolRegistry()
    registry.add(
        name="boom",
        description="Permission error.",
        params=[Param(name="x", description="Input.")],
        func=_boom,
    )
    result = registry.execute("boom", '{"x": "test"}')
    assert "Error:" in result
    assert "access denied" in result


def test_execute_catches_invalid_json() -> None:
    """Malformed JSON arguments are caught gracefully."""
    registry = ToolRegistry()
    registry.add(
        name="echo",
        description="Echo.",
        params=[Param(name="text", description="Text.")],
        func=lambda text: text,
    )
    result = registry.execute("echo", "not valid json")
    assert "Error:" in result
