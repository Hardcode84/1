"""Tests for mindloop.tools."""

from pathlib import Path

from mindloop.tools import Param, ToolRegistry, default_registry


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

    result = default_registry.execute("ls", f'{{"path": "{tmp_path}"}}')
    assert "f  a.txt" in result
    assert "d  subdir" in result


def test_ls_nonexistent() -> None:
    result = default_registry.execute("ls", '{"path": "/nonexistent_dir_xyz"}')
    assert "does not exist" in result


def test_ls_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hi")
    result = default_registry.execute("ls", f'{{"path": "{f}"}}')
    assert "not a directory" in result


def test_ls_empty_directory(tmp_path: Path) -> None:
    result = default_registry.execute("ls", f'{{"path": "{tmp_path}"}}')
    assert result == "(empty directory)"


# --- built-in read ---


def test_read_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = default_registry.execute("read", f'{{"path": "{f}"}}')
    assert result == "hello world"


def test_read_nonexistent() -> None:
    result = default_registry.execute("read", '{"path": "/nonexistent_file_xyz"}')
    assert "does not exist" in result


def test_read_not_a_file(tmp_path: Path) -> None:
    result = default_registry.execute("read", f'{{"path": "{tmp_path}"}}')
    assert "not a file" in result
