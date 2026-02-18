"""Tests for mindloop.tools."""

from pathlib import Path
from unittest.mock import patch

from mindloop.tools import Param, ToolError, ToolRegistry, create_default_registry


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
        result = create_default_registry().execute("ls", '{"path": "."}')
    assert "Type Name" in result
    assert "f    a.txt" in result
    assert "d    subdir" in result
    assert "(2 entries)" in result


def test_ls_nonexistent(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": "nope"}')
    assert "does not exist" in result


def test_ls_not_a_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hi")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": "file.txt"}')
    assert "not a directory" in result


def test_ls_empty_directory(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": "."}')
    assert result == "(empty directory)"


def _make_ls_dir(tmp_path: Path, n: int = 5) -> None:
    """Create files a.txt .. e.txt in tmp_path."""
    for i in range(n):
        (tmp_path / f"{chr(ord('a') + i)}.txt").write_text("")


def test_ls_limit(tmp_path: Path) -> None:
    _make_ls_dir(tmp_path)
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": ".", "limit": 2}')
    assert "a.txt" in result
    assert "b.txt" in result
    assert "c.txt" not in result
    assert "(5 entries)" in result


def test_ls_offset(tmp_path: Path) -> None:
    _make_ls_dir(tmp_path)
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "ls", '{"path": ".", "offset": 3, "limit": 10}'
        )
    assert "a.txt" not in result
    assert "d.txt" in result
    assert "e.txt" in result


def test_ls_negative_offset(tmp_path: Path) -> None:
    _make_ls_dir(tmp_path)
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": ".", "offset": -2}')
    assert "c.txt" not in result
    assert "d.txt" in result
    assert "e.txt" in result


def test_ls_invalid_limit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": ".", "limit": 0}')
    assert "limit must be >= 1" in result


# --- built-in read ---


def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "hello.txt"}')
    assert result == "hello world"


def test_read_nonexistent(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "nope.txt"}')
    assert "does not exist" in result


def test_read_not_a_file(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "."}')
    assert "not a file" in result


def test_read_binary_file(tmp_path: Path) -> None:
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "img.png"}')
    assert "binary file" in result


def test_read_large_file_truncated(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(250)]
    (tmp_path / "big.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "big.txt"}')
    assert "line 0" in result
    assert "line 49" in result
    assert "line 50" not in result
    assert "200 lines remaining" in result


def test_read_with_offset(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(250)]
    (tmp_path / "big.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "big.txt", "line_offset": 200}'
        )
    assert "line 199" not in result
    assert "line 200" in result
    assert "line 249" in result
    assert "remaining" not in result


def test_read_with_negative_offset(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(20)]
    (tmp_path / "f.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "f.txt", "line_offset": -5}'
        )
    assert "line 14" not in result
    assert "line 15" in result
    assert "line 19" in result
    assert "remaining" not in result


def test_read_with_negative_offset_and_limit(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(200)]
    (tmp_path / "f.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "f.txt", "line_offset": -100, "line_limit": 200}'
        )
    # Should return lines 100-199 (last 100 lines).
    assert "line 99" not in result
    assert "line 100" in result
    assert "line 199" in result
    assert "remaining" not in result


def test_read_with_limit(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(50)]
    (tmp_path / "f.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "f.txt", "line_limit": 10}'
        )
    assert "line 0" in result
    assert "line 9" in result
    assert "line 10" not in result
    assert "40 lines remaining" in result


def test_read_with_offset_and_limit(tmp_path: Path) -> None:
    lines = [f"line {i}\n" for i in range(100)]
    (tmp_path / "f.txt").write_text("".join(lines))
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "f.txt", "line_offset": 10, "line_limit": 5}'
        )
    assert "line 9" not in result
    assert "line 10" in result
    assert "line 14" in result
    assert "line 15" not in result
    assert "85 lines remaining" in result


def test_read_long_lines_truncated(tmp_path: Path) -> None:
    # 500 chars + \n = 501 char line, truncated at 100 -> 401 chars truncated.
    (tmp_path / "wide.txt").write_text("x" * 500 + "\nshort\n")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "wide.txt"}')
    assert "x" * 100 in result
    assert "x" * 101 not in result
    assert "chars truncated" in result
    assert "short" in result


def test_read_long_lines_custom_max(tmp_path: Path) -> None:
    # 100 chars + \n = 101 char line, truncated at 50 -> 51 chars truncated.
    (tmp_path / "wide.txt").write_text("y" * 100 + "\n")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "wide.txt", "max_line_length": 50}'
        )
    assert "y" * 50 in result
    assert "y" * 51 not in result
    assert "chars truncated" in result


# --- built-in write ---


def test_write_new_file(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "write", '{"path": "new.txt", "content": "hello\\n"}'
        )
    assert "Wrote 1 lines" in result
    assert (tmp_path / "new.txt").read_text() == "hello\n"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "write", '{"path": "a/b/c.txt", "content": "deep"}'
        )
    assert "Wrote" in result
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_write_existing_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("old")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "write", '{"path": "f.txt", "content": "new"}'
        )
    assert "already exists" in result
    assert (tmp_path / "f.txt").read_text() == "old"


def test_write_existing_with_overwrite(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("old")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "write", '{"path": "f.txt", "content": "new", "overwrite": true}'
        )
    assert "Wrote" in result
    assert (tmp_path / "f.txt").read_text() == "new"


def test_write_path_traversal_blocked(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "write", '{"path": "../escape.txt", "content": "bad"}'
        )
    assert "outside the working directory" in result


# --- built-in mv ---


def test_mv_basic(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "mv", '{"old_path": "a.txt", "new_path": "b.txt"}'
        )
    assert "Moved" in result
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").read_text() == "hello"


def test_mv_creates_parent_dirs(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("deep")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "mv", '{"old_path": "a.txt", "new_path": "sub/dir/b.txt"}'
        )
    assert "Moved" in result
    assert (tmp_path / "sub" / "dir" / "b.txt").read_text() == "deep"


def test_mv_no_overwrite(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("src")
    (tmp_path / "b.txt").write_text("dst")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "mv", '{"old_path": "a.txt", "new_path": "b.txt"}'
        )
    assert "already exists" in result
    assert (tmp_path / "a.txt").read_text() == "src"


def test_mv_with_overwrite(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("src")
    (tmp_path / "b.txt").write_text("dst")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "mv", '{"old_path": "a.txt", "new_path": "b.txt", "overwrite": true}'
        )
    assert "Moved" in result
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").read_text() == "src"


def test_mv_nonexistent(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "mv", '{"old_path": "nope.txt", "new_path": "b.txt"}'
        )
    assert "does not exist" in result


def test_mv_write_blocked(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    with patch("mindloop.tools._work_dir", tmp_path):
        reg = create_default_registry()
        reg.write_blocked[(tmp_path / "a.txt").resolve()] = ""
        result = reg.execute("mv", '{"old_path": "a.txt", "new_path": "b.txt"}')
    assert "denied" in result


# --- built-in edit ---


def test_edit_single_replacement(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello world")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "f.txt", "old_string": "hello", "new_string": "goodbye"}',
        )
    assert "Replaced 1" in result
    assert (tmp_path / "f.txt").read_text() == "goodbye world"


def test_edit_not_found(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello world")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "f.txt", "old_string": "missing", "new_string": "x"}',
        )
    assert "not found" in result


def test_edit_ambiguous_without_replace_all(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("aaa bbb aaa")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "f.txt", "old_string": "aaa", "new_string": "ccc"}',
        )
    assert "matches 2 locations" in result
    # File should be unchanged.
    assert (tmp_path / "f.txt").read_text() == "aaa bbb aaa"


def test_edit_replace_all(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("aaa bbb aaa")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "f.txt", "old_string": "aaa", "new_string": "ccc", "replace_all": true}',
        )
    assert "Replaced 2" in result
    assert (tmp_path / "f.txt").read_text() == "ccc bbb ccc"


def test_edit_identical_strings(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "f.txt", "old_string": "hello", "new_string": "hello"}',
        )
    assert "identical" in result


def test_edit_nonexistent_file(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "nope.txt", "old_string": "a", "new_string": "b"}',
        )
    assert "does not exist" in result


def test_edit_multiline(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("def foo():\n    return 1\n")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "edit",
            '{"path": "f.txt", "old_string": "def foo():\\n    return 1", "new_string": "def foo():\\n    return 2"}',
        )
    assert "Replaced 1" in result
    assert (tmp_path / "f.txt").read_text() == "def foo():\n    return 2\n"


# --- path sanitization ---


def test_path_traversal_blocked_ls(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("ls", '{"path": ".."}')
    assert "outside the working directory" in result


def test_path_traversal_blocked_read(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "../etc/passwd"}')
    assert "outside the working directory" in result


def test_nested_traversal_blocked(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute(
            "read", '{"path": "sub/../../etc/passwd"}'
        )
    assert "outside the working directory" in result


def test_absolute_path_outside_blocked(tmp_path: Path) -> None:
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "/etc/passwd"}')
    assert "outside the working directory" in result


def test_subdir_access_allowed(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "file.txt").write_text("nested")
    with patch("mindloop.tools._work_dir", tmp_path):
        result = create_default_registry().execute("read", '{"path": "sub/file.txt"}')
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


def test_blocked_dirs_prevents_read(tmp_path: Path) -> None:
    """File tools reject paths inside blocked directories."""
    secret = tmp_path / "secrets"
    secret.mkdir()
    (secret / "key.txt").write_text("top secret")

    reg = create_default_registry(blocked_dirs=[secret], root_dir=tmp_path)
    result = reg.execute("read", '{"path": "secrets/key.txt"}')
    assert "Access denied" in result


def test_blocked_dirs_allows_other_paths(tmp_path: Path) -> None:
    """File tools allow paths outside blocked directories."""
    (tmp_path / "ok.txt").write_text("visible")
    secret = tmp_path / "secrets"
    secret.mkdir()

    reg = create_default_registry(blocked_dirs=[secret], root_dir=tmp_path)
    result = reg.execute("read", '{"path": "ok.txt"}')
    assert "visible" in result


# --- virtual symlinks ---


def test_symlink_read(tmp_path: Path) -> None:
    """Reading a file through a virtual symlink works."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "data.txt").write_text("shared data")

    reg = create_default_registry(root_dir=workspace, symlinks={"shared": external})
    result = reg.execute("read", '{"path": "shared/data.txt"}')
    assert result == "shared data"


def test_symlink_ls(tmp_path: Path) -> None:
    """Listing a virtual symlink directory works."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "a.txt").write_text("")
    (external / "b.txt").write_text("")

    reg = create_default_registry(root_dir=workspace, symlinks={"docs": external})
    result = reg.execute("ls", '{"path": "docs"}')
    assert "a.txt" in result
    assert "b.txt" in result


def test_symlink_visible_in_parent_ls(tmp_path: Path) -> None:
    """Virtual symlinks appear as 'l' entries in their parent directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "real.txt").write_text("")
    external = tmp_path / "external"
    external.mkdir()

    reg = create_default_registry(root_dir=workspace, symlinks={"shared": external})
    result = reg.execute("ls", '{"path": "."}')
    assert "f    real.txt" in result
    assert "l    shared" in result


def test_symlink_write_blocked(tmp_path: Path) -> None:
    """Writing through a virtual symlink is blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "data.txt").write_text("original")

    reg = create_default_registry(root_dir=workspace, symlinks={"shared": external})
    result = reg.execute("write", '{"path": "shared/data.txt", "content": "hacked"}')
    assert "read-only symlink" in result
    assert (external / "data.txt").read_text() == "original"


def test_symlink_edit_blocked(tmp_path: Path) -> None:
    """Editing through a virtual symlink is blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "data.txt").write_text("original")

    reg = create_default_registry(root_dir=workspace, symlinks={"shared": external})
    result = reg.execute(
        "edit",
        '{"path": "shared/data.txt", "old_string": "original", "new_string": "changed"}',
    )
    assert "read-only symlink" in result


def test_symlink_escape_blocked(tmp_path: Path) -> None:
    """Traversal escaping a symlink target is blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    reg = create_default_registry(root_dir=workspace, symlinks={"shared": external})
    result = reg.execute("read", '{"path": "shared/../../../etc/passwd"}')
    assert "escapes" in result.lower() or "Error" in result
