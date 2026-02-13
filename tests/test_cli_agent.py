"""Tests for mindloop.cli.agent."""

import json
from pathlib import Path
from unittest.mock import patch

from mindloop.cli.agent import (
    _NOTES_MAX_CHARS,
    _latest_jsonl,
    _load_messages,
    _session_exit_reason,
    _setup_session,
)


def test_load_messages_strips_metadata(tmp_path: Path) -> None:
    """timestamp and usage fields are removed from loaded messages."""
    log = tmp_path / "session.jsonl"
    entry = {
        "timestamp": "2026-01-01T00:00:00",
        "role": "assistant",
        "content": "hello",
        "usage": {"total_tokens": 100},
    }
    log.write_text(json.dumps(entry) + "\n")

    msgs = _load_messages(log)
    assert len(msgs) == 1
    assert "timestamp" not in msgs[0]
    assert "usage" not in msgs[0]
    assert msgs[0]["content"] == "hello"


def test_load_messages_skips_stop_and_stats(tmp_path: Path) -> None:
    """System messages from the agent loop are filtered out."""
    log = tmp_path / "session.jsonl"
    lines = [
        json.dumps({"timestamp": "t", "role": "assistant", "content": "hi"}),
        json.dumps({"timestamp": "t", "role": "system", "content": "[stop] done"}),
        json.dumps({"timestamp": "t", "role": "system", "content": "[stats] {}"}),
        json.dumps(
            {
                "timestamp": "t",
                "role": "system",
                "content": "Warning: 50% of token budget used",
            }
        ),
    ]
    log.write_text("\n".join(lines) + "\n")

    msgs = _load_messages(log)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"


def test_load_messages_keeps_reflect(tmp_path: Path) -> None:
    """Reflect nudge system messages are preserved."""
    log = tmp_path / "session.jsonl"
    entry = {
        "timestamp": "t",
        "role": "system",
        "content": "You've been using tools for a while. Pause and reflect.",
    }
    log.write_text(json.dumps(entry) + "\n")

    msgs = _load_messages(log)
    assert len(msgs) == 1


def test_load_messages_preserves_tool_calls(tmp_path: Path) -> None:
    """Tool calls and tool results survive the round-trip."""
    log = tmp_path / "session.jsonl"
    lines = [
        json.dumps(
            {
                "timestamp": "t",
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": '{"path": "."}'},
                    }
                ],
            }
        ),
        json.dumps(
            {
                "timestamp": "t",
                "role": "tool",
                "tool_call_id": "c1",
                "content": "f  README.md",
            }
        ),
    ]
    log.write_text("\n".join(lines) + "\n")

    msgs = _load_messages(log)
    assert len(msgs) == 2
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "ls"
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "c1"


# --- Session setup tests ---


def test_setup_session_creates_dirs(tmp_path: Path) -> None:
    """Named session creates logs/ and workspace/ subdirs."""
    with patch("mindloop.cli.agent._SESSIONS_DIR", tmp_path / "sessions"):
        paths = _setup_session("test_run", isolated=False, timestamp="20260212_000000")

    assert paths.log_dir.exists()
    assert paths.log_dir.name == "logs"
    assert paths.workspace is not None
    assert paths.workspace.exists()
    assert paths.workspace.name == "workspace"
    assert paths.db_path.parent == paths.log_dir.parent
    assert not paths.blocked_dirs


def test_setup_session_isolated(tmp_path: Path) -> None:
    """Isolated mode creates a timestamped session with blocked dirs."""
    with patch("mindloop.cli.agent._SESSIONS_DIR", tmp_path / "sessions"):
        paths = _setup_session(None, isolated=True, timestamp="20260212_000000")

    assert "isolated_20260212_000000" in str(paths.log_dir)
    assert paths.blocked_dirs


def test_setup_default_paths(tmp_path: Path) -> None:
    """Default mode uses shared logs/ and memory/."""
    with (patch("mindloop.cli.agent.Path", side_effect=lambda p: tmp_path / p),):
        # Can't easily patch Path() calls, test the logic directly.
        paths = _setup_session(None, isolated=False, timestamp="20260212_000000")

    assert paths.workspace is None
    assert not paths.blocked_dirs


def test_setup_session_copies_template(tmp_path: Path) -> None:
    """Template files are copied into a fresh workspace."""
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "hello.txt").write_text("hi")
    (tpl / "sub").mkdir()
    (tpl / "sub" / "nested.txt").write_text("deep")

    with (
        patch("mindloop.cli.agent._SESSIONS_DIR", tmp_path / "sessions"),
        patch("mindloop.cli.agent._TEMPLATE_DIR", tpl),
    ):
        paths = _setup_session("tpl_run", isolated=False, timestamp="20260212_000000")

    assert paths.workspace is not None
    assert (paths.workspace / "hello.txt").read_text() == "hi"
    assert (paths.workspace / "sub" / "nested.txt").read_text() == "deep"


def test_setup_session_skips_template_on_existing(tmp_path: Path) -> None:
    """Template is not re-copied when workspace already exists."""
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "hello.txt").write_text("hi")

    sessions = tmp_path / "sessions"
    ws = sessions / "existing" / "workspace"
    ws.mkdir(parents=True)

    with (
        patch("mindloop.cli.agent._SESSIONS_DIR", sessions),
        patch("mindloop.cli.agent._TEMPLATE_DIR", tpl),
    ):
        paths = _setup_session("existing", isolated=False, timestamp="20260212_000000")

    assert paths.workspace is not None
    assert not (paths.workspace / "hello.txt").exists()


def test_setup_session_instance_counter(tmp_path: Path) -> None:
    """Instance number increments with each run in a session."""
    sessions = tmp_path / "sessions"
    with patch("mindloop.cli.agent._SESSIONS_DIR", sessions):
        p1 = _setup_session("run", isolated=False, timestamp="20260212_100000")
    assert p1.instance == 1

    # Simulate a log file from the first run.
    (p1.log_dir / "001_agent_20260212_100000.jsonl").write_text("{}\n")

    with patch("mindloop.cli.agent._SESSIONS_DIR", sessions):
        p2 = _setup_session("run", isolated=False, timestamp="20260212_110000")
    assert p2.instance == 2


def test_latest_jsonl(tmp_path: Path) -> None:
    """Finds the most recent JSONL file by sorted name."""
    (tmp_path / "001_agent_20260211_100000.jsonl").write_text("{}\n")
    (tmp_path / "002_agent_20260212_100000.jsonl").write_text("{}\n")

    latest = _latest_jsonl(tmp_path)
    assert latest is not None
    assert latest.name == "002_agent_20260212_100000.jsonl"


def test_latest_jsonl_empty(tmp_path: Path) -> None:
    """Returns None when no JSONL files exist."""
    assert _latest_jsonl(tmp_path) is None


# --- Session exit reason tests ---


def _make_log(tmp_path: Path, stop_content: str | None) -> Path:
    """Create a minimal JSONL log with an optional [stop] line."""
    log = tmp_path / "001_agent_20260213_000000.jsonl"
    lines = [json.dumps({"role": "assistant", "content": "hello"})]
    if stop_content is not None:
        lines.append(json.dumps({"role": "system", "content": stop_content}))
    log.write_text("\n".join(lines) + "\n")
    return log


def test_exit_reason_clean(tmp_path: Path) -> None:
    """Clean exit via done returns None."""
    log = _make_log(tmp_path, "[stop] model finished (1000 tokens used)")
    assert _session_exit_reason(log) is None


def test_exit_reason_token_budget(tmp_path: Path) -> None:
    """Token budget exceeded is detected."""
    log = _make_log(tmp_path, "[stop] token budget exceeded (limit 50000)")
    reason = _session_exit_reason(log)
    assert reason is not None
    assert "ran out of tokens" in reason


def test_exit_reason_max_iterations(tmp_path: Path) -> None:
    """Max iterations reached is detected."""
    log = _make_log(tmp_path, "[stop] max iterations reached (200)")
    reason = _session_exit_reason(log)
    assert reason is not None
    assert "iteration limit" in reason


def test_exit_reason_crash(tmp_path: Path) -> None:
    """Missing [stop] line indicates a crash."""
    log = _make_log(tmp_path, None)
    reason = _session_exit_reason(log)
    assert reason is not None
    assert "abruptly" in reason


# --- note_to_self tests ---


def _make_session_with_tool(tmp_path: Path) -> tuple[Path, "ToolRegistry"]:  # type: ignore[name-defined]  # noqa: F821
    """Set up a session and build a registry that includes note_to_self."""

    from mindloop.tools import Param, create_default_registry

    sessions = tmp_path / "sessions"
    with patch("mindloop.cli.agent._SESSIONS_DIR", sessions):
        paths = _setup_session(
            "notes_test", isolated=False, timestamp="20260213_000000"
        )

    assert paths.workspace is not None
    notes_path = paths.workspace / "_notes.md"

    registry = create_default_registry(root_dir=paths.workspace)
    # Block direct write/edit, same as main().
    registry.write_blocked.append(notes_path.resolve())
    # Replicate the closure from main().
    from mindloop.cli.agent import _NOTES_MAX_CHARS as max_chars

    def _note_to_self(content: str) -> str:
        if len(content) > max_chars:
            return (
                f"Error: content is {len(content)} chars, "
                f"max is {max_chars}. Trim and retry."
            )
        previous = notes_path.read_text() if notes_path.is_file() else None
        notes_path.write_text(content)
        result = f"Saved {len(content)} chars to notes."
        if previous:
            result += f"\n\n--- Previous notes (overwritten) ---\n{previous}"
        return result

    registry.add(
        name="note_to_self",
        description="Write notes.",
        params=[Param(name="content", description="Markdown content.")],
        func=_note_to_self,
    )
    return notes_path, registry


def test_note_to_self_writes_file(tmp_path: Path) -> None:
    """note_to_self writes content to _notes.md."""
    notes_path, registry = _make_session_with_tool(tmp_path)
    result = registry.execute("note_to_self", json.dumps({"content": "remember this"}))
    assert "Saved" in result
    assert "Previous notes" not in result
    assert notes_path.read_text() == "remember this"


def test_note_to_self_overwrites_and_shows_previous(tmp_path: Path) -> None:
    """Successive calls overwrite and return previous content."""
    notes_path, registry = _make_session_with_tool(tmp_path)
    registry.execute("note_to_self", json.dumps({"content": "first"}))
    result = registry.execute("note_to_self", json.dumps({"content": "second"}))
    assert notes_path.read_text() == "second"
    assert "Previous notes (overwritten)" in result
    assert "first" in result


def test_note_to_self_rejects_oversized(tmp_path: Path) -> None:
    """Content exceeding the size cap returns an error."""
    _, registry = _make_session_with_tool(tmp_path)
    big = "x" * (_NOTES_MAX_CHARS + 1)
    result = registry.execute("note_to_self", json.dumps({"content": big}))
    assert "Error" in result
    assert "Trim and retry" in result


def test_notes_write_blocked_but_readable(tmp_path: Path) -> None:
    """Direct write/edit of _notes.md is denied, but read is allowed."""
    notes_path, registry = _make_session_with_tool(tmp_path)
    # Write via note_to_self succeeds.
    registry.execute("note_to_self", json.dumps({"content": "ok"}))
    assert notes_path.read_text() == "ok"
    # Direct write is blocked.
    result = registry.execute(
        "write", json.dumps({"content": "bypass", "path": "_notes.md"})
    )
    assert "Write access denied" in result
    # Direct read is allowed.
    result = registry.execute("read", json.dumps({"path": "_notes.md"}))
    assert "ok" in result
