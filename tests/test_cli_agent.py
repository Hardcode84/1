"""Tests for mindloop.cli.agent."""

import json
from pathlib import Path
from unittest.mock import patch

from mindloop.cli.agent import _latest_jsonl, _load_messages, _setup_session


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


def test_latest_jsonl(tmp_path: Path) -> None:
    """Finds the most recent JSONL file by sorted name."""
    (tmp_path / "agent_20260211_100000.jsonl").write_text("{}\n")
    (tmp_path / "agent_20260212_100000.jsonl").write_text("{}\n")
    (tmp_path / "agent_20260210_100000.jsonl").write_text("{}\n")

    latest = _latest_jsonl(tmp_path)
    assert latest is not None
    assert latest.name == "agent_20260212_100000.jsonl"


def test_latest_jsonl_empty(tmp_path: Path) -> None:
    """Returns None when no JSONL files exist."""
    assert _latest_jsonl(tmp_path) is None
