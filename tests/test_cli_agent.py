"""Tests for mindloop.cli.agent."""

import json
from pathlib import Path

from mindloop.cli.agent import _load_messages


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
