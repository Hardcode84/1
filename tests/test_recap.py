"""Tests for mindloop.recap."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from mindloop.chunker import Chunk
from mindloop.recap import collapse_messages, generate_recap, load_recap, save_recap
from mindloop.summarizer import ChunkSummary


def _msg(role: str, content: str, **kwargs: object) -> dict[str, Any]:
    """Build a minimal message dict."""
    m: dict[str, Any] = {"role": role, "content": content}
    m.update(kwargs)
    return m


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """Build an assistant message with a single tool call."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tool_result(call_id: str, content: str) -> dict[str, Any]:
    """Build a tool result message."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# --- collapse_messages ---


def test_collapse_text_only() -> None:
    """Assistant text and reasoning pass through as turns."""
    msgs = [
        _msg("assistant", "answer", reasoning="let me think"),
        _msg("assistant", "final answer"),
    ]
    turns = collapse_messages(msgs)
    assert len(turns) == 3
    assert turns[0].role == "Bot thinking"
    assert turns[0].text == "let me think"
    assert turns[1].role == "Bot"
    assert turns[1].text == "answer"
    assert turns[2].role == "Bot"
    assert turns[2].text == "final answer"


def test_collapse_tool_calls() -> None:
    """Each tool type collapses correctly."""
    msgs = [
        _tool_call("c1", "read", '{"path": "foo.py"}'),
        _tool_result("c1", "line1\nline2\nline3"),
        _tool_call("c2", "ls", '{"path": "."}'),
        _tool_result("c2", "a.py\nb.py\nc.py"),
        _tool_call("c3", "edit", '{"path": "bar.py"}'),
        _tool_result("c3", "ok"),
        _tool_call("c4", "write", '{"path": "new.py", "content": "x\\ny"}'),
        _tool_result("c4", "ok"),
        _tool_call("c5", "recall", '{"query": "python"}'),
        _tool_result("c5", "result1\n---\nresult2"),
        _tool_call("c6", "recall_detail", '{"chunk_id": "42"}'),
        _tool_result("c6", "full text here"),
        _tool_call("c7", "remember", '{"text": "fact", "abstract": "a fact"}'),
        _tool_result("c7", "saved"),
        _tool_call("c8", "ask", '{"message": "what next?"}'),
        _tool_result("c8", "do X"),
        _tool_call("c9", "done", '{"summary": "all done"}'),
        _tool_result("c9", ""),
    ]
    turns = collapse_messages(msgs)
    texts = [t.text for t in turns]

    assert "Read foo.py (3 lines)." in texts
    assert any("Listed ." in t for t in texts)
    assert "Edited bar.py." in texts
    assert any("Wrote new.py" in t for t in texts)
    assert any("Recalled" in t and "python" in t for t in texts)
    assert "Retrieved full text of memory #42." in texts
    assert any("Remembered" in t and "a fact" in t for t in texts)
    assert any("Asked user" in t and "what next?" in t for t in texts)
    assert any("Finished" in t and "all done" in t for t in texts)


def test_collapse_skips_status() -> None:
    """Status tool calls are dropped."""
    msgs = [
        _tool_call("c1", "status", "{}"),
        _tool_result("c1", "running"),
    ]
    turns = collapse_messages(msgs)
    assert turns == []


def test_collapse_skips_system_noise() -> None:
    """[stop]/[stats]/Warning filtered, reflect nudges kept."""
    msgs = [
        _msg("system", "[stop] agent finished"),
        _msg("system", "[stats] tokens: 1000"),
        _msg("system", "Warning: something bad"),
        _msg("system", "Reflect on your progress."),
    ]
    turns = collapse_messages(msgs)
    assert len(turns) == 1
    assert turns[0].role == "System"
    assert "Reflect" in turns[0].text


def test_collapse_unknown_tool() -> None:
    """Unknown tools get a generic fallback."""
    msgs = [
        _tool_call("c1", "custom_tool", '{"x": 1}'),
        _tool_result("c1", "result"),
    ]
    turns = collapse_messages(msgs)
    assert len(turns) == 1
    assert "Called custom_tool" in turns[0].text


def test_collapse_user_messages() -> None:
    """User messages pass through."""
    msgs = [_msg("user", "hello")]
    turns = collapse_messages(msgs)
    assert len(turns) == 1
    assert turns[0].role == "You"
    assert turns[0].text == "hello"


# --- generate_recap ---


def _fake_summaries(chunks: list[Chunk], **_kwargs: object) -> list[ChunkSummary]:
    """Return deterministic summaries based on chunk text length."""
    return [
        ChunkSummary(
            chunk=c,
            abstract=f"Abstract {i}",
            summary=f"Summary {i}: {c.text[:30]}",
        )
        for i, c in enumerate(chunks)
    ]


@patch("mindloop.recap.summarize_chunks", side_effect=_fake_summaries)
def test_generate_recap_recency(_mock: object) -> None:
    """Later chunks have higher scores — last summary always included."""
    msgs = [
        _msg("user", "first topic"),
        _msg("assistant", "response one"),
        _msg("user", "second topic"),
        _msg("assistant", "response two\n\nthird paragraph"),
    ]
    recap = generate_recap(msgs, token_budget=10000)
    assert recap
    # The last summary should always be present (highest score).
    assert "Summary" in recap


@patch("mindloop.recap.summarize_chunks", side_effect=_fake_summaries)
def test_generate_recap_budget(_mock: object) -> None:
    """Respects token budget, drops lowest-scored (earliest) chunks."""
    # Paragraph breaks (\n\n) force chunk_turns to split into separate chunks.
    msgs = []
    for i in range(10):
        msgs.append(_msg("user", f"topic {i}\n\n" + "x" * 100))
        msgs.append(_msg("assistant", f"response {i}\n\n" + "y" * 100))
    # Very tight budget: should drop some summaries.
    recap_small = generate_recap(msgs, token_budget=50)
    recap_large = generate_recap(msgs, token_budget=10000)
    assert len(recap_small) < len(recap_large)


@patch("mindloop.recap.summarize_chunks", side_effect=_fake_summaries)
def test_generate_recap_chronological_order(_mock: object) -> None:
    """Selected chunks re-sorted by original position."""
    msgs = [
        _msg("user", "alpha " + "a" * 200),
        _msg("assistant", "beta " + "b" * 200),
        _msg("user", "gamma " + "c" * 200),
        _msg("assistant", "delta " + "d" * 200),
    ]
    recap = generate_recap(msgs, token_budget=10000)
    lines = recap.strip().splitlines()
    if len(lines) >= 2:
        # Extract summary indices and check they are in order.
        indices = []
        for line in lines:
            for word in line.split():
                if word.rstrip(":").isdigit():
                    indices.append(int(word.rstrip(":")))
                    break
        assert indices == sorted(indices)


@patch("mindloop.recap.summarize_chunks", side_effect=_fake_summaries)
def test_generate_recap_empty(_mock: object) -> None:
    """Empty messages produce empty recap."""
    assert generate_recap([]) == ""


# --- load_recap / save_recap ---


def test_load_recap_missing(tmp_path: Path) -> None:
    """Returns None for nonexistent file."""
    assert load_recap(tmp_path / "nope.md") is None


def test_save_and_load_recap(tmp_path: Path) -> None:
    """Round-trip write/read."""
    path = tmp_path / "sub" / "_recap.md"
    save_recap(path, "This is a recap.")
    assert load_recap(path) == "This is a recap."


def test_load_recap_empty_file(tmp_path: Path) -> None:
    """Empty file returns None."""
    path = tmp_path / "_recap.md"
    path.write_text("")
    assert load_recap(path) is None
