"""Tests for mindloop.cli.rebuild."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.cli.rebuild import (
    _has_boundaries,
    _is_boundary,
    _parse_remember_calls,
    _save_extracted,
    rebuild_from_log,
)
from mindloop.memory import MemoryStore

_EMB_A = np.array([1.0, 0.0], dtype=np.float32)


@contextmanager
def _patch_embeddings() -> Iterator[None]:
    """Patch get_embeddings to return uniform vectors."""

    def _get_embeddings(texts: list[str], **_kw: object) -> np.ndarray:
        return np.tile(_EMB_A, (len(texts), 1))

    with patch("mindloop.memory.get_embeddings", side_effect=_get_embeddings):
        yield


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


# --- Boundary detection ---


def test_is_boundary_nudge() -> None:
    """User nudge message is detected as boundary."""
    msg: dict[str, object] = {
        "role": "user",
        "content": "Continue autonomously\nYou have 0 new messages.",
    }
    assert _is_boundary(msg)


def test_is_boundary_reflect() -> None:
    """System reflect message is detected as boundary."""
    msg: dict[str, object] = {
        "role": "system",
        "content": "You've been using tools for a while. Pause and reflect.",
    }
    assert _is_boundary(msg)


def test_is_boundary_normal_message() -> None:
    """Regular messages are not boundaries."""
    msg: dict[str, object] = {"role": "assistant", "content": "Hello world."}
    assert not _is_boundary(msg)


def test_has_boundaries_true() -> None:
    """Log with a nudge marker returns True."""
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "You've been using tools for a while."},
        {"role": "assistant", "content": "ok"},
    ]
    assert _has_boundaries(messages)


def test_has_boundaries_false() -> None:
    """Log without markers returns False."""
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert not _has_boundaries(messages)


# --- Remember parsing ---


def test_parse_remember_calls_extracts() -> None:
    """Remember tool calls are parsed from assistant messages."""
    msg: dict[str, object] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "functions.remember:5",
                "type": "function",
                "function": {
                    "name": "remember",
                    "arguments": json.dumps(
                        {"text": "Cats are great", "abstract": "Cats"}
                    ),
                },
            }
        ],
    }
    result = _parse_remember_calls(msg)
    assert len(result) == 1
    assert result[0] == ("Cats are great", "Cats")


def test_parse_remember_calls_ignores_other_tools() -> None:
    """Non-remember tool calls are ignored."""
    msg: dict[str, object] = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "recall", "arguments": '{"query": "cats"}'},
            }
        ],
    }
    assert _parse_remember_calls(msg) == []


def test_parse_remember_calls_no_tool_calls() -> None:
    """Messages without tool_calls return empty."""
    msg: dict[str, object] = {"role": "assistant", "content": "hello"}
    assert _parse_remember_calls(msg) == []


def test_parse_remember_calls_malformed_args() -> None:
    """Malformed arguments JSON is skipped gracefully."""
    msg: dict[str, object] = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "remember", "arguments": "not json{"},
            }
        ],
    }
    assert _parse_remember_calls(msg) == []


# --- _save_extracted ---


def test_save_extracted_dry_run() -> None:
    """Dry run counts saves without writing."""
    facts = [
        {"text": "fact one", "abstract": "abs one"},
        {"text": "fact two", "abstract": "abs two", "summary": "detail two"},
    ]
    # store is unused in dry run, pass a dummy.
    saved, skipped = _save_extracted(None, facts, model="test-model", dry_run=True, log=lambda _: None)  # type: ignore[arg-type]
    assert saved == 2
    assert skipped == 0


def test_save_extracted_dedup(store: MemoryStore) -> None:
    """Facts above dedup threshold are skipped."""
    from mindloop.chunker import Chunk, Turn
    from mindloop.summarizer import ChunkSummary
    from datetime import datetime

    # Pre-populate store with a matching memory.
    chunk = Chunk(
        turns=[
            Turn(timestamp=datetime.now(), role="memory", text="memory: cats are great")
        ]
    )
    cs = ChunkSummary(chunk=chunk, abstract="cats", summary="cats are great")
    with _patch_embeddings():
        store.save(cs)

    facts = [{"text": "cats are great", "abstract": "cats"}]

    with (
        _patch_embeddings(),
        patch("mindloop.cli.rebuild.save_memory"),
    ):
        saved, skipped = _save_extracted(
            store, facts, model="test-model", dry_run=False, log=lambda _: None
        )

    # Uniform embeddings → cosine=1.0 → duplicate.
    assert saved == 0
    assert skipped == 1


# --- rebuild_from_log ---


def _make_facts_json(facts: list[dict[str, str]]) -> str:
    return json.dumps(facts)


def test_rebuild_processes_remembers(store: MemoryStore) -> None:
    """Remember tool calls are saved via _save_remember."""
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "arguments": json.dumps(
                            {"text": "User likes cats", "abstract": "Cat preference"}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "functions.remember:1", "content": "Saved."},
    ]

    with (
        _patch_embeddings(),
        patch("mindloop.cli.rebuild.summarize_chunk") as mock_summ,
        patch("mindloop.cli.rebuild.save_memory") as mock_save,
        patch("mindloop.cli.rebuild.extract_window", return_value=[]),
    ):
        from mindloop.chunker import Chunk, Turn
        from mindloop.summarizer import ChunkSummary
        from datetime import datetime

        chunk = Chunk(turns=[Turn(timestamp=datetime.now(), role="memory", text="x")])
        mock_summ.return_value = ChunkSummary(chunk=chunk, abstract="a", summary="s")

        r, s, sk = rebuild_from_log(
            messages, store, model="test-model", dry_run=False, log=lambda _: None
        )

    assert r == 1
    assert mock_save.call_count == 1


def test_rebuild_extracts_at_boundaries(store: MemoryStore) -> None:
    """Extraction happens at boundary markers."""
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {
            "role": "system",
            "content": "You've been using tools for a while. Pause.",
        },
        {"role": "user", "content": "more stuff"},
        {"role": "assistant", "content": "ok"},
    ]

    extract_calls: list[int] = []
    facts_json = _make_facts_json(
        [{"text": "a fact", "abstract": "abs", "summary": "sum"}]
    )

    def _mock_extract(msgs: list[dict[str, Any]], model: str) -> list[dict[str, str]]:
        extract_calls.append(len(msgs))
        return list(json.loads(facts_json))

    with (
        _patch_embeddings(),
        patch("mindloop.cli.rebuild.extract_window", side_effect=_mock_extract),
        patch("mindloop.cli.rebuild.save_memory"),
    ):
        r, s, sk = rebuild_from_log(
            messages, store, model="test-model", dry_run=False, log=lambda _: None
        )

    # Two extraction calls: one at boundary, one at end-of-log.
    assert len(extract_calls) == 2
    # First window has 3 messages (up to and including boundary).
    assert extract_calls[0] == 3
    # Second window has 2 messages (after boundary).
    assert extract_calls[1] == 2


def test_rebuild_fallback_window(store: MemoryStore) -> None:
    """Without boundaries, extraction happens every _FALLBACK_WINDOW messages."""
    from mindloop.cli.rebuild import _FALLBACK_WINDOW

    # Create messages without any boundary markers.
    n = _FALLBACK_WINDOW + 5
    messages: list[dict[str, object]] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(n)
    ]

    extract_calls: list[int] = []

    def _mock_extract(msgs: list[dict[str, Any]], model: str) -> list[dict[str, str]]:
        extract_calls.append(len(msgs))
        return []

    with patch("mindloop.cli.rebuild.extract_window", side_effect=_mock_extract):
        r, s, sk = rebuild_from_log(
            messages, store, model="test-model", dry_run=False, log=lambda _: None
        )

    # First flush at _FALLBACK_WINDOW, second for the remainder.
    assert len(extract_calls) == 2
    assert extract_calls[0] == _FALLBACK_WINDOW
    assert extract_calls[1] == 5


def test_rebuild_dry_run(store: MemoryStore) -> None:
    """Dry run does not write to the store."""
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "arguments": json.dumps({"text": "A fact", "abstract": "Fact"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "r:1", "content": "Saved."},
        {
            "role": "system",
            "content": "You've been using tools for a while. Reflect.",
        },
    ]

    facts = [{"text": "extracted", "abstract": "ext", "summary": "ext detail"}]

    with patch("mindloop.cli.rebuild.extract_window", return_value=facts):
        r, s, sk = rebuild_from_log(
            messages, store, model="test-model", dry_run=True, log=lambda _: None
        )

    assert r == 1
    assert s == 1
    # Store should be empty since this was dry run.
    assert store.count() == 0
