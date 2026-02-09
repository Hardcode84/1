"""Tests for mindloop.chunker."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mindloop.chunker import (
    Chunk,
    Turn,
    chunk_turns,
    cosine_similarities,
    merge_chunks,
    parse_log,
    parse_turns,
    parse_turns_jsonl,
)


def _turn(h: int, m: int, s: int, role: str, text: str, blank: bool = False) -> Turn:
    """Helper to create a Turn with a timestamp."""
    return Turn(
        timestamp=datetime(1900, 1, 1, h, m, s),
        role=role,
        text=text,
        preceded_by_blank=blank,
    )


# --- parse_turns ---


def test_parse_turns_basic(tmp_path: Path) -> None:
    log = tmp_path / "chat.log"
    log.write_text("14:30:05 You: hello\n14:30:07 Bot: hi there\n")
    turns = parse_turns(log)
    assert len(turns) == 2
    assert turns[0].role == "You"
    assert turns[0].text == "hello"
    assert turns[1].role == "Bot"
    assert turns[1].text == "hi there"


def test_parse_turns_continuation(tmp_path: Path) -> None:
    log = tmp_path / "chat.log"
    log.write_text("14:30:05 You: line one\n  line two\n14:30:07 Bot: reply\n")
    turns = parse_turns(log)
    assert len(turns) == 2
    assert turns[0].text == "line one\nline two"


def test_parse_turns_blank_line(tmp_path: Path) -> None:
    log = tmp_path / "chat.log"
    log.write_text("14:30:05 You: hello\n\n14:30:07 Bot: hi\n")
    turns = parse_turns(log)
    assert len(turns) == 2
    assert not turns[0].preceded_by_blank
    assert turns[1].preceded_by_blank


def test_parse_turns_empty(tmp_path: Path) -> None:
    log = tmp_path / "chat.log"
    log.write_text("")
    assert parse_turns(log) == []


# --- parse_turns_jsonl ---


def test_parse_turns_jsonl_basic(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "hello"}\n'
        '{"timestamp": "2025-01-15T14:30:07", "role": "assistant", "content": "hi there"}\n'
    )
    turns = parse_turns_jsonl(log)
    assert len(turns) == 2
    assert turns[0].role == "You"
    assert turns[0].text == "hello"
    assert turns[1].role == "Bot"
    assert turns[1].text == "hi there"


def test_parse_turns_jsonl_multiline_content(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "line one\\nline two"}\n'
    )
    turns = parse_turns_jsonl(log)
    assert turns[0].text == "line one\nline two"


def test_parse_turns_jsonl_empty(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text("")
    assert parse_turns_jsonl(log) == []


def test_parse_turns_jsonl_preserves_timestamp(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "hi"}\n'
    )
    turns = parse_turns_jsonl(log)
    assert turns[0].timestamp == datetime(2025, 1, 15, 14, 30, 5)


# --- parse_log dispatcher ---


def test_parse_log_dispatches_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "hello"}\n'
    )
    turns = parse_log(log)
    assert len(turns) == 1
    assert turns[0].role == "You"


def test_parse_log_dispatches_log(tmp_path: Path) -> None:
    log = tmp_path / "chat.log"
    log.write_text("14:30:05 You: hello\n")
    turns = parse_log(log)
    assert len(turns) == 1
    assert turns[0].role == "You"


# --- chunk_turns ---


def test_chunk_turns_empty() -> None:
    assert chunk_turns([]) == []


def test_chunk_turns_single_chunk() -> None:
    turns = [
        _turn(14, 30, 0, "You", "hi"),
        _turn(14, 30, 5, "Bot", "hello"),
    ]
    chunks = chunk_turns(turns, gap_threshold=120)
    assert len(chunks) == 1
    assert len(chunks[0].turns) == 2


def test_chunk_turns_split_by_gap() -> None:
    turns = [
        _turn(14, 30, 0, "You", "hi"),
        _turn(14, 30, 5, "Bot", "hello"),
        _turn(14, 35, 0, "You", "new topic"),
    ]
    chunks = chunk_turns(turns, gap_threshold=120)
    assert len(chunks) == 2
    assert len(chunks[0].turns) == 2
    assert len(chunks[1].turns) == 1


def test_chunk_turns_split_by_blank() -> None:
    turns = [
        _turn(14, 30, 0, "You", "hi"),
        _turn(14, 30, 5, "Bot", "hello", blank=True),
    ]
    chunks = chunk_turns(turns, gap_threshold=9999)
    assert len(chunks) == 2


# --- Chunk properties ---


def test_chunk_text() -> None:
    chunk = Chunk(
        turns=[
            _turn(14, 0, 0, "You", "question"),
            _turn(14, 0, 1, "Bot", "answer"),
        ]
    )
    assert chunk.text == "You: question\nBot: answer"


def test_chunk_time_range() -> None:
    chunk = Chunk(
        turns=[
            _turn(14, 0, 0, "You", "a"),
            _turn(14, 5, 30, "Bot", "b"),
        ]
    )
    assert chunk.time_range == "14:00:00 - 14:05:30"


# --- cosine_similarities ---


def test_cosine_identical_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    sims = cosine_similarities(embeddings)
    assert len(sims) == 2
    assert all(pytest.approx(s, abs=1e-6) == 1.0 for s in sims)


def test_cosine_orthogonal_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = cosine_similarities(embeddings)
    assert len(sims) == 1
    assert pytest.approx(sims[0], abs=1e-6) == 0.0


def test_cosine_opposite_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]])
    sims = cosine_similarities(embeddings)
    assert pytest.approx(sims[0], abs=1e-6) == -1.0


# --- merge_chunks ---


def test_merge_all_similar() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "You", "a")]),
        Chunk(turns=[_turn(14, 0, 1, "Bot", "b")]),
        Chunk(turns=[_turn(14, 0, 2, "You", "c")]),
    ]
    # All high similarity — should merge into one.
    sims = np.array([0.9, 0.9])
    merged = merge_chunks(chunks, sims)
    assert len(merged) == 1
    assert len(merged[0].turns) == 3


def test_merge_none_similar() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "You", "a")]),
        Chunk(turns=[_turn(14, 0, 1, "Bot", "b")]),
        Chunk(turns=[_turn(14, 0, 2, "You", "c")]),
    ]
    # One high, one low — threshold will split at the low one.
    sims = np.array([0.9, 0.1])
    merged = merge_chunks(chunks, sims)
    assert len(merged) == 2


def test_merge_respects_max_chars() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "You", "a" * 100)]),
        Chunk(turns=[_turn(14, 0, 1, "Bot", "b" * 100)]),
    ]
    sims = np.array([0.99])
    # Tiny limit prevents merging.
    merged = merge_chunks(chunks, sims, max_chunk_chars=50)
    assert len(merged) == 2
