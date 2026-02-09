"""Tests for mindloop.chunker."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mindloop.chunker import (
    Chunk,
    Turn,
    chunk_turns,
    compact_chunks,
    cosine_similarities,
    merge_chunks,
    parse_turns,
)


def _turn(h: int, m: int, s: int, role: str, text: str) -> Turn:
    """Helper to create a Turn with a timestamp."""
    return Turn(timestamp=datetime(1900, 1, 1, h, m, s), role=role, text=text)


# --- parse_turns ---


def test_parse_turns_basic(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "hello"}\n'
        '{"timestamp": "2025-01-15T14:30:07", "role": "assistant", "content": "hi there"}\n'
    )
    turns = parse_turns(log)
    assert len(turns) == 2
    assert turns[0].role == "You"
    assert turns[0].text == "hello"
    assert turns[1].role == "Bot"
    assert turns[1].text == "hi there"


def test_parse_turns_multiline_content(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "line one\\nline two"}\n'
    )
    turns = parse_turns(log)
    assert turns[0].text == "line one\nline two"


def test_parse_turns_empty(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text("")
    assert parse_turns(log) == []


def test_parse_turns_preserves_timestamp(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "user", "content": "hi"}\n'
    )
    turns = parse_turns(log)
    assert turns[0].timestamp == datetime(2025, 1, 15, 14, 30, 5)


def test_parse_turns_with_reasoning(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "assistant",'
        ' "content": "answer", "reasoning": "let me think"}\n'
    )
    turns = parse_turns(log)
    assert len(turns) == 2
    assert turns[0].role == "Bot thinking"
    assert turns[0].text == "let me think"
    assert turns[1].role == "Bot"
    assert turns[1].text == "answer"


def test_parse_turns_without_reasoning(tmp_path: Path) -> None:
    log = tmp_path / "chat.jsonl"
    log.write_text(
        '{"timestamp": "2025-01-15T14:30:05", "role": "assistant", "content": "hi"}\n'
    )
    turns = parse_turns(log)
    assert len(turns) == 1
    assert turns[0].role == "Bot"


# --- chunk_turns ---


def test_chunk_turns_empty() -> None:
    assert chunk_turns([]) == []


def test_chunk_turns_single_chunk() -> None:
    turns = [
        _turn(14, 30, 0, "You", "hi"),
        _turn(14, 30, 5, "Bot", "hello"),
    ]
    chunks = chunk_turns(turns)
    assert len(chunks) == 1
    assert len(chunks[0].turns) == 2


def test_chunk_turns_no_blank_lines() -> None:
    turns = [
        _turn(14, 30, 0, "You", "hi"),
        _turn(14, 30, 5, "Bot", "hello"),
        _turn(14, 35, 0, "You", "new topic"),
    ]
    chunks = chunk_turns(turns)
    assert len(chunks) == 1
    assert len(chunks[0].turns) == 3


def test_chunk_turns_split_by_blank_in_content() -> None:
    turns = [
        _turn(14, 30, 0, "Bot", "paragraph one\n\nparagraph two"),
    ]
    chunks = chunk_turns(turns)
    assert len(chunks) == 2
    assert chunks[0].turns[0].text == "paragraph one"
    assert chunks[1].turns[0].text == "paragraph two"


def test_chunk_turns_blank_across_turns() -> None:
    turns = [
        _turn(14, 30, 0, "You", "question"),
        _turn(14, 30, 5, "Bot", "first part\n\nsecond part"),
        _turn(14, 31, 0, "You", "follow up"),
    ]
    chunks = chunk_turns(turns)
    assert len(chunks) == 2
    assert len(chunks[0].turns) == 2  # "question" + "first part".
    assert len(chunks[1].turns) == 2  # "second part" + "follow up".


# --- compact_chunks ---


def test_compact_chunks_empty() -> None:
    assert compact_chunks([]) == []


def test_compact_chunks_absorbs_small_into_next() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "Bot", "----")]),
        Chunk(turns=[_turn(14, 0, 1, "You", "a" * 100)]),
    ]
    result = compact_chunks(chunks, min_chars=20)
    assert len(result) == 1
    assert len(result[0].turns) == 2


def test_compact_chunks_absorbs_small_trailing() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "You", "a" * 100)]),
        Chunk(turns=[_turn(14, 0, 1, "Bot", "ok")]),
    ]
    result = compact_chunks(chunks, min_chars=20)
    assert len(result) == 1
    assert len(result[0].turns) == 2


def test_compact_chunks_keeps_large() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "You", "a" * 100)]),
        Chunk(turns=[_turn(14, 0, 1, "Bot", "b" * 100)]),
    ]
    result = compact_chunks(chunks, min_chars=20)
    assert len(result) == 2


def test_compact_chunks_consecutive_small() -> None:
    chunks = [
        Chunk(turns=[_turn(14, 0, 0, "Bot", "---")]),
        Chunk(turns=[_turn(14, 0, 1, "Bot", "===")]),
        Chunk(turns=[_turn(14, 0, 2, "You", "a" * 100)]),
    ]
    result = compact_chunks(chunks, min_chars=20)
    assert len(result) == 1
    assert len(result[0].turns) == 3


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
