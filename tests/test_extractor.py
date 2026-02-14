"""Tests for mindloop.extractor."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.chunker import Chunk
from mindloop.extractor import extract_facts, extract_session
from mindloop.memory import MemoryStore
from mindloop.summarizer import ChunkSummary

_EMB_A = np.array([1.0, 0.0], dtype=np.float32)


@contextmanager
def _patch_embeddings() -> Iterator[None]:
    """Patch get_embeddings to return uniform vectors."""

    def _get_embeddings(texts: list[str], **_kw: object) -> np.ndarray:
        return np.tile(_EMB_A, (len(texts), 1))

    with (
        patch("mindloop.memory.get_embeddings", side_effect=_get_embeddings),
        patch("mindloop.extractor.get_embeddings", side_effect=_get_embeddings),
    ):
        yield


def _mock_chat_returning(content: str) -> Any:
    """Return a mock chat that always returns the given content."""

    def _chat(messages: list[dict[str, Any]], **_kw: object) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    return _chat


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


# --- extract_facts tests ---


def test_extract_facts_parses_json() -> None:
    """Valid JSON array is parsed correctly."""
    facts_json = json.dumps([{"text": "Python is great", "abstract": "Python praise"}])
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning(facts_json)):
        result = extract_facts("some conversation")
    assert len(result) == 1
    assert result[0]["text"] == "Python is great"
    assert result[0]["abstract"] == "Python praise"


def test_extract_facts_empty_array() -> None:
    """Empty JSON array returns empty list."""
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning("[]")):
        result = extract_facts("boring conversation")
    assert result == []


def test_extract_facts_malformed_json() -> None:
    """Malformed JSON returns empty list without crashing."""
    with patch(
        "mindloop.extractor.chat", side_effect=_mock_chat_returning("not valid json{")
    ):
        result = extract_facts("some text")
    assert result == []


def test_extract_facts_with_context() -> None:
    """Context prefix is prepended to the user message."""
    calls: list[list[dict[str, Any]]] = []

    def _capturing_chat(
        messages: list[dict[str, Any]], **_kw: object
    ) -> dict[str, str]:
        calls.append(messages)
        return {"role": "assistant", "content": "[]"}

    with patch("mindloop.extractor.chat", side_effect=_capturing_chat):
        extract_facts("current chunk", context="previous tail")

    assert len(calls) == 1
    user_msg = calls[0][0]["content"]
    assert user_msg.startswith("Previous context: previous tail\n---\n")
    assert "current chunk" in user_msg


def test_extract_facts_filters_invalid_entries() -> None:
    """Entries missing required keys are filtered out."""
    facts_json = json.dumps(
        [
            {"text": "good", "abstract": "ok"},
            {"only_text": "bad"},
            "not a dict",
        ]
    )
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning(facts_json)):
        result = extract_facts("text")
    assert len(result) == 1
    assert result[0]["text"] == "good"


# --- extract_session tests ---


def test_extract_session_saves_facts(store: MemoryStore) -> None:
    """Facts are extracted and saved to the store."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Tell me about cats"},
        {"role": "assistant", "content": "Cats are great pets."},
    ]
    facts_json = json.dumps(
        [{"text": "Cats are great pets", "abstract": "Cats as pets"}]
    )

    with (
        _patch_embeddings(),
        patch("mindloop.extractor.chat", side_effect=_mock_chat_returning(facts_json)),
        patch(
            "mindloop.extractor.summarize_chunk",
            return_value=ChunkSummary(
                chunk=Chunk(),
                abstract="Cats as pets",
                summary="Cats are great pets.",
            ),
        ),
    ):
        saved = extract_session(messages, store, workers=1)

    assert saved == 1
    assert store.count() == 1


def test_extract_session_context_prefix(store: MemoryStore) -> None:
    """Chunk i gets tail of chunk i-1 as context."""
    # Create messages that produce at least 2 chunks.
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "A" * 200},
        {"role": "assistant", "content": "B" * 200},
        {"role": "user", "content": "\n\n" + "C" * 200},
        {"role": "assistant", "content": "D" * 200},
    ]

    extraction_calls: list[tuple[str, str | None]] = []

    def _tracking_extract(
        text: str, context: str | None = None, model: str | None = None
    ) -> list[dict[str, str]]:
        extraction_calls.append((text, context))
        return []

    with (
        _patch_embeddings(),
        patch("mindloop.extractor.extract_facts", side_effect=_tracking_extract),
    ):
        saved = extract_session(messages, store, workers=1)

    assert saved == 0
    # First chunk should have no context.
    assert extraction_calls[0][1] is None
    # Subsequent chunks should have context from the previous chunk.
    for i in range(1, len(extraction_calls)):
        assert extraction_calls[i][1] is not None


def test_extract_session_empty_log(store: MemoryStore) -> None:
    """Empty messages returns 0 without crashing."""
    saved = extract_session([], store, workers=1)
    assert saved == 0
