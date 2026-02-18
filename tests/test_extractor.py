"""Tests for mindloop.extractor."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.extractor import (
    extract_facts,
    extract_session,
    extract_window,
    verify_fact,
    verify_facts,
)
from mindloop.memory import MemoryStore

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
        result = extract_facts("some conversation", model="test-model")
    assert len(result) == 1
    assert result[0]["text"] == "Python is great"
    assert result[0]["abstract"] == "Python praise"


def test_extract_facts_empty_array() -> None:
    """Empty JSON array returns empty list."""
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning("[]")):
        result = extract_facts("boring conversation", model="test-model")
    assert result == []


def test_extract_facts_malformed_json() -> None:
    """Malformed JSON on both attempts returns empty list."""
    with patch(
        "mindloop.extractor.chat", side_effect=_mock_chat_returning("not valid json{")
    ):
        result = extract_facts("some text", model="test-model")
    assert result == []


def test_extract_facts_strips_markdown_fences() -> None:
    """JSON wrapped in markdown fences is parsed correctly."""
    fenced = '```json\n[{"text": "fact", "abstract": "abs"}]\n```'
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning(fenced)):
        result = extract_facts("some text", model="test-model")
    assert len(result) == 1
    assert result[0]["text"] == "fact"


def test_extract_facts_retries_on_malformed_json() -> None:
    """Malformed first response triggers retry; valid second response succeeds."""
    good_json = json.dumps([{"text": "recovered", "abstract": "abs"}])
    call_count = 0

    def _failing_then_ok(
        messages: list[dict[str, Any]], **_kw: object
    ) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"role": "assistant", "content": "oops not json"}
        return {"role": "assistant", "content": good_json}

    with patch("mindloop.extractor.chat", side_effect=_failing_then_ok):
        result = extract_facts("some text", model="test-model")

    assert call_count == 2
    assert len(result) == 1
    assert result[0]["text"] == "recovered"


def test_extract_facts_retry_sends_bad_output_back() -> None:
    """Retry includes the failed response in the conversation for correction."""
    calls: list[list[dict[str, Any]]] = []

    def _capturing(messages: list[dict[str, Any]], **_kw: object) -> dict[str, str]:
        calls.append(list(messages))
        return {"role": "assistant", "content": "garbage"}

    with patch("mindloop.extractor.chat", side_effect=_capturing):
        extract_facts("input text", model="test-model")

    assert len(calls) == 2
    # Retry messages should include the original user message, the bad assistant
    # response, and a correction request.
    retry_msgs = calls[1]
    assert retry_msgs[0]["role"] == "user"
    assert retry_msgs[1] == {"role": "assistant", "content": "garbage"}
    assert "not valid JSON" in retry_msgs[2]["content"]


def test_extract_facts_with_context() -> None:
    """Context prefix is prepended to the user message."""
    calls: list[list[dict[str, Any]]] = []

    def _capturing_chat(
        messages: list[dict[str, Any]], **_kw: object
    ) -> dict[str, str]:
        calls.append(messages)
        return {"role": "assistant", "content": "[]"}

    with patch("mindloop.extractor.chat", side_effect=_capturing_chat):
        extract_facts("current chunk", context="previous tail", model="test-model")

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
        result = extract_facts("text", model="test-model")
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
        [
            {
                "text": "Cats are great pets",
                "abstract": "Cats as pets",
                "summary": "Cats are great pets.",
            }
        ]
    )

    with (
        _patch_embeddings(),
        patch("mindloop.extractor.chat", side_effect=_mock_chat_returning(facts_json)),
    ):
        saved = extract_session(messages, store, model="test-model", workers=1)

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
        text: str, context: str | None = None, model: str = ""
    ) -> list[dict[str, str]]:
        extraction_calls.append((text, context))
        return []

    with (
        _patch_embeddings(),
        patch("mindloop.extractor.extract_facts", side_effect=_tracking_extract),
    ):
        saved = extract_session(messages, store, model="test-model", workers=1)

    assert saved == 0
    # First chunk should have no context.
    assert extraction_calls[0][1] is None
    # Subsequent chunks should have context from the previous chunk.
    for i in range(1, len(extraction_calls)):
        assert extraction_calls[i][1] is not None


def test_extract_session_empty_log(store: MemoryStore) -> None:
    """Empty messages returns 0 without crashing."""
    saved = extract_session([], store, model="test-model", workers=1)
    assert saved == 0


# --- extract_window tests ---


def test_extract_window_returns_facts() -> None:
    """extract_window collapses messages and returns facts."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Cats are pets"},
        {"role": "assistant", "content": "Indeed they are."},
    ]
    facts_json = json.dumps([{"text": "Cats are pets", "abstract": "Cats"}])

    calls: list[str] = []

    def _capturing_chat(msgs: list[dict[str, Any]], **_kw: object) -> dict[str, str]:
        calls.append(msgs[0]["content"])
        return {"role": "assistant", "content": facts_json}

    with patch("mindloop.extractor.chat", side_effect=_capturing_chat):
        result = extract_window(messages, model="test-model")

    assert len(result) == 1
    assert result[0]["text"] == "Cats are pets"
    # Verify the collapsed text was sent to the LLM.
    assert "Cats are pets" in calls[0]


def test_extract_window_empty_messages() -> None:
    """Empty messages returns empty list without LLM call."""
    result = extract_window([], model="test-model")
    assert result == []


# --- verify_fact / verify_facts tests ---

_SAMPLE_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "I prefer dark mode."},
    {"role": "assistant", "content": "Noted, switching to dark mode."},
]

_SAMPLE_FACT: dict[str, str] = {
    "text": "User prefers dark mode",
    "abstract": "Dark mode preference",
}


def test_verify_fact_yes() -> None:
    """Chat returning 'yes' means the fact is verified."""
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning("yes")):
        assert verify_fact(_SAMPLE_FACT, _SAMPLE_MESSAGES, model="test") is True


def test_verify_fact_no() -> None:
    """Chat returning 'no' means the fact is rejected."""
    with patch("mindloop.extractor.chat", side_effect=_mock_chat_returning("no")):
        assert verify_fact(_SAMPLE_FACT, _SAMPLE_MESSAGES, model="test") is False


def test_verify_fact_ambiguous() -> None:
    """Ambiguous response is treated as rejection (fail-closed)."""
    with patch(
        "mindloop.extractor.chat", side_effect=_mock_chat_returning("I'm not sure")
    ):
        assert verify_fact(_SAMPLE_FACT, _SAMPLE_MESSAGES, model="test") is False


def test_verify_facts_filters() -> None:
    """verify_facts keeps only confirmed facts."""
    facts = [
        {"text": "fact A", "abstract": "A"},
        {"text": "fact B", "abstract": "B"},
        {"text": "fact C", "abstract": "C"},
    ]

    def _selective_chat(
        messages: list[dict[str, Any]], **_kw: object
    ) -> dict[str, str]:
        content = messages[-1]["content"]
        if "fact B" in content:
            return {"role": "assistant", "content": "no"}
        return {"role": "assistant", "content": "yes"}

    with patch("mindloop.extractor.chat", side_effect=_selective_chat):
        result = verify_facts(facts, _SAMPLE_MESSAGES, model="test", workers=1)

    texts = {f["text"] for f in result}
    assert texts == {"fact A", "fact C"}


def test_verify_facts_empty() -> None:
    """Empty facts list returns empty without LLM calls."""
    result = verify_facts([], _SAMPLE_MESSAGES, model="test")
    assert result == []
