"""Tests for mindloop.values."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from mindloop.values import (
    _MIN_DISPOSITIONS,
    append_dispositions,
    distill_values,
    extract_core_principles,
    extract_dispositions,
    extract_dispositions_window,
    load_values,
    save_values,
)


def _mock_chat_response(content: str) -> dict[str, str]:
    """Create a mock chat response with given content."""
    return {"role": "assistant", "content": content}


# --- extract_dispositions ---


@patch("mindloop.values.chat")
def test_extract_dispositions_parses_json(mock_chat: MagicMock) -> None:
    """Dispositions are extracted from valid JSON response."""
    dispositions = [
        {"text": "I tend to investigate before committing.", "abstract": "Cautious."},
        {"text": "I prefer depth over breadth.", "abstract": "Depth-first."},
    ]
    mock_chat.return_value = _mock_chat_response(json.dumps(dispositions))

    result = extract_dispositions("Bot: Let me dig deeper into this.", model="test")
    assert len(result) == 2
    assert result[0]["abstract"] == "Cautious."
    mock_chat.assert_called_once()


@patch("mindloop.values.chat")
def test_extract_dispositions_empty_array(mock_chat: MagicMock) -> None:
    """Empty array is a valid response (most windows have no dispositions)."""
    mock_chat.return_value = _mock_chat_response("[]")

    result = extract_dispositions("Bot: Read the file.", model="test")
    assert result == []


@patch("mindloop.values.chat")
def test_extract_dispositions_retries_bad_json(mock_chat: MagicMock) -> None:
    """Retries once when first response is malformed."""
    mock_chat.side_effect = [
        _mock_chat_response("not json"),
        _mock_chat_response(
            '[{"text": "I question assumptions.", "abstract": "Skeptical."}]'
        ),
    ]

    result = extract_dispositions("Bot: Wait, let me reconsider.", model="test")
    assert len(result) == 1
    assert mock_chat.call_count == 2


@patch("mindloop.values.chat")
def test_extract_dispositions_strips_fences(mock_chat: MagicMock) -> None:
    """Markdown code fences around JSON are stripped."""
    fenced = '```json\n[{"text": "I reflect often.", "abstract": "Reflective."}]\n```'
    mock_chat.return_value = _mock_chat_response(fenced)

    result = extract_dispositions("Bot: Let me think about this.", model="test")
    assert len(result) == 1


# --- extract_dispositions_window ---


@patch("mindloop.values.chat")
def test_extract_dispositions_window(mock_chat: MagicMock) -> None:
    """Window extraction collapses messages and calls extract_dispositions."""
    mock_chat.return_value = _mock_chat_response("[]")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = extract_dispositions_window(messages, model="test")
    assert result == []
    mock_chat.assert_called_once()


@patch("mindloop.values.chat")
def test_extract_dispositions_window_empty(mock_chat: MagicMock) -> None:
    """Empty messages produce no extraction call."""
    result = extract_dispositions_window([], model="test")
    assert result == []
    mock_chat.assert_not_called()


# --- append_dispositions / save_values / load_values ---


def test_append_dispositions(tmp_path: Path) -> None:
    """Dispositions are appended as JSONL."""
    path = tmp_path / "_dispositions.jsonl"
    d1 = [{"text": "I'm cautious.", "abstract": "Cautious."}]
    d2 = [{"text": "I go deep.", "abstract": "Depth."}]

    append_dispositions(path, d1)
    append_dispositions(path, d2)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["abstract"] == "Cautious."
    assert json.loads(lines[1])["abstract"] == "Depth."


def test_append_dispositions_empty(tmp_path: Path) -> None:
    """Empty list does not create the file."""
    path = tmp_path / "_dispositions.jsonl"
    append_dispositions(path, [])
    assert not path.exists()


def test_save_and_load_values(tmp_path: Path) -> None:
    """Round-trip save/load."""
    path = tmp_path / "_values.md"
    save_values(path, "I tend to be cautious.")
    assert load_values(path) == "I tend to be cautious."


def test_load_values_missing(tmp_path: Path) -> None:
    """Missing file returns None."""
    assert load_values(tmp_path / "nope.md") is None


def test_load_values_empty(tmp_path: Path) -> None:
    """Empty file returns None."""
    path = tmp_path / "_values.md"
    path.write_text("")
    assert load_values(path) is None


# --- distill_values ---


@patch("mindloop.values.chat")
def test_distill_values_produces_text(mock_chat: MagicMock, tmp_path: Path) -> None:
    """Distillation produces values text from enough dispositions."""
    path = tmp_path / "_dispositions.jsonl"
    dispositions = [
        {"text": f"Disposition {i}.", "abstract": f"Abstract {i}."}
        for i in range(_MIN_DISPOSITIONS + 1)
    ]
    with open(path, "w") as f:
        for d in dispositions:
            f.write(json.dumps(d) + "\n")

    mock_chat.return_value = _mock_chat_response(
        "I tend to investigate thoroughly before making decisions."
    )

    result = distill_values(path, model="test")
    assert "investigate" in result
    mock_chat.assert_called_once()
    # Verify the user message contains the abstracts.
    user_msg = mock_chat.call_args.args[0][0]["content"]
    assert "Abstract 0." in user_msg


@patch("mindloop.values.chat")
def test_distill_values_too_few(mock_chat: MagicMock, tmp_path: Path) -> None:
    """Too few dispositions returns empty string without LLM call."""
    path = tmp_path / "_dispositions.jsonl"
    with open(path, "w") as f:
        for i in range(_MIN_DISPOSITIONS - 1):
            f.write(json.dumps({"text": f"D{i}.", "abstract": f"A{i}."}) + "\n")

    result = distill_values(path, model="test")
    assert result == ""
    mock_chat.assert_not_called()


@patch("mindloop.values.chat")
def test_distill_values_deduplicates(mock_chat: MagicMock, tmp_path: Path) -> None:
    """Duplicate abstracts are deduplicated before sending to LLM."""
    path = tmp_path / "_dispositions.jsonl"
    # Write many duplicates but only _MIN_DISPOSITIONS unique ones.
    with open(path, "w") as f:
        for i in range(_MIN_DISPOSITIONS):
            for _ in range(3):  # Each repeated 3 times.
                f.write(json.dumps({"text": f"D{i}.", "abstract": f"A{i}."}) + "\n")

    mock_chat.return_value = _mock_chat_response("Values distilled.")

    result = distill_values(path, model="test")
    assert result == "Values distilled."
    # Verify abstracts in user message are unique.
    user_msg = mock_chat.call_args.args[0][0]["content"]
    lines = [line for line in user_msg.strip().splitlines() if line.startswith("- ")]
    assert len(lines) == _MIN_DISPOSITIONS


def test_distill_values_missing_file(tmp_path: Path) -> None:
    """Missing dispositions file returns empty string."""
    result = distill_values(tmp_path / "nope.jsonl", model="test")
    assert result == ""


# --- extract_core_principles ---


@patch("mindloop.values.chat")
def test_extract_core_principles(mock_chat: MagicMock, tmp_path: Path) -> None:
    """Core principles extracted from sufficient dispositions."""
    path = tmp_path / "_dispositions.jsonl"
    dispositions = [
        {"text": f"Disposition {i}.", "abstract": f"Abstract {i}."}
        for i in range(_MIN_DISPOSITIONS + 1)
    ]
    with open(path, "w") as f:
        for d in dispositions:
            f.write(json.dumps(d) + "\n")

    principles = [
        {"text": "I practice restraint.", "abstract": "Restraint."},
        {"text": "I investigate before acting.", "abstract": "Investigation."},
    ]
    mock_chat.return_value = _mock_chat_response(json.dumps(principles))

    result = extract_core_principles(path, model="test")
    assert len(result) == 2
    assert result[0]["text"] == "I practice restraint."
    mock_chat.assert_called_once()


@patch("mindloop.values.chat")
def test_extract_core_principles_too_few(mock_chat: MagicMock, tmp_path: Path) -> None:
    """Too few dispositions returns empty list without LLM call."""
    path = tmp_path / "_dispositions.jsonl"
    with open(path, "w") as f:
        for i in range(_MIN_DISPOSITIONS - 1):
            f.write(json.dumps({"text": f"D{i}.", "abstract": f"A{i}."}) + "\n")

    result = extract_core_principles(path, model="test")
    assert result == []
    mock_chat.assert_not_called()


def test_extract_core_principles_missing_file(tmp_path: Path) -> None:
    """Missing file returns empty list."""
    result = extract_core_principles(tmp_path / "nope.jsonl", model="test")
    assert result == []
