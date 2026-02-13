"""Tests for mindloop.summarizer."""

from datetime import datetime
from typing import Any
from unittest.mock import patch

from mindloop.chunker import Chunk, Turn
from mindloop.summarizer import summarize_chunk, summarize_chunks


def _make_chunk(text: str) -> Chunk:
    return Chunk(turns=[Turn(timestamp=datetime(2025, 1, 1), role="You", text=text)])


def _mock_chat(messages: list[dict[str, str]], **_kw: Any) -> dict[str, str]:
    """Return a valid ABSTRACT/SUMMARY response keyed on user content."""
    user_text = messages[0]["content"]
    return {
        "content": f"ABSTRACT: abs of {user_text}\nSUMMARY: sum of {user_text}",
    }


@patch("mindloop.summarizer.chat", side_effect=_mock_chat)
def test_summarize_chunk_parses_response(_mock: Any) -> None:
    """Single chunk produces correct abstract and summary."""
    cs = summarize_chunk(_make_chunk("hello"))
    assert cs.abstract == "abs of You: hello"
    assert cs.summary == "sum of You: hello"


@patch("mindloop.summarizer.chat", side_effect=_mock_chat)
def test_summarize_chunks_sequential(_mock: Any) -> None:
    """Sequential summarization preserves order."""
    chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
    results = summarize_chunks(chunks, workers=1)
    assert len(results) == 3
    assert [r.abstract for r in results] == [
        "abs of You: a",
        "abs of You: b",
        "abs of You: c",
    ]


@patch("mindloop.summarizer.chat", side_effect=_mock_chat)
def test_summarize_chunks_parallel(_mock: Any) -> None:
    """Parallel summarization returns results in original order."""
    chunks = [_make_chunk("x"), _make_chunk("y"), _make_chunk("z")]
    results = summarize_chunks(chunks, workers=2)
    assert len(results) == 3
    assert [r.abstract for r in results] == [
        "abs of You: x",
        "abs of You: y",
        "abs of You: z",
    ]


@patch("mindloop.summarizer.chat", side_effect=_mock_chat)
def test_summarize_chunks_parallel_logs_progress(_mock: Any) -> None:
    """Parallel path calls the log function for each completed chunk."""
    logged: list[str] = []
    chunks = [_make_chunk("p"), _make_chunk("q")]
    summarize_chunks(chunks, workers=2, log=logged.append)
    assert len(logged) == 2


@patch("mindloop.summarizer.chat", return_value={"content": "garbage"})
def test_summarize_chunk_parse_error(_mock: Any) -> None:
    """Unparseable response returns parse-error abstract."""
    cs = summarize_chunk(_make_chunk("bad"))
    assert cs.abstract == "(parse error)"
    assert cs.summary == "garbage"
