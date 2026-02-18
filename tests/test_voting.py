"""Tests for mindloop.voting."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from mindloop.client import Message
from mindloop.voting import (
    EXPERT_STRATEGIES,
    TEMPERATURE_SCALES,
    _DEFAULT_TEMPERATURE,
    chat_best_of_n,
    compression_score,
    majority_coherence,
    ngram_diversity,
)


# --- Scoring unit tests ---


def test_compression_score_normal() -> None:
    """Normal text has a higher compression ratio than repetitive text."""
    normal = "The quick brown fox jumps over the lazy dog near the river bank."
    repetitive = "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat"
    assert compression_score(normal) > compression_score(repetitive)


def test_compression_score_empty() -> None:
    """Empty text returns 1.0."""
    assert compression_score("") == 1.0


def test_ngram_diversity_varied() -> None:
    """Varied text has higher diversity than repetitive text."""
    varied = "one two three four five six seven eight nine ten"
    repetitive = "one two three one two three one two three one two three"
    assert ngram_diversity(varied) > ngram_diversity(repetitive)


def test_ngram_diversity_short() -> None:
    """Text shorter than n returns 1.0."""
    assert ngram_diversity("hi there", n=3) == 1.0


# --- Selector unit tests ---


def _make_tool_msg(name: str, args: dict[str, Any], content: str = "") -> Message:
    """Helper to build a tool-call response message."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def test_majority_coherence_tool_calls() -> None:
    """Majority of identical tool calls wins over a single outlier."""
    majority_msg = _make_tool_msg("read", {"path": "a.py"})
    outlier = _make_tool_msg("write", {"path": "b.py"})
    responses = [majority_msg, majority_msg, majority_msg, outlier]
    winner = majority_coherence(responses)
    assert winner in (0, 1, 2)


def test_majority_coherence_tiebreak() -> None:
    """Within majority, highest coherence wins."""
    # Same action but different content — coherence breaks the tie.
    low = _make_tool_msg("read", {"path": "a.py"}, content="cat " * 100)
    high = _make_tool_msg(
        "read",
        {"path": "a.py"},
        content="The function reads a configuration file and parses it.",
    )
    responses = [low, high]
    winner = majority_coherence(responses)
    assert winner == 1


def test_majority_coherence_text_responses() -> None:
    """Text-only responses cluster by content."""
    a: Message = {"role": "assistant", "content": "yes"}
    b: Message = {"role": "assistant", "content": "yes"}
    c: Message = {"role": "assistant", "content": "no"}
    winner = majority_coherence([a, b, c])
    assert winner in (0, 1)


# --- Integration tests ---


@patch("mindloop.voting.chat")
def test_passthrough_n1(mock_chat: MagicMock) -> None:
    """n=1 calls chat() once, passing through all kwargs as-is."""
    msg: Message = {"role": "assistant", "content": "hello"}
    mock_chat.return_value = msg

    result = chat_best_of_n(
        [{"role": "user", "content": "hi"}], model="m", n=1, stream=True
    )
    assert result == msg
    mock_chat.assert_called_once()
    assert mock_chat.call_args.kwargs.get("stream") is True


@patch("mindloop.voting.chat")
def test_parallel_n4(mock_chat: MagicMock) -> None:
    """n=4 calls chat() 4 times with different strategy suffixes."""
    msg: Message = {"role": "assistant", "content": "ok"}
    mock_chat.return_value = msg

    result = chat_best_of_n(
        [{"role": "user", "content": "hi"}],
        model="m",
        n=4,
        system_prompt="You are helpful.",
    )
    assert result == msg
    assert mock_chat.call_count == 4

    # Each call gets a strategy suffix from the first 4 strategies.
    prompts = [
        c.kwargs.get("system_prompt", c.args[2] if len(c.args) > 2 else None)
        for c in mock_chat.call_args_list
    ]
    for p in prompts:
        assert p is not None
        assert p.startswith("You are helpful")
    # The 4 prompts should use the first 4 strategies (round-robin).
    for i, strategy in enumerate(EXPERT_STRATEGIES[:4]):
        if strategy:
            assert strategy in prompts[i]


@patch("mindloop.voting.chat")
def test_custom_select_fn(mock_chat: MagicMock) -> None:
    """Custom selector that always picks index 0."""
    msgs = [{"role": "assistant", "content": f"response {i}"} for i in range(3)]
    mock_chat.side_effect = msgs

    result = chat_best_of_n(
        [{"role": "user", "content": "hi"}],
        model="m",
        n=3,
        select_fn=lambda rs: 0,
    )
    assert result["content"] == "response 0"


@patch("mindloop.voting.chat")
def test_temperature_scales_default(mock_chat: MagicMock) -> None:
    """n=16: first 8 at base*1.0, next 8 at base*1.3."""
    msg: Message = {"role": "assistant", "content": "ok"}
    mock_chat.return_value = msg

    chat_best_of_n([{"role": "user", "content": "hi"}], model="m", n=16)
    assert mock_chat.call_count == 16

    temps = [c.kwargs.get("temperature") for c in mock_chat.call_args_list]
    base = _DEFAULT_TEMPERATURE
    assert temps[:8] == [base * TEMPERATURE_SCALES[0]] * 8
    assert temps[8:] == [base * TEMPERATURE_SCALES[1]] * 8


@patch("mindloop.voting.chat")
def test_temperature_scales_custom(mock_chat: MagicMock) -> None:
    """Caller's temperature is used as base for scaling."""
    msg: Message = {"role": "assistant", "content": "ok"}
    mock_chat.return_value = msg

    chat_best_of_n([{"role": "user", "content": "hi"}], model="m", n=8, temperature=0.5)
    temps = [c.kwargs.get("temperature") for c in mock_chat.call_args_list]
    assert all(t == 0.5 * TEMPERATURE_SCALES[0] for t in temps)


@patch("mindloop.voting.chat")
def test_streaming_disabled(mock_chat: MagicMock) -> None:
    """All parallel calls use stream=False."""
    msg: Message = {"role": "assistant", "content": "ok"}
    mock_chat.return_value = msg

    chat_best_of_n([{"role": "user", "content": "hi"}], model="m", n=3)
    for c in mock_chat.call_args_list:
        assert c.kwargs.get("stream") is False
