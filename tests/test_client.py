"""Tests for mindloop.client."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from mindloop.client import Stats, _embedding_cache, _record_usage, chat, get_embeddings


def _mock_non_streaming(message: dict[str, Any]) -> MagicMock:
    """Create a mock response for non-streaming chat."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": message}]}
    return mock_resp


def _mock_streaming(tokens: list[str]) -> MagicMock:
    """Create a mock response for streaming chat."""
    lines: list[bytes] = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        lines.append(b"data: " + json.dumps(chunk).encode())
    lines.append(b"data: [DONE]")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines
    return mock_resp


def _mock_embeddings(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock response for embeddings."""
    data = [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": data}
    return mock_resp


# --- chat non-streaming ---


@patch("mindloop.client.requests.post")
def test_chat_non_streaming(mock_post: MagicMock) -> None:
    message = {"role": "assistant", "content": "hello"}
    mock_post.return_value = _mock_non_streaming(message)

    result = chat([{"role": "user", "content": "hi"}], model="test-model", stream=False)
    assert result == message
    mock_post.assert_called_once()


@patch("mindloop.client.requests.post")
def test_chat_non_streaming_with_system_prompt(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_non_streaming({"role": "assistant", "content": "ok"})

    chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        system_prompt="be nice",
        stream=False,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["messages"][0] == {"role": "system", "content": "be nice"}


@patch("mindloop.client.requests.post")
def test_chat_non_streaming_with_tools(mock_post: MagicMock) -> None:
    tool_call_msg = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "function": {"name": "ls", "arguments": '{"path": "."}'}}
        ],
    }
    mock_post.return_value = _mock_non_streaming(tool_call_msg)

    tools = [{"type": "function", "function": {"name": "ls"}}]
    result = chat(
        [{"role": "user", "content": "list files"}],
        model="test-model",
        tools=tools,
        stream=False,
    )
    assert result.get("tool_calls") is not None
    assert result["tool_calls"][0]["function"]["name"] == "ls"


@patch("mindloop.client.requests.post")
def test_chat_non_streaming_temperature_and_seed(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_non_streaming({"role": "assistant", "content": "ok"})

    chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=False,
        temperature=0,
        seed=42,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["temperature"] == 0
    assert payload["seed"] == 42


# --- chat streaming ---


@patch("mindloop.client.requests.post")
def test_chat_streaming(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_streaming(["hel", "lo", " world"])

    collected: list[str] = []
    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=True,
        on_token=lambda t: collected.append(t),
    )
    assert result == {"role": "assistant", "content": "hello world"}
    assert collected == ["hel", "lo", " world"]


@patch("mindloop.client.requests.post")
def test_chat_streaming_empty(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_streaming([])

    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        on_token=lambda t: None,
    )
    assert result == {"role": "assistant", "content": ""}


@patch("mindloop.client.requests.post")
def test_chat_streaming_tool_calls_with_null_fields(mock_post: MagicMock) -> None:
    """Tool call deltas with None name/arguments don't crash."""
    lines: list[bytes] = []
    # First delta: id and name.
    lines.append(
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "ls", "arguments": None},
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()
    )
    # Second delta: arguments chunk.
    lines.append(
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": None, "arguments": '{"path"'},
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()
    )
    # Third delta: rest of arguments.
    lines.append(
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ': "."}'}}
                            ]
                        }
                    }
                ]
            }
        ).encode()
    )
    lines.append(b"data: [DONE]")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines
    mock_post.return_value = mock_resp

    result = chat(
        [{"role": "user", "content": "list files"}],
        model="test-model",
        stream=True,
        on_token=lambda t: None,
    )
    assert result["tool_calls"] is not None
    tc = result["tool_calls"][0]
    assert tc["function"]["name"] == "ls"
    assert tc["function"]["arguments"] == '{"path": "."}'


# --- get_embeddings ---


@patch("mindloop.client.requests.post")
def test_get_embeddings(mock_post: MagicMock) -> None:
    _embedding_cache.clear()
    mock_post.return_value = _mock_embeddings([[0.1, 0.2], [0.3, 0.4]])

    result = get_embeddings(["hello", "world"], model="test-model")
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], [0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(result[1], [0.3, 0.4], atol=1e-6)
    mock_post.assert_called_once()


@patch("mindloop.client.requests.post")
def test_get_embeddings_caching(mock_post: MagicMock) -> None:
    _embedding_cache.clear()
    mock_post.return_value = _mock_embeddings([[0.1, 0.2]])

    # First call — hits API.
    get_embeddings(["hello"], model="test-model")
    assert mock_post.call_count == 1

    # Second call — cached.
    result = get_embeddings(["hello"], model="test-model")
    assert mock_post.call_count == 1
    np.testing.assert_allclose(result[0], [0.1, 0.2], atol=1e-6)


@patch("mindloop.client.requests.post")
def test_get_embeddings_partial_cache(mock_post: MagicMock) -> None:
    _embedding_cache.clear()

    # Prime cache with "hello".
    mock_post.return_value = _mock_embeddings([[0.1, 0.2]])
    get_embeddings(["hello"], model="test-model")

    # Request "hello" + "world" — only "world" should hit API.
    mock_post.return_value = _mock_embeddings([[0.3, 0.4]])
    result = get_embeddings(["hello", "world"], model="test-model")
    np.testing.assert_allclose(result[0], [0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(result[1], [0.3, 0.4], atol=1e-6)
    # Second call sent only 1 text.
    sent_input = mock_post.call_args.kwargs["json"]["input"]
    assert sent_input == ["world"]


# --- timeout retries ---


@patch("mindloop.client.time.sleep")
@patch("mindloop.client.requests.post")
def test_chat_streaming_retries_on_timeout(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Streaming chat retries after ReadTimeout and succeeds."""
    mock_post.side_effect = [
        requests.exceptions.ReadTimeout("timed out"),
        _mock_streaming(["hello"]),
    ]
    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=True,
        on_token=lambda t: None,
    )
    assert result["content"] == "hello"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("mindloop.client.time.sleep")
@patch("mindloop.client.requests.post")
def test_chat_non_streaming_retries_on_timeout(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Non-streaming chat retries after ConnectTimeout and succeeds."""
    mock_post.side_effect = [
        requests.exceptions.ConnectTimeout("timed out"),
        _mock_non_streaming({"role": "assistant", "content": "ok"}),
    ]
    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=False,
        on_token=lambda t: None,
    )
    assert result["content"] == "ok"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("mindloop.client.time.sleep")
@patch("mindloop.client.requests.post")
def test_get_embeddings_retries_on_timeout(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """get_embeddings retries after ReadTimeout and succeeds."""
    _embedding_cache.clear()
    mock_post.side_effect = [
        requests.exceptions.ReadTimeout("timed out"),
        _mock_embeddings([[0.1, 0.2]]),
    ]
    result = get_embeddings(["hello"], model="test-model")
    np.testing.assert_allclose(result[0], [0.1, 0.2], atol=1e-6)
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("mindloop.client.time.sleep")
@patch("mindloop.client.requests.post")
def test_retry_exhausted_raises(mock_post: MagicMock, mock_sleep: MagicMock) -> None:
    """All retries exhausted — exception propagates."""
    mock_post.side_effect = requests.exceptions.ReadTimeout("timed out")
    try:
        chat(
            [{"role": "user", "content": "hi"}],
            model="test-model",
            stream=True,
            on_token=lambda t: None,
        )
        assert False, "Expected ReadTimeout"
    except requests.exceptions.ReadTimeout:
        pass
    assert mock_post.call_count == 4


@patch("mindloop.client.time.sleep")
@patch("mindloop.client.requests.post")
def test_keyboard_interrupt_retries(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """KeyboardInterrupt during request retries without backoff."""
    mock_post.side_effect = [
        KeyboardInterrupt(),
        _mock_streaming(["hello"]),
    ]
    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=True,
        on_token=lambda t: None,
    )
    assert result["content"] == "hello"
    assert mock_post.call_count == 2
    mock_sleep.assert_not_called()


# --- max_tokens ---


@patch("mindloop.client.requests.post")
def test_chat_max_tokens_in_payload(mock_post: MagicMock) -> None:
    """max_tokens is included in the API payload when set."""
    mock_post.return_value = _mock_non_streaming({"role": "assistant", "content": "ok"})
    chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=False,
        max_tokens=4096,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 4096


@patch("mindloop.client.requests.post")
def test_chat_max_tokens_absent_when_none(mock_post: MagicMock) -> None:
    """max_tokens is omitted from the payload when not set."""
    mock_post.return_value = _mock_non_streaming({"role": "assistant", "content": "ok"})
    chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=False,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert "max_tokens" not in payload


# --- coherence detection ---


@patch("mindloop.client.requests.post")
def test_coherence_aborts_degenerate_stream(mock_post: MagicMock) -> None:
    """Streaming aborts early when content is degenerate repetition."""
    # ~50 bytes per phrase, 200 repetitions = 10KB. Should trigger well before end.
    phrase = "the cat sat on the mat and looked at the door. "
    tokens = [phrase] * 200
    mock_post.return_value = _mock_streaming(tokens)

    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=True,
        on_token=lambda t: None,
    )
    # Should have been truncated well before all 200 repetitions.
    assert len(result["content"]) < len(phrase) * 200


@patch("mindloop.client.requests.post")
def test_coherence_allows_normal_text(mock_post: MagicMock) -> None:
    """Normal varied text is not aborted by coherence detection."""
    # Diverse sentences that compress poorly (high entropy).
    tokens = [
        "The quick brown fox jumps over the lazy dog. ",
        "Meanwhile, scientists discovered a new species in the Amazon. ",
        "Financial markets responded positively to the latest earnings report. ",
        "A chef in Tokyo created an innovative fusion dish last night. ",
        "The orchestra performed Beethoven's ninth symphony to a packed house. ",
    ]
    mock_post.return_value = _mock_streaming(tokens)

    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=True,
        on_token=lambda t: None,
    )
    assert result["content"] == "".join(tokens)


def _mock_streaming_reasoning(reasoning_tokens: list[str]) -> MagicMock:
    """Create a mock streaming response with reasoning_details deltas."""
    lines: list[bytes] = []
    for token in reasoning_tokens:
        reason_chunk: dict[str, Any] = {
            "choices": [{"delta": {"reasoning_details": [{"text": token}]}}]
        }
        lines.append(b"data: " + json.dumps(reason_chunk).encode())
    # End with a content token so the response has content.
    content_chunk: dict[str, Any] = {"choices": [{"delta": {"content": "done"}}]}
    lines.append(b"data: " + json.dumps(content_chunk).encode())
    lines.append(b"data: [DONE]")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines
    return mock_resp


@patch("mindloop.client.requests.post")
def test_coherence_aborts_degenerate_reasoning(mock_post: MagicMock) -> None:
    """Streaming aborts early when reasoning tokens are degenerate."""
    phrase = "Let me reconsider this problem from another angle. "
    mock_post.return_value = _mock_streaming_reasoning([phrase] * 200)

    result = chat(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        stream=True,
        on_token=lambda t: None,
    )
    # Reasoning should be truncated, and content token never reached.
    reasoning = result.get("reasoning", "")
    assert len(reasoning) < len(phrase) * 200
    assert result["content"] == ""


# --- Stats tests ---


def test_stats_captures_delta() -> None:
    """Stats.counters returns delta, not absolute total."""
    # Record some usage before the context to prove delta isolation.
    _record_usage("pre-model", {"total_tokens": 999})

    with Stats() as s:
        _record_usage("model-a", {"total_tokens": 100, "cost": 0.01})
        _record_usage("model-b", {"total_tokens": 50, "cost": 0.005})
        c = s.counters
    assert c.tokens == 150
    assert abs(c.cost - 0.015) < 1e-9


def test_stats_per_model() -> None:
    """per_model returns per-model deltas."""
    with Stats() as s:
        _record_usage("fast", {"total_tokens": 200, "cost": 0.02})
        _record_usage("slow", {"total_tokens": 80, "cost": 0.10})
        pm = s.per_model
    assert pm["fast"].tokens == 200
    assert pm["slow"].tokens == 80
    assert abs(pm["slow"].cost - 0.10) < 1e-9


def test_stats_nesting() -> None:
    """Inner Stats scope sees only its own usage."""
    with Stats() as outer:
        _record_usage("m", {"total_tokens": 100})
        with Stats() as inner:
            _record_usage("m", {"total_tokens": 50})
            assert inner.counters.tokens == 50
        assert outer.counters.tokens == 150


def test_stats_no_cost() -> None:
    """Cost stays zero when API omits it."""
    with Stats() as s:
        _record_usage("m", {"total_tokens": 42})
        assert s.counters.cost == 0.0
        assert s.counters.tokens == 42
