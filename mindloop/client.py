"""OpenRouter API client for chat and embeddings."""

import json
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import requests

API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "openrouter/free"
DEFAULT_EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"

Message = dict[str, Any]
Tool = dict[str, Any]

# 1D embedding vector, shape (dim,), dtype float32.
Embedding = np.ndarray
# 2D embedding matrix, shape (n, dim), dtype float32.
Embeddings = np.ndarray


_TRANSIENT_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


def _with_retry(
    func: Callable[..., Any],
    on_token: Callable[[str], None],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call *func* with retries on transient network errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except _TRANSIENT_ERRORS:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _RETRY_BACKOFF * (attempt + 1)
            on_token(f"\n[connection error, retrying in {wait:.0f}s...]")
            time.sleep(wait)
    return None  # Unreachable, satisfies type checker.


def _default_on_token(token: str) -> None:
    print(token, end="", flush=True)


_ANTHROPIC_PREFIXES = ("anthropic/", "claude")
_CACHE_CONTROL = {"type": "ephemeral"}


def _needs_cache_control(model: str) -> bool:
    """Check if the model requires explicit cache_control breakpoints."""
    return model.startswith(_ANTHROPIC_PREFIXES)


def _to_multipart(content: Any) -> list[dict[str, Any]]:
    """Ensure message content is in multipart (list-of-blocks) format."""
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _apply_cache_control(
    payload: dict[str, Any], *, cache_messages: bool = True
) -> None:
    """Inject Anthropic cache_control breakpoints into a payload.

    Adds up to 3 breakpoints (of 4 allowed):
      1. Last tool definition — caches all tool schemas.
      2. System message — caches the system prompt.
      3. Last message — caches the conversation history prefix
         (only when *cache_messages* is True).
    """
    # 1. Last tool definition.
    tools = payload.get("tools")
    if tools:
        tools[-1]["cache_control"] = _CACHE_CONTROL

    # 2. System message (first message if role=system).
    msgs = payload.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = _to_multipart(msgs[0]["content"])
        msgs[0]["content"][-1]["cache_control"] = _CACHE_CONTROL

    # 3. Last message — caches the entire conversation prefix.
    if cache_messages and len(msgs) > 1:
        last = dict(msgs[-1])  # Copy to avoid mutating caller's data.
        last["content"] = _to_multipart(last["content"])
        last["content"][-1]["cache_control"] = _CACHE_CONTROL
        msgs[-1] = last


def _stream_request(
    payload: dict[str, Any],
    on_token: Callable[[str], None],
    on_thinking: Callable[[str], None] | None,
) -> Message:
    """Execute a streaming chat request and assemble the response."""
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        stream=True,
    )
    response.raise_for_status()

    full_reply: list[str] = []
    full_reasoning: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] | None = None

    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if "usage" in chunk:
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        # Accumulate reasoning tokens.
        for detail in delta.get("reasoning_details", []):
            text = detail.get("text", "")
            if text:
                if on_thinking is not None:
                    on_thinking(text)
                full_reasoning.append(text)

        # Accumulate content tokens.
        token = delta.get("content", "")
        if token:
            on_token(token)
            full_reply.append(token)

        # Accumulate tool call deltas.
        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta["index"]
            if idx not in tool_calls_by_index:
                tool_calls_by_index[idx] = {
                    "id": tc_delta.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            tc = tool_calls_by_index[idx]
            func_delta = tc_delta.get("function", {})
            if func_delta.get("name"):
                tc["function"]["name"] += func_delta["name"]
            if func_delta.get("arguments"):
                tc["function"]["arguments"] += func_delta["arguments"]

    result: Message = {"role": "assistant", "content": "".join(full_reply)}
    if full_reasoning:
        result["reasoning"] = "".join(full_reasoning)
    if tool_calls_by_index:
        result["tool_calls"] = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]
    if usage is not None:
        result["usage"] = usage
    return result


def chat(
    messages: list[Message],
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
    stream: bool = True,
    on_token: Callable[[str], None] = _default_on_token,
    on_thinking: Callable[[str], None] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    cache_messages: bool = True,
) -> Message:
    """Send a chat completion request. Returns the full response message dict."""
    full_messages = list(messages)
    if system_prompt is not None:
        full_messages.insert(0, {"role": "system", "content": system_prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
    }
    if tools:
        payload["tools"] = [dict(t) for t in tools]  # Shallow copy to avoid mutation.
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort is not None:
        payload["reasoning"] = {"enabled": True, "effort": reasoning_effort}

    if _needs_cache_control(model):
        _apply_cache_control(payload, cache_messages=cache_messages)

    if not stream:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        msg: Message = body["choices"][0]["message"]
        if "usage" in body:
            msg["usage"] = body["usage"]
        return msg

    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    result: Message = _with_retry(
        _stream_request, on_token, payload, on_token, on_thinking
    )
    return result


# Cache: (model, text) -> embedding vector.
_embedding_cache: dict[tuple[str, str], Embedding] = {}


def get_embeddings(
    texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL
) -> Embeddings:
    """Fetch embeddings with per-text caching. Returns (n, dim) float32 ndarray."""
    results: dict[int, Embedding] = {}
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        key = (model, text)
        if key in _embedding_cache:
            results[i] = _embedding_cache[key]
        else:
            uncached.append((i, text))

    if uncached:
        response = requests.post(
            f"{BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": model, "input": [text for _, text in uncached]},
        )
        response.raise_for_status()
        data = response.json()["data"]
        # Sort by index to match input order within the batch.
        sorted_data = sorted(data, key=lambda x: x["index"])
        for (orig_idx, text), item in zip(uncached, sorted_data):
            vec: Embedding = np.array(item["embedding"], dtype=np.float32)
            _embedding_cache[(model, text)] = vec
            results[orig_idx] = vec

    return np.stack([results[i] for i in range(len(texts))])
