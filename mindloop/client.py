"""OpenRouter API client for chat and embeddings."""

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import requests

API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "openrouter/free"
DEFAULT_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

Message = dict[str, Any]
Tool = dict[str, Any]

# 1D embedding vector, shape (dim,), dtype float32.
Embedding = np.ndarray
# 2D embedding matrix, shape (n, dim), dtype float32.
Embeddings = np.ndarray


def _default_on_token(token: str) -> None:
    print(token, end="", flush=True)


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
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort is not None:
        payload["reasoning"] = {"enabled": True, "effort": reasoning_effort}

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
