"""OpenRouter API client for chat and embeddings."""

import json
import os
import threading
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import requests

from mindloop.util import noop

API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "openrouter/free"
DEFAULT_EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"

Message = dict[str, Any]
Tool = dict[str, Any]

# Kwargs for deterministic one-shot LLM calls (summarization, merging, extraction).
DETERMINISTIC_PARAMS: dict[str, Any] = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 4096,
}

# 1D embedding vector, shape (dim,), dtype float32.
Embedding = np.ndarray
# 2D embedding matrix, shape (n, dim), dtype float32.
Embeddings = np.ndarray


# --- Monotonic API usage counters (thread-safe, per-model). ---


@dataclass
class Counters:
    """API usage snapshot."""

    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


_counters_lock = threading.Lock()
_counters: defaultdict[str, Counters] = defaultdict(Counters)


def _record_usage(model: str, usage: dict[str, Any]) -> None:
    """Accumulate token and cost from an API response."""
    tokens = int(usage.get("total_tokens", 0))
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    cost = usage.get("cost")
    with _counters_lock:
        c = _counters[model]
        c.tokens += tokens
        c.input_tokens += input_tokens
        c.output_tokens += output_tokens
        if cost is not None:
            c.cost += float(cost)


def _estimate_tokens(messages: list[Message], response: Message) -> dict[str, int]:
    """Estimate token counts for a call from character counts."""
    from mindloop.util import CHARS_PER_TOKEN

    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    response_chars = len(str(response.get("content", "")))
    response_chars += len(str(response.get("reasoning", "")))
    prompt_tokens = prompt_chars // CHARS_PER_TOKEN
    completion_tokens = response_chars // CHARS_PER_TOKEN
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class Stats:
    """Context manager that captures API usage over a scope.

    Snapshots the monotonic per-model counters on entry.  The
    ``counters`` property returns an aggregate delta; ``by_model``
    returns per-model deltas.
    """

    def __init__(self) -> None:
        self._start: dict[str, Counters] = {}

    def __enter__(self) -> "Stats":
        with _counters_lock:
            self._start = {
                m: Counters(c.tokens, c.input_tokens, c.output_tokens, c.cost)
                for m, c in _counters.items()
            }
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # Nothing to clean up; counters keep growing.

    @property
    def counters(self) -> Counters:
        """Aggregate delta across all models since entering the context."""
        with _counters_lock:
            tok = sum(c.tokens for c in _counters.values()) - sum(
                c.tokens for c in self._start.values()
            )
            inp = sum(c.input_tokens for c in _counters.values()) - sum(
                c.input_tokens for c in self._start.values()
            )
            out = sum(c.output_tokens for c in _counters.values()) - sum(
                c.output_tokens for c in self._start.values()
            )
            cst = sum(c.cost for c in _counters.values()) - sum(
                c.cost for c in self._start.values()
            )
        return Counters(tokens=tok, input_tokens=inp, output_tokens=out, cost=cst)

    @property
    def per_model(self) -> defaultdict[str, Counters]:
        """Per-model deltas since entering the context."""
        with _counters_lock:
            result: defaultdict[str, Counters] = defaultdict(Counters)
            for model, current in _counters.items():
                start = self._start.get(model, Counters())
                delta = Counters(
                    tokens=current.tokens - start.tokens,
                    input_tokens=current.input_tokens - start.input_tokens,
                    output_tokens=current.output_tokens - start.output_tokens,
                    cost=current.cost - start.cost,
                )
                if delta.tokens or delta.cost:
                    result[model] = delta
            return result


_TRANSIENT_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.JSONDecodeError,
)
_MAX_RETRIES = 4
_RETRY_BACKOFF = 2.0
_REQUEST_TIMEOUT = 60

# Coherence detection via streaming zlib incremental compression ratio.
# The first window is warmup (zlib builds its dictionary). After that, each
# window's incremental ratio (new_compressed / new_raw) cleanly separates
# normal text (~0.09-0.12) from degenerate loops (~0.016-0.017).
_COHERENCE_WARMUP_BYTES = 2048
_COHERENCE_CHECK_INTERVAL = 1024
_COHERENCE_THRESHOLD = 0.05
_COHERENCE_STREAK = 3


class _CoherenceMonitor:
    """Detect degenerate repetition via streaming zlib incremental ratio."""

    def __init__(self) -> None:
        self._compressor = zlib.compressobj()
        self._raw = 0
        self._compressed = 0
        self._last_check = 0
        self._last_raw = 0
        self._last_compressed = 0
        self._warmed_up = False
        self._bad_streak = 0

    def feed(self, text: str) -> bool:
        """Feed text and return True if degeneration detected."""
        encoded = text.encode()
        self._raw += len(encoded)
        self._compressed += len(self._compressor.compress(encoded))
        if self._raw - self._last_check < _COHERENCE_CHECK_INTERVAL:
            return False
        self._compressed += len(self._compressor.flush(zlib.Z_SYNC_FLUSH))
        self._last_check = self._raw

        if not self._warmed_up:
            if self._raw >= _COHERENCE_WARMUP_BYTES:
                self._warmed_up = True
                self._last_raw = self._raw
                self._last_compressed = self._compressed
            return False

        # Incremental ratio for this window.
        d_raw = self._raw - self._last_raw
        d_compressed = self._compressed - self._last_compressed
        self._last_raw = self._raw
        self._last_compressed = self._compressed

        ratio = d_compressed / d_raw if d_raw > 0 else 1.0
        if ratio < _COHERENCE_THRESHOLD:
            self._bad_streak += 1
        else:
            self._bad_streak = 0
        return self._bad_streak >= _COHERENCE_STREAK


def _with_retry(
    func: Callable[..., Any],
    on_error: Callable[[str], None] | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call *func* with retries on transient network errors and KeyboardInterrupt."""
    _on_error = on_error or noop
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            if attempt == _MAX_RETRIES - 1:
                raise
            _on_error("\n[interrupted, retrying...]")
        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            if resp is not None and resp.status_code >= 500:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BACKOFF * (attempt + 1)
                _on_error(f"\n[server {resp.status_code}, retrying in {wait:.0f}s...]")
                time.sleep(wait)
            else:
                raise  # 4xx errors are not retryable.
        except _TRANSIENT_ERRORS:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _RETRY_BACKOFF * (attempt + 1)
            _on_error(f"\n[connection error, retrying in {wait:.0f}s...]")
            time.sleep(wait)
    return None  # Unreachable, satisfies type checker.


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
    on_token: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None,
) -> Message:
    """Execute a streaming chat request and assemble the response."""
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text}",
            response=response,
        )

    full_reply: list[str] = []
    full_reasoning: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    # Streaming coherence monitors for content and reasoning.
    content_monitor = _CoherenceMonitor()
    reasoning_monitor = _CoherenceMonitor()

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
        # OpenRouter returns reasoning in two formats depending on provider:
        #   - reasoning_details: list of {"text": ...} (DeepSeek, etc.)
        #   - reasoning_content: plain string (Anthropic)
        degenerate = False
        reasoning_texts: list[str] = []
        for detail in delta.get("reasoning_details", []):
            text = detail.get("text", "")
            if text:
                reasoning_texts.append(text)
        rc = delta.get("reasoning_content", "")
        if rc:
            reasoning_texts.append(rc)
        for text in reasoning_texts:
            if on_thinking is not None:
                on_thinking(text)
            full_reasoning.append(text)
            if reasoning_monitor.feed(text):
                degenerate = True
        if degenerate:
            break

        # Accumulate content tokens.
        token = delta.get("content", "")
        if token:
            if on_token is not None:
                on_token(token)
            full_reply.append(token)
            if content_monitor.feed(token):
                break

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
    model: str,
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
    stream: bool = True,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    cache_messages: bool = True,
    max_tokens: int | None = None,
) -> Message:
    """Send a chat completion request. Returns the full response message dict."""
    if not model:
        raise ValueError("model must be a non-empty string")
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
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        payload["reasoning"] = {"enabled": True, "effort": reasoning_effort}

    if _needs_cache_control(model):
        _apply_cache_control(payload, cache_messages=cache_messages)

    if not stream:

        def _non_streaming_request() -> Message:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            if response.status_code >= 400:
                raise requests.HTTPError(
                    f"{response.status_code} {response.reason}: {response.text}",
                    response=response,
                )
            body = response.json()
            msg: Message = body["choices"][0]["message"]
            if "usage" in body:
                msg["usage"] = body["usage"]
            return msg

        msg: Message = _with_retry(_non_streaming_request, on_error)
        if "usage" in msg:
            _record_usage(model, msg["usage"])
        else:
            _record_usage(model, _estimate_tokens(full_messages, msg))
        return msg

    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    result: Message = _with_retry(
        _stream_request, on_error, payload, on_token, on_thinking
    )
    if "usage" in result:
        _record_usage(model, result["usage"])
    else:
        _record_usage(model, _estimate_tokens(full_messages, result))
    return result


# Cache: (model, text) -> embedding vector.
_embedding_cache: dict[tuple[str, str], Embedding] = {}


def get_embeddings(texts: list[str], model: str) -> Embeddings:
    """Fetch embeddings with per-text caching. Returns (n, dim) float32 ndarray."""
    if not model:
        raise ValueError("model must be a non-empty string")
    results: dict[int, Embedding] = {}
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        key = (model, text)
        if key in _embedding_cache:
            results[i] = _embedding_cache[key]
        else:
            uncached.append((i, text))

    if uncached:

        def _fetch_embeddings() -> list[dict[str, Any]]:
            response = requests.post(
                f"{BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": model, "input": [text for _, text in uncached]},
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()["data"]  # type: ignore[no-any-return]

        data: list[dict[str, Any]] = _with_retry(_fetch_embeddings, None)
        # Sort by index to match input order within the batch.
        sorted_data = sorted(data, key=lambda x: x["index"])
        for (orig_idx, text), item in zip(uncached, sorted_data):
            vec: Embedding = np.array(item["embedding"], dtype=np.float32)
            _embedding_cache[(model, text)] = vec
            results[orig_idx] = vec

    return np.stack([results[i] for i in range(len(texts))])
