"""Best-of-N sampling with majority-coherence selection."""

import json
import zlib
from collections.abc import Callable
from typing import Any

from mindloop.client import Message, chat
from mindloop.pool import submit

EXPERT_STRATEGIES = [
    "",
    "Think step by step.",
    "First identify what could go wrong, then solve.",
    "Work backwards from the desired outcome.",
    "List your assumptions before reasoning.",
    "Break the problem into smaller sub-problems.",
    "Consider the simplest possible solution first.",
    "Explain your reasoning as if teaching someone.",
]

_DEFAULT_TEMPERATURE = 0.7
# Multipliers applied to the base temperature when cycling through tiers.
TEMPERATURE_SCALES = [1.0, 1.3, 0.7]


def compression_score(text: str) -> float:
    """Compression ratio via zlib: len(compressed) / len(encoded)."""
    encoded = text.encode()
    if not encoded:
        return 1.0
    compressed = zlib.compress(encoded)
    return len(compressed) / len(encoded)


def ngram_diversity(text: str, n: int = 3) -> float:
    """Ratio of unique n-grams to total n-grams."""
    words = text.split()
    if len(words) < n:
        return 1.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def coherence_score(text: str) -> float:
    """Weighted combination of compression and n-gram diversity."""
    return 0.5 * compression_score(text) + 0.5 * ngram_diversity(text)


def _extract_action(msg: Message) -> str:
    """Extract a hashable action key from a response."""
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        tc = tool_calls[0]
        name = tc.get("function", {}).get("name", "")
        args = tc.get("function", {}).get("arguments", "")
        # Normalize arguments by re-serializing parsed JSON.
        try:
            args = json.dumps(json.loads(args), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            pass
        return f"tool:{name}:{args}"
    return f"text:{msg.get('content', '')}"


def majority_coherence(responses: list[Message]) -> int:
    """Select the best response by majority vote then coherence tiebreak."""
    # Cluster by action.
    clusters: dict[str, list[int]] = {}
    for i, r in enumerate(responses):
        action = _extract_action(r)
        clusters.setdefault(action, []).append(i)

    # Pick largest cluster (ties broken by first-seen order).
    best_cluster = max(clusters.values(), key=len)

    # Within cluster, rank by coherence on content/reasoning.
    def _score(idx: int) -> float:
        r = responses[idx]
        text = r.get("content", "") + r.get("reasoning", "")
        return coherence_score(text)

    return max(best_cluster, key=_score)


def chat_best_of_n(
    messages: list[Message],
    model: str,
    n: int = 1,
    select_fn: Callable[[list[Message]], int] = majority_coherence,
    **chat_kwargs: Any,
) -> Message:
    """Generate N candidates in parallel and select the best one."""
    if n <= 1:
        return chat(messages, model, **chat_kwargs)

    # Build (strategy, temperature) pairs.
    caller_temp = chat_kwargs.get("temperature")
    base_temp = caller_temp if caller_temp is not None else _DEFAULT_TEMPERATURE
    pairs: list[tuple[str, float]] = []
    for i in range(n):
        strategy = EXPERT_STRATEGIES[i % len(EXPERT_STRATEGIES)]
        scale = TEMPERATURE_SCALES[
            i // len(EXPERT_STRATEGIES) % len(TEMPERATURE_SCALES)
        ]
        pairs.append((strategy, min(base_temp * scale, 2.0)))

    # Strip settings that don't apply to parallel candidates.
    _strip = ("temperature", "stream", "on_token", "on_thinking")
    kwargs = {k: v for k, v in chat_kwargs.items() if k not in _strip}

    # Launch parallel calls.
    base_prompt = kwargs.pop("system_prompt", None) or ""
    futures = []
    for strategy, temp in pairs:
        prompt = f"{base_prompt}\n{strategy}".strip()
        fut = submit(
            chat,
            messages,
            model,
            system_prompt=prompt,
            stream=False,
            temperature=temp,
            **kwargs,
        )
        futures.append(fut)

    # Collect responses preserving submission order.
    responses: list[Message] = [f.result() for f in futures]

    winner = select_fn(responses)
    return responses[winner]
