"""Session recap: collapse, summarize, and persist between agent instances."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from mindloop.chunker import Turn, chunk_turns, compact_chunks
from mindloop.summarizer import ChunkSummary, summarize_chunks


def _noop(_msg: str) -> None:
    pass


# Tool call results that should be skipped entirely.
_SKIP_TOOLS = {"status"}

# System message prefixes to filter (mirrors _SKIP_PREFIXES in cli/agent.py).
_SKIP_PREFIXES = ("[stop]", "[stats]", "Warning:")

# Approximate characters per token for budget estimation.
_CHARS_PER_TOKEN = 4

# Collapsed turns are short; compact aggressively to keep chunk count manageable.
_RECAP_MIN_CHUNK_CHARS = 500


def _collapse_tool_call(name: str, args: dict[str, str], result: str) -> str | None:
    """Format a tool-call + result pair as a concise natural-language string."""
    if name in _SKIP_TOOLS:
        return None

    if name == "read":
        line_count = result.count("\n") + 1
        return f"Read {args.get('path', '?')} ({line_count} lines)."
    if name == "ls":
        entries = result.strip().splitlines()
        preview = ", ".join(entries[:5])
        if len(entries) > 5:
            preview += ", ..."
        return f"Listed {args.get('path', '.')}: {preview}"
    if name == "recall":
        n = result.count("---") + (1 if result.strip() else 0)
        query = args.get("query", "?")
        return f"Recalled {n} memories about '{query}'."
    if name == "recall_detail":
        chunk_id = args.get("chunk_id", "?")
        return f"Retrieved full text of memory #{chunk_id}."
    if name == "edit":
        return f"Edited {args.get('path', '?')}."
    if name == "write":
        content = args.get("content", "")
        line_count = content.count("\n") + 1
        return f"Wrote {args.get('path', '?')} ({line_count} lines)."
    if name == "remember":
        abstract = args.get("abstract", args.get("text", "?"))
        return f"Remembered: '{abstract}'."
    if name == "ask":
        message = args.get("message", "?")
        # Trim long results for compactness.
        short = result[:120] + "..." if len(result) > 120 else result
        return f"Asked user: '{message}'. Response: '{short}'."
    if name == "done":
        summary = args.get("summary", result)
        return f"Finished: '{summary}'."

    # Unknown tool — generic fallback.
    return f"Called {name}({json.dumps(args, ensure_ascii=False)})."


def collapse_messages(messages: list[dict[str, Any]]) -> list[Turn]:
    """Convert raw JSONL messages into Turn objects with tool calls collapsed."""
    now = datetime.now()
    turns: list[Turn] = []

    # Index tool_calls by their id for pairing with tool results.
    pending_calls: dict[str, dict[str, str]] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning")
        tool_calls = msg.get("tool_calls")

        if role == "system":
            if any(content.startswith(p) for p in _SKIP_PREFIXES):
                continue
            if content.strip():
                turns.append(Turn(timestamp=now, role="System", text=content))
            continue

        if role == "assistant":
            if reasoning:
                turns.append(Turn(timestamp=now, role="Bot thinking", text=reasoning))
            if tool_calls:
                for tc in tool_calls:
                    call_id = tc.get("id", "")
                    func = tc.get("function", {})
                    pending_calls[call_id] = {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    }
            if content:
                turns.append(Turn(timestamp=now, role="Bot", text=content))
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id", "")
            call = pending_calls.pop(call_id, None)
            if call is None:
                continue
            try:
                args = json.loads(call["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            collapsed = _collapse_tool_call(call["name"], args, content)
            if collapsed is not None:
                turns.append(Turn(timestamp=now, role="Tool", text=collapsed))
            continue

        if role == "user" and content:
            turns.append(Turn(timestamp=now, role="You", text=content))

    return turns


def generate_recap(
    messages: list[dict[str, Any]],
    model: str | None = None,
    token_budget: int = 1000,
    log: Callable[[str], None] = _noop,
) -> str:
    """Full recap pipeline: collapse, chunk, summarize, score, select."""
    log(f"Collapsing {len(messages)} messages...")
    turns = collapse_messages(messages)
    if not turns:
        return ""
    log(f"  {len(turns)} turns.")

    chunks = compact_chunks(chunk_turns(turns), min_chars=_RECAP_MIN_CHUNK_CHARS)
    if not chunks:
        return ""
    log(f"Chunked into {len(chunks)} chunks.")

    log(f"Summarizing {len(chunks)} chunks...")
    summaries = summarize_chunks(chunks, model=model, log=log)
    if not summaries:
        return ""

    # Score by recency: linear 0..1 toward end.
    scored: list[tuple[float, int, ChunkSummary]] = []
    for i, s in enumerate(summaries):
        score = (i + 1) / len(summaries)
        scored.append((score, i, s))

    # Sort descending by score (highest = most recent first).
    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top summaries within budget.
    budget_chars = token_budget * _CHARS_PER_TOKEN
    selected: list[tuple[int, str]] = []
    used = 0
    for score, idx, s in scored:
        text = s.summary or s.abstract
        if used + len(text) > budget_chars and selected:
            break
        selected.append((idx, text))
        used += len(text)

    # Re-sort by original position for chronological output.
    selected.sort(key=lambda x: x[0])

    log(
        f"Selected {len(selected)}/{len(summaries)} summaries"
        f" (~{used // _CHARS_PER_TOKEN} tokens)."
    )
    return "\n".join(text for _, text in selected)


def load_recap(path: Path) -> str | None:
    """Read a recap file if it exists, otherwise return None."""
    if path.is_file():
        content = path.read_text().strip()
        return content or None
    return None


def save_recap(path: Path, recap: str) -> None:
    """Write recap text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recap)
