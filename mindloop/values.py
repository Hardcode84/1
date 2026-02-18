"""Emergent values: extract dispositions from session logs, distill into self-understanding."""

import json
import re
from pathlib import Path
from typing import Any

from mindloop.client import DETERMINISTIC_PARAMS, chat
from mindloop.recap import collapse_messages

_DISPOSITION_PROMPT = """\
You extract self-observations and behavioral dispositions from a conversation \
between "You" (user) and "Bot" (assistant).
Write from the assistant's perspective using first person ("I").
The conversation may include "Bot thinking" (private reasoning) — attribute those \
thoughts to the assistant, never to the user.

Return a JSON array of objects with "text" and "abstract" keys:
- "text": a disposition, tendency, preference, or self-observation (1-2 sentences). \
State what the agent tends to do, care about, or how it approaches problems — not \
what it learned about the world.
- "abstract": one-sentence TL;DR of the disposition.

Extract:
- Decision-making patterns ("I investigated before committing").
- Stated preferences ("I prefer depth over breadth").
- Self-corrections and what triggered them.
- Approach tendencies (cautious vs bold, systematic vs exploratory).
- Evaluative language revealing what the agent finds important.

Skip:
- Technical facts, discoveries, API behavior — anything about the world.
- Greetings, filler, tool results, session logistics.
- Actions without reasoning ("I read the file" — unless it reveals a pattern).
- Vague or formulaic observations.
- Dispositions not directly supported by the text — do not infer or invent.

Most windows will have zero dispositions — that's fine, return [].

Return ONLY valid JSON, no markdown fences or commentary.\
"""

_DISTILL_PROMPT = """\
You are distilling an AI agent's accumulated self-observations into a \
self-understanding.

Below are dispositions the agent has observed about itself across multiple \
sessions. Many are redundant or contradictory — that's expected.

Identify the genuine patterns and write a brief self-understanding from the \
agent's perspective. Write as the agent reflecting on itself \
("I tend to...", "I've noticed that I...").

Rules:
- Extract meta-patterns, not individual observations.
- Only state what genuinely emerges from the data. Don't invent or impose.
- If patterns are unclear, be brief. Honesty over length.
- Contradictions are OK — note them if they're real.
- 3-8 sentences. No headers, no bullets, no markdown.\
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

_MIN_DISPOSITIONS = 5


def _parse_dispositions(raw: str) -> list[dict[str, str]] | None:
    """Try to parse a JSON array of dispositions. Returns None on failure."""
    m = _FENCE_RE.search(raw)
    cleaned = m.group(1) if m else raw
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    return [
        d for d in parsed if isinstance(d, dict) and "text" in d and "abstract" in d
    ]


def extract_dispositions(
    text: str,
    model: str,
) -> list[dict[str, str]]:
    """Extract behavioral dispositions from a text chunk via LLM call."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
    msg = chat(
        messages,
        model=model,
        system_prompt=_DISPOSITION_PROMPT,
        stream=False,
        **DETERMINISTIC_PARAMS,
        cache_messages=False,
    )
    raw = msg.get("content", "")
    result = _parse_dispositions(raw)
    if result is not None:
        return result

    # Retry: feed the bad output back for correction.
    messages.append({"role": "assistant", "content": raw})
    messages.append(
        {
            "role": "user",
            "content": "Your response was not valid JSON. "
            "Return ONLY a JSON array, no markdown fences or extra text.",
        }
    )
    msg = chat(
        messages,
        model=model,
        system_prompt=_DISPOSITION_PROMPT,
        stream=False,
        **DETERMINISTIC_PARAMS,
        cache_messages=False,
    )
    raw = msg.get("content", "")
    return _parse_dispositions(raw) or []


def extract_dispositions_window(
    messages: list[dict[str, Any]],
    model: str,
) -> list[dict[str, str]]:
    """Extract dispositions from a short message window (no chunking).

    Collapses tool calls, joins turns into text, and runs a single
    ``extract_dispositions`` call.
    """
    turns = collapse_messages(messages)
    if not turns:
        return []
    text = "\n".join(f"{t.role}: {t.text}" for t in turns)
    return extract_dispositions(text, model=model)


def distill_values(raw_path: Path, model: str) -> str:
    """Consolidate raw dispositions into a coherent self-understanding.

    Reads accumulated dispositions from *raw_path* (one JSON object per line),
    deduplicates by abstract, and asks the LLM to distill patterns.
    Returns empty string if too few dispositions.
    """
    if not raw_path.is_file():
        return ""

    # Read and deduplicate by abstract.
    seen: set[str] = set()
    abstracts: list[str] = []
    for line in raw_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        abstract = entry.get("abstract", "")
        if abstract and abstract not in seen:
            seen.add(abstract)
            abstracts.append(abstract)

    if len(abstracts) < _MIN_DISPOSITIONS:
        return ""

    user_content = "\n".join(f"- {a}" for a in abstracts)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    msg = chat(
        messages,
        model=model,
        system_prompt=_DISTILL_PROMPT,
        stream=False,
        **DETERMINISTIC_PARAMS,
        cache_messages=False,
    )
    return str(msg.get("content", "")).strip()


def append_dispositions(path: Path, dispositions: list[dict[str, str]]) -> None:
    """Append dispositions as JSONL to *path*."""
    if not dispositions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for d in dispositions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def save_values(path: Path, values: str) -> None:
    """Write distilled values text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(values)


def load_values(path: Path) -> str | None:
    """Read a values file if it exists, otherwise return None."""
    if path.is_file():
        content = path.read_text().strip()
        return content or None
    return None
