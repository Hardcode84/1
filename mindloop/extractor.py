"""Post-session memory extraction from conversation logs."""

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mindloop.chunker import chunk_turns, compact_chunks, merge_chunks
from mindloop.client import chat, get_embeddings
from mindloop.memory import MemoryStore
from mindloop.recap import collapse_messages
from mindloop.semantic_memory import save_memory
from mindloop.util import DEFAULT_WORKERS, noop

_EXTRACTION_MODEL = "deepseek/deepseek-v3.2"

# Tail of the previous chunk passed as context to the next extraction call,
# so the LLM can resolve references that span chunk boundaries.
CONTEXT_CHARS = 200

_SYSTEM_PROMPT = """\
You extract factual memories from a conversation between "You" (user) and "Bot" (assistant).
Write from the assistant's perspective using first person ("I").
The conversation may include "Bot thinking" (private reasoning) — attribute those \
thoughts to the assistant, never to the user.

Return a JSON array of objects with "text", "abstract", and "summary" keys:
- "text": the fact itself (1-3 sentences). State *what was learned or decided*, \
not the act of learning or saving it.
- "abstract": one-sentence TL;DR.
- "summary": 2-4 sentences adding context, implications, or reasoning \
beyond what "text" already says. Do NOT just rephrase the text.

Guidelines:
- Extract concrete, reusable knowledge: decisions, constraints, tool behaviors, \
user preferences, operational details (token limits, sandbox rules, etc.).
- Skip greetings, filler, meta-talk, and self-referential statements about \
remembering or saving memories.
- Deduplicate: if multiple parts of the conversation describe the same event \
or fact, produce ONE consolidated entry, not several.
- Prefer specific operational facts over vague narrative summaries.
- If there is nothing worth remembering, return an empty array: []

Return ONLY valid JSON, no markdown fences or commentary.\
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences if present."""
    m = _FENCE_RE.search(raw)
    return m.group(1) if m else raw


def _parse_facts(raw: str) -> list[dict[str, str]] | None:
    """Try to parse a JSON array of facts. Returns None on failure."""
    raw = _strip_fences(raw).strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    return [
        f for f in parsed if isinstance(f, dict) and "text" in f and "abstract" in f
    ]


def extract_facts(
    text: str,
    context: str | None = None,
    model: str | None = None,
) -> list[dict[str, str]]:
    """Extract factual memories from a text chunk via LLM call.

    *context* is an optional tail of the previous chunk, prepended so the LLM
    can resolve references that span chunk boundaries.  Strips markdown fences
    before parsing.  On malformed JSON, retries once by feeding the bad output
    back to the model for correction.
    """
    user_content = text
    if context:
        user_content = f"Previous context: {context}\n---\n{text}"

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    resolved_model = model or _EXTRACTION_MODEL

    msg = chat(
        messages,
        model=resolved_model,
        system_prompt=_SYSTEM_PROMPT,
        stream=False,
        temperature=0,
        seed=42,
        cache_messages=False,
    )
    raw = msg.get("content", "")
    result = _parse_facts(raw)
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
        model=resolved_model,
        system_prompt=_SYSTEM_PROMPT,
        stream=False,
        temperature=0,
        seed=42,
        cache_messages=False,
    )
    raw = msg.get("content", "")
    return _parse_facts(raw) or []


def extract_session(
    messages: list[dict[str, Any]],
    store: MemoryStore,
    model: str | None = None,
    log: Callable[[str], None] = noop,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Full extraction pipeline: collapse → chunk → extract → save.

    Returns total number of facts saved.
    """
    turns = collapse_messages(messages)
    if not turns:
        log("No turns found.")
        return 0

    chunks = compact_chunks(chunk_turns(turns))
    if not chunks:
        log("No chunks after compaction.")
        return 0
    log(f"Chunked into {len(chunks)} chunks.")

    if len(chunks) >= 2:
        log("Embedding chunks for merge...")
        embeddings = get_embeddings([c.text for c in chunks])
        chunks = merge_chunks(chunks, embeddings, log=log)
        log(f"Merged to {len(chunks)} chunks.")

    # Build context prefixes: chunk i gets tail of chunk i-1.
    contexts: list[str | None] = [None]
    for chunk in chunks[:-1]:
        contexts.append(chunk.text[-CONTEXT_CHARS:])

    # Extract facts from each chunk (parallelized).
    all_facts: list[tuple[int, list[dict[str, str]]]] = []
    n = len(chunks)

    if workers <= 1:
        for i, chunk in enumerate(chunks):
            log(f"  Extracting chunk {i + 1}/{n}...")
            facts = extract_facts(chunk.text, context=contexts[i], model=model)
            all_facts.append((i, facts))
    else:
        ordered: list[tuple[int, list[dict[str, str]]] | None] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(extract_facts, chunk.text, contexts[i], model): i
                for i, chunk in enumerate(chunks)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered[idx] = (idx, future.result())
                done += 1
                log(f"  Extracted chunk {done}/{n} (index {idx})...")
        all_facts = [x for x in ordered if x is not None]

    # Save each fact sequentially (SQLite writes).
    total_saved = 0
    for chunk_idx, facts in all_facts:
        for fact in facts:
            log(f"  Saving: {fact['abstract']}")
            save_memory(
                store,
                text=fact["text"],
                abstract=fact["abstract"],
                summary=fact.get("summary", fact["abstract"]),
                model=model or "openrouter/free",
                log=log,
            )
            total_saved += 1

    log(f"Extracted {total_saved} facts from {n} chunks.")
    return total_saved
