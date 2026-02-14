"""Post-session memory extraction from conversation logs."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from mindloop.chunker import Chunk, Turn, chunk_turns, compact_chunks, merge_chunks
from mindloop.client import chat, get_embeddings
from mindloop.memory import MemoryStore
from mindloop.recap import collapse_messages
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunk
from mindloop.util import DEFAULT_WORKERS, noop

_EXTRACTION_MODEL = "deepseek/deepseek-v3.2"

_CONTEXT_CHARS = 200

_SYSTEM_PROMPT = """\
You extract factual memories from a conversation excerpt.
Return a JSON array of objects, each with "text" and "abstract" keys.
"text" is the full fact (1-3 sentences). "abstract" is a one-sentence TL;DR.
Only extract concrete, reusable facts — skip greetings, filler, and meta-talk.
If there is nothing worth remembering, return an empty array: []
Return ONLY valid JSON, no markdown fences or commentary.\
"""


def extract_facts(
    text: str,
    context: str | None = None,
    model: str | None = None,
) -> list[dict[str, str]]:
    """Extract factual memories from a text chunk via a single LLM call."""
    user_content = text
    if context:
        user_content = f"Previous context: {context}\n---\n{text}"

    msg = chat(
        [{"role": "user", "content": user_content}],
        model=model or _EXTRACTION_MODEL,
        system_prompt=_SYSTEM_PROMPT,
        stream=False,
        temperature=0,
        seed=42,
        cache_messages=False,
    )
    raw = msg.get("content", "")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    # Validate each entry has the required keys.
    return [
        f for f in parsed if isinstance(f, dict) and "text" in f and "abstract" in f
    ]


def extract_session(
    messages: list[dict[str, Any]],
    store: MemoryStore,
    model: str | None = None,
    log: Callable[[str], None] = noop,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Full extraction pipeline: collapse → chunk → extract → summarize → save.

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
        contexts.append(chunk.text[-_CONTEXT_CHARS:])

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

    # Summarize and save each fact sequentially (SQLite writes).
    total_saved = 0
    for chunk_idx, facts in all_facts:
        for fact in facts:
            log(f"  Saving: {fact['abstract']}")
            fact_chunk = Chunk(
                turns=[Turn(timestamp=datetime.now(), role="memory", text=fact["text"])]
            )
            cs = summarize_chunk(fact_chunk, model=model)
            save_memory(
                store,
                text=fact["text"],
                abstract=cs.abstract or fact["abstract"],
                summary=cs.summary,
                model=model or "openrouter/free",
                log=log,
            )
            total_saved += 1

    log(f"Extracted {total_saved} facts from {n} chunks.")
    return total_saved
