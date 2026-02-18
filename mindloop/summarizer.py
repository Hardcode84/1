"""Chunk summarization into abstract + expanded summary."""

from collections.abc import Callable
from concurrent.futures import as_completed
from dataclasses import dataclass

from mindloop.chunker import Chunk
from mindloop.client import DETERMINISTIC_PARAMS, chat
from mindloop.pool import submit
from mindloop.util import noop


_SYSTEM_PROMPT = """\
You summarize conversation excerpts from a chat log between "You" (user) and "Bot" (assistant).
Write from the assistant's perspective using first person ("I"). \
Never use third person ("the AI", "the assistant", "the bot", "an instance").
Respond in exactly this format (two lines, keep the prefixes):
ABSTRACT: <one sentence TL;DR>
SUMMARY: <2-4 sentence expanded overview>\
"""


@dataclass
class ChunkSummary:
    chunk: Chunk
    abstract: str
    summary: str


def summarize_chunk(chunk: Chunk, model: str) -> ChunkSummary:
    """Summarize a single chunk into abstract + expanded summary."""
    messages = [{"role": "user", "content": chunk.text}]
    msg = chat(
        messages,
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        stream=False,
        **DETERMINISTIC_PARAMS,
        cache_messages=False,
    )
    raw = msg.get("content", "")

    abstract = ""
    summary = ""
    for line in raw.strip().splitlines():
        if line.upper().startswith("ABSTRACT:"):
            abstract = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if not abstract and not summary:
        return ChunkSummary(chunk=chunk, abstract="(parse error)", summary=raw)
    return ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)


def summarize_chunks(
    chunks: list[Chunk],
    model: str,
    log: Callable[[str], None] = noop,
) -> list[ChunkSummary]:
    """Summarize a list of chunks in parallel via global pool."""
    n = len(chunks)
    ordered: list[ChunkSummary | None] = [None] * n
    future_to_idx = {
        submit(summarize_chunk, chunk, model): i for i, chunk in enumerate(chunks)
    }
    done = 0
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        ordered[idx] = future.result()
        done += 1
        log(f"  Summarized chunk {done}/{n} (index {idx})...")

    return list(ordered)  # type: ignore[arg-type]
