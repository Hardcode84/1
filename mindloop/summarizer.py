"""Chunk summarization into abstract + expanded summary."""

from collections.abc import Callable
from dataclasses import dataclass

from mindloop.chunker import Chunk
from mindloop.client import chat


def _noop(_msg: str) -> None:
    pass


# SUMMARIZATION_MODEL = "tngtech/deepseek-r1t2-chimera:free"
SUMMARIZATION_MODEL = "deepseek/deepseek-v3.2"

SYSTEM_PROMPT = """\
You summarize conversation excerpts from a chat log.
Respond in exactly this format (two lines, keep the prefixes):
ABSTRACT: <one sentence TL;DR>
SUMMARY: <2-4 sentence expanded overview>\
"""


@dataclass
class ChunkSummary:
    chunk: Chunk
    abstract: str
    summary: str


def summarize_chunk(chunk: Chunk, model: str | None = None) -> ChunkSummary:
    """Summarize a single chunk into abstract + expanded summary."""
    messages = [{"role": "user", "content": chunk.text}]
    msg = chat(
        messages,
        model=model or SUMMARIZATION_MODEL,
        system_prompt=SYSTEM_PROMPT,
        stream=False,
        temperature=0,
        seed=42,
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
    model: str | None = None,
    log: Callable[[str], None] = _noop,
) -> list[ChunkSummary]:
    """Summarize a list of chunks sequentially."""
    results: list[ChunkSummary] = []
    for i, chunk in enumerate(chunks, 1):
        log(f"  Summarizing chunk {i}/{len(chunks)}...")
        results.append(summarize_chunk(chunk, model=model))
    return results
