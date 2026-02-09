"""Chunk summarization into abstract + expanded summary."""

from dataclasses import dataclass

from mindloop.chunker import Chunk
from mindloop.client import DEFAULT_MODEL, chat

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
    raw = chat(
        messages,
        model=model or DEFAULT_MODEL,
        system_prompt=SYSTEM_PROMPT,
        stream=False,
    )

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
    chunks: list[Chunk], model: str | None = None
) -> list[ChunkSummary]:
    """Summarize a list of chunks sequentially."""
    return [summarize_chunk(chunk, model=model) for chunk in chunks]
