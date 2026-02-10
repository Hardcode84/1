"""Chat log parsing, chunking, and semantic merging."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from mindloop.client import Embedding, Embeddings

# JSONL role -> display role.
_ROLE_MAP = {"user": "You", "assistant": "Bot", "tool": "Tool"}


@dataclass
class Turn:
    timestamp: datetime
    role: str
    text: str


@dataclass
class Chunk:
    turns: list[Turn] = field(default_factory=list)

    @property
    def time_range(self) -> str:
        start = self.turns[0].timestamp.strftime("%H:%M:%S")
        end = self.turns[-1].timestamp.strftime("%H:%M:%S")
        return f"{start} - {end}"

    @property
    def text(self) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.turns)


def parse_turns(path: Path) -> list[Turn]:
    """Parse a JSONL log file into a list of turns."""
    turns: list[Turn] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        ts = datetime.fromisoformat(entry["timestamp"])
        role = _ROLE_MAP.get(entry["role"], entry["role"])
        # Emit reasoning as a separate turn before the main content.
        reasoning = entry.get("reasoning")
        if reasoning:
            turns.append(Turn(timestamp=ts, role=f"{role} thinking", text=reasoning))
        content = entry.get("content") or ""
        turns.append(Turn(timestamp=ts, role=role, text=content))
    return turns


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def parse_turns_md(path: Path) -> list[Turn]:
    """Parse a Markdown file into turns, splitting on headings."""
    content = path.read_text()
    if not content.strip():
        return []

    ts = datetime.fromtimestamp(path.stat().st_mtime)
    turns: list[Turn] = [Turn(timestamp=ts, role="Doc", text=path.name)]
    current_lines: list[str] = []

    def _flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            turns.append(Turn(timestamp=ts, role="Doc", text=body))

    for line in content.splitlines():
        if _HEADING_RE.match(line):
            _flush()
            current_lines = [line]
        else:
            current_lines.append(line)

    _flush()
    return turns


def chunk_turns(turns: list[Turn]) -> list[Chunk]:
    """Group turns into chunks, splitting on blank lines inside message content."""
    if not turns:
        return []

    chunks: list[Chunk] = [Chunk()]
    for turn in turns:
        # Split turn text on blank lines.
        parts = turn.text.split("\n\n")
        # First part extends the current chunk.
        chunks[-1].turns.append(
            Turn(timestamp=turn.timestamp, role=turn.role, text=parts[0])
        )
        # Each subsequent part starts a new chunk.
        for part in parts[1:]:
            chunks.append(
                Chunk(turns=[Turn(timestamp=turn.timestamp, role=turn.role, text=part)])
            )

    return chunks


# Chunks shorter than this are absorbed into a neighbor.
DEFAULT_MIN_CHUNK_CHARS = 80


def compact_chunks(
    chunks: list[Chunk], min_chars: int = DEFAULT_MIN_CHUNK_CHARS
) -> list[Chunk]:
    """Merge undersized chunks into their nearest neighbor."""
    if not chunks:
        return []

    result: list[Chunk] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(result[-1].text) < min_chars:
            # Previous chunk is too small — absorb current into it.
            result[-1] = Chunk(turns=result[-1].turns + chunk.turns)
        else:
            result.append(chunk)

    # If the last chunk ended up too small, merge it back into the previous.
    if len(result) >= 2 and len(result[-1].text) < min_chars:
        result[-2] = Chunk(turns=result[-2].turns + result[-1].turns)
        result.pop()

    return result


def cosine_similarities(embeddings: Embeddings) -> Embedding:
    """Compute cosine similarity between each consecutive pair. Returns 1D ndarray."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    # Dot product of each row with the next one.
    result: Embedding = (normalized[:-1] * normalized[1:]).sum(axis=1)
    return result


# ~4 chars per token, 2048 tokens default budget.
DEFAULT_MAX_CHUNK_CHARS = 8192


_ALWAYS_MERGE_SIMILARITY = 0.8


def merge_chunks(
    chunks: list[Chunk],
    embeddings: Embeddings,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Merge adjacent chunks until no more merges are possible.

    Merges when similarity >= statistical threshold (mean - 0.5*std)
    or >= 0.8 (always merge). Repeats until fixed point.
    """
    while len(chunks) >= 2:
        sims = cosine_similarities(embeddings)
        threshold = float(sims.mean() - 0.5 * sims.std())
        print(
            f"Merge pass ({len(chunks)} chunks): threshold={threshold:.4f}"
            f" (mean={sims.mean():.4f}, std={sims.std():.4f})"
        )

        new_chunks: list[Chunk] = [Chunk(turns=list(chunks[0].turns))]
        new_embeddings: list[Embedding] = [embeddings[0]]
        merge_count = 0

        for i, sim in enumerate(sims):
            can_merge = sim >= threshold or sim >= _ALWAYS_MERGE_SIMILARITY
            candidate = Chunk(turns=new_chunks[-1].turns + chunks[i + 1].turns)
            if can_merge and len(candidate.text) <= max_chunk_chars:
                new_chunks[-1] = candidate
                # Average the embeddings of merged chunks.
                n = merge_count + 2  # Number of original chunks in this group.
                new_embeddings[-1] = (
                    new_embeddings[-1] * (n - 1) + embeddings[i + 1]
                ) / n
                merge_count += 1
            else:
                new_chunks.append(Chunk(turns=list(chunks[i + 1].turns)))
                new_embeddings.append(embeddings[i + 1])
                merge_count = 0

        if len(new_chunks) == len(chunks):
            break  # Fixed point.
        chunks = new_chunks
        embeddings = np.stack(new_embeddings)

    return chunks
