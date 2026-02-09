"""Chat log parsing, chunking, and semantic merging."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from mindloop.client import Embedding, Embeddings

# JSONL role -> display role.
_ROLE_MAP = {"user": "You", "assistant": "Bot"}

# Default gap (in seconds) between turns that triggers a new chunk.
DEFAULT_GAP_THRESHOLD = 120


@dataclass
class Turn:
    timestamp: datetime
    role: str
    text: str
    preceded_by_blank: bool = False


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
        turns.append(Turn(timestamp=ts, role=role, text=entry["content"]))
    return turns


def chunk_turns(
    turns: list[Turn], gap_threshold: int = DEFAULT_GAP_THRESHOLD
) -> list[Chunk]:
    """Group turns into chunks, splitting on time gaps and blank lines."""
    if not turns:
        return []

    chunks = [Chunk(turns=[turns[0]])]
    for prev, cur in zip(turns, turns[1:]):
        gap = (cur.timestamp - prev.timestamp).total_seconds()
        if gap > gap_threshold or cur.preceded_by_blank:
            chunks.append(Chunk())
        chunks[-1].turns.append(cur)

    return chunks


def cosine_similarities(embeddings: Embeddings) -> Embedding:
    """Compute cosine similarity between each consecutive pair. Returns 1D ndarray."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    # Dot product of each row with the next one.
    result: Embedding = (normalized[:-1] * normalized[1:]).sum(axis=1)
    return result


# ~4 chars per token, 2048 tokens default budget.
DEFAULT_MAX_CHUNK_CHARS = 8192


def merge_chunks(
    chunks: list[Chunk],
    similarities: Embedding,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Merge adjacent chunks whose similarity is above mean - 0.5 * stddev."""
    threshold = float(similarities.mean() - 0.5 * similarities.std())
    print(
        f"Merge threshold: {threshold:.4f} (mean={similarities.mean():.4f}, std={similarities.std():.4f})\n"
    )

    merged = [Chunk(turns=list(chunks[0].turns))]
    for i, sim in enumerate(similarities):
        candidate = Chunk(turns=merged[-1].turns + chunks[i + 1].turns)
        if sim >= threshold and len(candidate.text) <= max_chunk_chars:
            merged[-1] = candidate
        else:
            merged.append(Chunk(turns=list(chunks[i + 1].turns)))

    return merged
