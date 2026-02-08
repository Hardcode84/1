"""Chat log parsing, chunking, and semantic merging."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Matches log lines like "14:30:05 You: hello" or "14:30:07 Bot: hi there".
TURN_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(You|Bot):\s+(.*)$")

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
    """Parse log file into a list of turns."""
    turns = []
    saw_blank = False
    for line in path.read_text().splitlines():
        if not line.strip():
            saw_blank = True
            continue
        m = TURN_RE.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%H:%M:%S")
            turns.append(
                Turn(
                    timestamp=ts,
                    role=m.group(2),
                    text=m.group(3),
                    preceded_by_blank=saw_blank,
                )
            )
            saw_blank = False
        elif turns:
            # Continuation line — append to previous turn.
            turns[-1].text += "\n" + line.strip()
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


def cosine_similarities(embeddings: list[list[float]]) -> list[float]:
    """Compute cosine similarity between each consecutive pair of embeddings."""
    vecs = np.array(embeddings)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / norms
    # Dot product of each row with the next one.
    result: list[float] = (normalized[:-1] * normalized[1:]).sum(axis=1).tolist()
    return result


def merge_chunks(chunks: list[Chunk], similarities: list[float]) -> list[Chunk]:
    """Merge adjacent chunks whose similarity is above mean - 0.5 * stddev."""
    sims = np.array(similarities)
    threshold = sims.mean() - 0.5 * sims.std()
    print(
        f"Merge threshold: {threshold:.4f} (mean={sims.mean():.4f}, std={sims.std():.4f})\n"
    )

    merged = [Chunk(turns=list(chunks[0].turns))]
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            merged[-1].turns.extend(chunks[i + 1].turns)
        else:
            merged.append(Chunk(turns=list(chunks[i + 1].turns)))

    return merged
