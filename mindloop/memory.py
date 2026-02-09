"""Persistent chunk storage and retrieval using SQLite + numpy."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType

import numpy as np

from mindloop.chunker import Chunk, Turn
from mindloop.client import get_embeddings
from mindloop.summarizer import ChunkSummary

DEFAULT_DB_PATH = Path("memory.db")


@dataclass
class SearchResult:
    chunk_summary: ChunkSummary
    score: float


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            abstract TEXT NOT NULL,
            summary TEXT NOT NULL,
            time_range TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()


class MemoryStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.conn = sqlite3.connect(db_path)
        _init_db(self.conn)

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def save(self, chunk_summary: ChunkSummary, embedding: list[float]) -> int:
        """Save a chunk summary with its embedding. Returns the row id."""
        vec = np.array(embedding, dtype=np.float32)
        cursor = self.conn.execute(
            "INSERT INTO chunks (text, abstract, summary, time_range, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                chunk_summary.chunk.text,
                chunk_summary.abstract,
                chunk_summary.summary,
                chunk_summary.chunk.time_range,
                vec.tobytes(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def save_many(
        self, summaries: list[ChunkSummary], embeddings: list[list[float]]
    ) -> list[int]:
        """Save multiple chunk summaries with embeddings."""
        return [self.save(summary, emb) for summary, emb in zip(summaries, embeddings)]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find the most relevant chunks by cosine similarity to the query."""
        query_emb = np.array(get_embeddings([query])[0], dtype=np.float32)

        rows = self.conn.execute(
            "SELECT id, text, abstract, summary, time_range, embedding FROM chunks"
        ).fetchall()

        if not rows:
            return []

        # Build matrix of stored embeddings.
        ids = []
        meta = []
        vecs = []
        for row in rows:
            ids.append(row[0])
            meta.append(row[1:5])  # text, abstract, summary, time_range.
            vecs.append(np.frombuffer(row[5], dtype=np.float32))

        matrix = np.stack(vecs)
        # Cosine similarity: dot product of normalized vectors.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        query_norm = max(float(np.linalg.norm(query_emb)), 1e-10)
        scores: np.ndarray = (matrix @ query_emb) / (norms.squeeze() * query_norm)

        # Top-K indices.
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            text, abstract, summary, time_range = meta[idx]
            # Reconstruct a minimal Chunk from stored text.
            chunk = Chunk(
                turns=[
                    Turn(
                        timestamp=datetime.min,
                        role="",
                        text=text,
                    )
                ]
            )
            cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
            results.append(SearchResult(chunk_summary=cs, score=float(scores[idx])))

        return results

    def count(self) -> int:
        """Return the number of stored chunks."""
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self.conn.close()
