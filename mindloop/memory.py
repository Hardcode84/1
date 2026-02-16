"""Persistent chunk storage and retrieval using SQLite + numpy."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType

import numpy as np

from mindloop.chunker import Chunk, Turn
from mindloop.client import DEFAULT_EMBEDDING_MODEL, Embedding, get_embeddings
from mindloop.summarizer import ChunkSummary

DEFAULT_DB_PATH = Path("memory.db")


@dataclass
class SearchResult:
    id: int
    chunk_summary: ChunkSummary
    score: float
    cosine_score: float = 0.0
    source_a: int | None = None
    source_b: int | None = None


@dataclass
class LineageNode:
    """A node in the merge lineage tree."""

    id: int
    text: str
    abstract: str
    active: bool
    source_a: "LineageNode | None" = None
    source_b: "LineageNode | None" = None

    @property
    def is_leaf(self) -> bool:
        """True if this node has no sources (original input)."""
        return self.source_a is None and self.source_b is None


_RRF_K = 60


def faithfulness(
    merged_text: str,
    source_a_text: str,
    source_b_text: str,
    threshold: float = 0.7,
) -> tuple[bool, float, float]:
    """Check whether merged text faithfully represents both sources.

    Embeds all three texts and checks that cosine similarity between the
    merged text and each source meets *threshold*.  Returns
    ``(passed, sim_a, sim_b)``.
    """
    embs = get_embeddings(
        [merged_text, source_a_text, source_b_text], model=DEFAULT_EMBEDDING_MODEL
    )
    m, a, b = embs[0], embs[1], embs[2]
    norm_m = max(float(np.linalg.norm(m)), 1e-10)
    norm_a = max(float(np.linalg.norm(a)), 1e-10)
    norm_b = max(float(np.linalg.norm(b)), 1e-10)
    sim_a = float(np.dot(m, a) / (norm_m * norm_a))
    sim_b = float(np.dot(m, b) / (norm_m * norm_b))
    return (sim_a >= threshold and sim_b >= threshold, sim_a, sim_b)


def leaf_faithfulness(
    merged_text: str,
    leaf_texts: list[str],
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """Check whether merged text faithfully represents all original leaves.

    Embeds ``[merged_text] + leaf_texts`` in one call and checks that cosine
    similarity between the merged text and each leaf meets *threshold*.
    Returns ``(all_passed, min_similarity)``.
    """
    embs = get_embeddings([merged_text] + leaf_texts, model=DEFAULT_EMBEDDING_MODEL)
    m = embs[0]
    norm_m = max(float(np.linalg.norm(m)), 1e-10)
    min_sim = 1.0
    for i in range(1, len(embs)):
        norm_l = max(float(np.linalg.norm(embs[i])), 1e-10)
        sim = float(np.dot(m, embs[i]) / (norm_m * norm_l))
        if sim < min_sim:
            min_sim = sim
    passed = min_sim >= threshold
    return (passed, min_sim)


def _collect_leaf_texts(node: LineageNode) -> list[str]:
    """Collect text from all leaf nodes in a lineage tree."""
    if node.is_leaf:
        return [node.text]
    leaves: list[str] = []
    if node.source_a is not None:
        leaves.extend(_collect_leaf_texts(node.source_a))
    if node.source_b is not None:
        leaves.extend(_collect_leaf_texts(node.source_b))
    return leaves


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            abstract TEXT NOT NULL,
            summary TEXT NOT NULL,
            time_range TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_a INTEGER,
            source_b INTEGER
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active)")

    # Migrate: drop the embedding column if it exists from an older schema.
    cols = [row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]
    if "embedding" in cols:
        conn.execute("ALTER TABLE chunks RENAME TO chunks_old")
        conn.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                abstract TEXT NOT NULL,
                summary TEXT NOT NULL,
                time_range TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_a INTEGER,
                source_b INTEGER
            )
        """
        )
        conn.execute(
            """
            INSERT INTO chunks
                (id, text, abstract, summary, time_range, active,
                 created_at, source_a, source_b)
            SELECT id, text, abstract, summary, time_range, active,
                   created_at, source_a, source_b
            FROM chunks_old
        """
        )
        conn.execute("DROP TABLE chunks_old")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active)")

    # Migrate existing databases that lack the source columns.
    for col in ("source_a", "source_b"):
        try:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists.

    # FTS5 full-text index for BM25 keyword search.
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, abstract, summary,
            content='chunks', content_rowid='id'
        )
    """
    )
    # Rebuild FTS index from existing rows (idempotent, fast for small tables).
    conn.execute("INSERT OR IGNORE INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    # Edges between related-but-unmergeable chunks.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_edges (
            source_id INTEGER NOT NULL REFERENCES chunks(id),
            target_id INTEGER NOT NULL REFERENCES chunks(id),
            edge_type TEXT NOT NULL,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_id, target_id, edge_type)
        )
    """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_source ON chunk_edges(source_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_target ON chunk_edges(target_id)"
    )
    conn.commit()


def _fts_escape(query: str) -> str:
    """Escape a raw query string for safe use in FTS5 MATCH."""
    # Double-quote each token so special chars are treated as literals.
    tokens = query.split()
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


class MemoryStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.conn = sqlite3.connect(db_path)
        self._transaction_depth = 0
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

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Batch multiple operations into a single atomic commit.

        Supports nesting — only the outermost transaction commits or rolls back.
        """
        self._transaction_depth += 1
        try:
            yield
            if self._transaction_depth == 1:
                self.conn.commit()
        except BaseException:
            if self._transaction_depth == 1:
                self.conn.rollback()
            raise
        finally:
            self._transaction_depth -= 1

    def _auto_commit(self) -> None:
        """Commit unless inside a transaction."""
        if self._transaction_depth == 0:
            self.conn.commit()

    def find_exact(self, text: str) -> int | None:
        """Return the id of an active chunk with exactly matching text, or None."""
        row = self.conn.execute(
            "SELECT id FROM chunks WHERE active = 1 AND text = ?", (text,)
        ).fetchone()
        return row[0] if row else None

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        score: float | None = None,
    ) -> None:
        """Record a directional edge between two chunks.

        Uses INSERT OR IGNORE so duplicate (source, target, type) triples
        are silently skipped.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO chunk_edges "
            "(source_id, target_id, edge_type, score) VALUES (?, ?, ?, ?)",
            (source_id, target_id, edge_type, score),
        )
        self._auto_commit()

    def edges(self, chunk_id: int) -> list[tuple[int, str, float | None]]:
        """Return (other_id, edge_type, score) for all edges touching *chunk_id*."""
        rows = self.conn.execute(
            """
            SELECT target_id, edge_type, score
                FROM chunk_edges WHERE source_id = ?
            UNION ALL
            SELECT source_id, edge_type, score
                FROM chunk_edges WHERE target_id = ?
            """,
            (chunk_id, chunk_id),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def inherit_edges(self, new_id: int, absorbed_ids: list[int]) -> None:
        """Move edges from absorbed chunks onto *new_id* with re-scored cosine.

        Collects all edges from *absorbed_ids* that point outside the merge
        (i.e. not to *new_id* or any of the *absorbed_ids*), computes cosine
        similarity between *new_id*'s text and each target, creates new
        edges, and deletes the old ones.  One ``get_embeddings`` call for the
        whole batch.
        """
        exclude = set(absorbed_ids) | {new_id}

        # Collect unique external targets.
        targets: set[int] = set()
        for aid in absorbed_ids:
            for other_id, _etype, _score in self.edges(aid):
                if other_id not in exclude:
                    targets.add(other_id)

        # Delete all edges touching absorbed chunks.
        for aid in absorbed_ids:
            self.conn.execute(
                "DELETE FROM chunk_edges WHERE source_id = ? OR target_id = ?",
                (aid, aid),
            )

        if not targets:
            self._auto_commit()
            return

        # Fetch texts for new chunk + all targets.
        target_ids = sorted(targets)
        new_text_row = self.conn.execute(
            "SELECT text FROM chunks WHERE id = ?", (new_id,)
        ).fetchone()
        if new_text_row is None:
            self._auto_commit()
            return
        ph = ",".join("?" for _ in target_ids)
        rows = self.conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({ph})", target_ids
        ).fetchall()
        text_by_id = {r[0]: r[1] for r in rows}

        # Order target texts for a single batch embedding call.
        ordered_ids = [tid for tid in target_ids if tid in text_by_id]
        if not ordered_ids:
            self._auto_commit()
            return
        texts = [new_text_row[0]] + [text_by_id[tid] for tid in ordered_ids]
        embs = get_embeddings(texts, model=DEFAULT_EMBEDDING_MODEL)

        new_emb = embs[0]
        norm_new = max(float(np.linalg.norm(new_emb)), 1e-10)
        for i, tid in enumerate(ordered_ids):
            target_emb = embs[i + 1]
            norm_t = max(float(np.linalg.norm(target_emb)), 1e-10)
            sim = float(np.dot(new_emb, target_emb) / (norm_new * norm_t))
            self.add_edge(new_id, tid, "related_to", score=sim)

    def edge_counts(self, chunk_ids: list[int]) -> dict[int, int]:
        """Return the number of edges touching each chunk id."""
        if not chunk_ids:
            return {}
        ph = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"""
            SELECT id, SUM(cnt) FROM (
                SELECT source_id AS id, COUNT(*) AS cnt
                    FROM chunk_edges WHERE source_id IN ({ph}) GROUP BY source_id
                UNION ALL
                SELECT target_id AS id, COUNT(*) AS cnt
                    FROM chunk_edges WHERE target_id IN ({ph}) GROUP BY target_id
            ) GROUP BY id
            """,
            chunk_ids + chunk_ids,
        ).fetchall()
        return {cid: cnt for cid, cnt in rows}

    def save(
        self,
        chunk_summary: ChunkSummary,
        source_a: int | None = None,
        source_b: int | None = None,
    ) -> int:
        """Save a chunk summary. Returns the row id."""
        cursor = self.conn.execute(
            "INSERT INTO chunks "
            "(text, abstract, summary, time_range, source_a, source_b) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk_summary.chunk.text,
                chunk_summary.abstract,
                chunk_summary.summary,
                chunk_summary.chunk.time_range,
                source_a,
                source_b,
            ),
        )
        row_id = cursor.lastrowid or 0
        # Keep FTS5 index in sync.
        self.conn.execute(
            "INSERT INTO chunks_fts(rowid, text, abstract, summary) VALUES (?, ?, ?, ?)",
            (
                row_id,
                chunk_summary.chunk.text,
                chunk_summary.abstract,
                chunk_summary.summary,
            ),
        )
        self._auto_commit()
        return row_id

    def save_many(self, summaries: list[ChunkSummary]) -> list[int]:
        """Save multiple chunk summaries."""
        return [self.save(summary) for summary in summaries]

    def search(
        self, query: str, top_k: int = 5, original_only: bool = False
    ) -> list[SearchResult]:
        """Find the most relevant chunks via hybrid embedding + BM25 search.

        Combines cosine similarity (embeddings) and BM25 (FTS5) using
        Reciprocal Rank Fusion.  When *original_only* is True, search only
        leaf chunks (those not produced by merging) regardless of active
        status.  Otherwise search active chunks only.
        """
        query_emb: Embedding = get_embeddings([query], model=DEFAULT_EMBEDDING_MODEL)[0]

        if original_only:
            where = "WHERE source_a IS NULL AND source_b IS NULL"
        else:
            where = "WHERE active = 1"

        rows = self.conn.execute(
            "SELECT id, text, abstract, summary, time_range, "
            f"source_a, source_b FROM chunks {where}"
        ).fetchall()

        if not rows:
            return []

        # Build lookup of chunk metadata keyed by id.
        meta_by_id: dict[int, tuple[str, str, str, str, int | None, int | None]] = {}
        texts_by_id: dict[int, str] = {}
        for row in rows:
            meta_by_id[row[0]] = (row[1], row[2], row[3], row[4], row[5], row[6])
            texts_by_id[row[0]] = row[1]

        ids = list(meta_by_id.keys())

        # --- Embedding ranks ---
        chunk_embeddings = get_embeddings(
            [texts_by_id[cid] for cid in ids], model=DEFAULT_EMBEDDING_MODEL
        )
        matrix = np.stack([chunk_embeddings[i] for i in range(len(ids))])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        query_norm = max(float(np.linalg.norm(query_emb)), 1e-10)
        cos_scores: np.ndarray = (matrix @ query_emb) / (norms.squeeze() * query_norm)
        cosine_by_id: dict[int, float] = {
            ids[i]: float(cos_scores[i]) for i in range(len(ids))
        }
        emb_order = np.argsort(cos_scores)[::-1]
        emb_rank: dict[int, int] = {
            ids[idx]: rank for rank, idx in enumerate(emb_order)
        }

        # --- BM25 ranks via FTS5 ---
        bm25_rank: dict[int, int] = {}
        try:
            fts_query = _fts_escape(query)
            if original_only:
                fts_join = (
                    "SELECT c.id, f.rank FROM chunks_fts f "
                    "JOIN chunks c ON c.id = f.rowid "
                    "WHERE chunks_fts MATCH ? "
                    "AND c.source_a IS NULL AND c.source_b IS NULL "
                    "ORDER BY f.rank"
                )
            else:
                fts_join = (
                    "SELECT c.id, f.rank FROM chunks_fts f "
                    "JOIN chunks c ON c.id = f.rowid "
                    "WHERE chunks_fts MATCH ? AND c.active = 1 "
                    "ORDER BY f.rank"
                )
            fts_rows = self.conn.execute(fts_join, (fts_query,)).fetchall()
            for rank, (cid, _bm25_score) in enumerate(fts_rows):
                bm25_rank[cid] = rank
        except sqlite3.OperationalError:
            pass  # FTS5 unavailable or query parse error — fall back to embedding only.

        # --- Reciprocal Rank Fusion ---
        all_ids = set(emb_rank) | set(bm25_rank)
        fallback = len(all_ids)  # Penalty rank for missing entries.
        # Theoretical max is 2/_RRF_K (rank 0 in both). Normalize to [0, 1].
        rrf_max = 2.0 / _RRF_K
        rrf_scores: dict[int, float] = {}
        for cid in all_ids:
            e_rank = emb_rank.get(cid, fallback)
            b_rank = bm25_rank.get(cid, fallback)
            raw = 1.0 / (_RRF_K + e_rank) + 1.0 / (_RRF_K + b_rank)
            rrf_scores[cid] = raw / rrf_max

        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        results = []
        for cid in top_ids:
            text, abstract, summary, time_range, src_a, src_b = meta_by_id[cid]
            chunk = Chunk(turns=[Turn(timestamp=datetime.min, role="", text=text)])
            cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
            results.append(
                SearchResult(
                    id=cid,
                    chunk_summary=cs,
                    score=rrf_scores[cid],
                    cosine_score=cosine_by_id[cid],
                    source_a=src_a,
                    source_b=src_b,
                )
            )

        return results

    def neighbor_score(self, text: str, top_k: int = 3) -> float:
        """Mean cosine similarity of the top-k neighbors for *text*.

        Returns 0.0 when no active chunks exist.  Uses cosine scores
        rather than RRF ranking scores for meaningful similarity thresholds.
        """
        results = self.search(text, top_k=top_k)
        if not results:
            return 0.0
        return sum(r.cosine_score for r in results) / len(results)

    def deactivate(self, chunk_ids: list[int]) -> None:
        """Mark chunks as inactive. They remain in the DB but are excluded from search."""
        self.conn.executemany(
            "UPDATE chunks SET active = 0 WHERE id = ?",
            [(cid,) for cid in chunk_ids],
        )
        self._auto_commit()

    def activate(self, chunk_ids: list[int]) -> None:
        """Re-activate previously deactivated chunks."""
        self.conn.executemany(
            "UPDATE chunks SET active = 1 WHERE id = ?",
            [(cid,) for cid in chunk_ids],
        )
        self._auto_commit()

    def count(self, active_only: bool = True) -> int:
        """Return the number of stored chunks."""
        query = "SELECT COUNT(*) FROM chunks"
        if active_only:
            query += " WHERE active = 1"
        row = self.conn.execute(query).fetchone()
        return row[0] if row else 0

    def merge_stats(self, chunk_ids: list[int]) -> dict[int, tuple[int, int]]:
        """Return (depth, leaf_count) for each chunk id.

        Depth is the height of the merge tree (0 for originals).
        Leaf count is the number of original inputs the chunk was built from.
        """
        if not chunk_ids:
            return {}
        # Bulk-fetch all chunk sources.
        rows = self.conn.execute("SELECT id, source_a, source_b FROM chunks").fetchall()
        sources: dict[int, tuple[int | None, int | None]] = {
            r[0]: (r[1], r[2]) for r in rows
        }

        cache: dict[int, tuple[int, int]] = {}

        def _walk(cid: int) -> tuple[int, int]:
            if cid in cache:
                return cache[cid]
            sa, sb = sources.get(cid, (None, None))
            if sa is None and sb is None:
                cache[cid] = (0, 1)
                return (0, 1)
            d_a, l_a = _walk(sa) if sa is not None else (0, 0)
            d_b, l_b = _walk(sb) if sb is not None else (0, 0)
            result = (max(d_a, d_b) + 1, l_a + l_b)
            cache[cid] = result
            return result

        return {cid: _walk(cid) for cid in chunk_ids if cid in sources}

    def lineage(self, chunk_id: int) -> LineageNode | None:
        """Build the full merge tree rooted at *chunk_id*.

        Recursively follows source_a / source_b links.  Returns None if
        the chunk does not exist.
        """
        cache: dict[int, LineageNode] = {}

        def _build(cid: int) -> LineageNode | None:
            if cid in cache:
                return cache[cid]
            row = self.conn.execute(
                "SELECT id, text, abstract, active, source_a, source_b "
                "FROM chunks WHERE id = ?",
                (cid,),
            ).fetchone()
            if row is None:
                return None
            node = LineageNode(
                id=row[0],
                text=row[1],
                abstract=row[2],
                active=bool(row[3]),
            )
            # Cache early to handle hypothetical cycles gracefully.
            cache[cid] = node
            if row[4] is not None:
                node.source_a = _build(row[4])
            if row[5] is not None:
                node.source_b = _build(row[5])
            return node

        return _build(chunk_id)

    def leaf_texts(self, chunk_id: int) -> list[str]:
        """Return the texts of all leaf nodes in the merge tree for *chunk_id*.

        Returns ``[]`` if the chunk does not exist.
        """
        node = self.lineage(chunk_id)
        if node is None:
            return []
        return _collect_leaf_texts(node)

    def close(self) -> None:
        self.conn.close()
