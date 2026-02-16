"""Agent-facing memory tools: remember, recall, recall_detail."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from mindloop.chunker import Chunk, Turn
from mindloop.memory import DEFAULT_DB_PATH, LineageNode, MemoryStore
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunk
from mindloop.util import noop


def _format_lineage(node: LineageNode, prefix: str = "", is_last: bool = True) -> str:
    """Render a lineage tree as indented text."""
    status = "active" if node.active else "leaf" if node.is_leaf else "intermediate"
    label = f'#{node.id} ({status}) "{node.abstract}"'

    if prefix:
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}{label}"
    else:
        line = label

    lines = [line]
    children = [c for c in (node.source_a, node.source_b) if c is not None]
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        lines.append(
            _format_lineage(child, child_prefix, is_last=(i == len(children) - 1))
        )
    return "\n".join(lines)


def _collect_leaves(node: LineageNode) -> list[LineageNode]:
    """Return all leaf nodes from a lineage tree."""
    if node.is_leaf:
        return [node]
    leaves: list[LineageNode] = []
    if node.source_a is not None:
        leaves.extend(_collect_leaves(node.source_a))
    if node.source_b is not None:
        leaves.extend(_collect_leaves(node.source_b))
    return leaves


class MemoryTools:
    """Encapsulates agent memory tool implementations bound to a store."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        model: str = "",
        stats: dict[Any, Any] | None = None,
        log: Callable[[str], None] = noop,
    ) -> None:
        self._store = MemoryStore(db_path=db_path)
        self._model = model
        self._stats = stats if stats is not None else {}
        self._log = log

    @property
    def store(self) -> MemoryStore:
        """Expose the underlying store for testing."""
        return self._store

    def _track(self, tool: str) -> None:
        """Increment call count for a memory tool."""
        counts = self._stats.setdefault("memory", {})
        counts[tool] = counts.get(tool, 0) + 1

    def remember(self, text: str, abstract: str) -> str:
        """Save a memory. Auto-generates summary via LLM, then merges."""
        self._track("remember")
        chunk = Chunk(turns=[Turn(timestamp=datetime.now(), role="memory", text=text)])
        self._log("[memory] Summarizing...")
        cs = summarize_chunk(chunk, model=self._model)
        # Use the LLM-generated summary but keep the agent-provided abstract.
        row_id = save_memory(
            self._store,
            text,
            abstract,
            cs.summary,
            model=self._model,
            log=self._log,
        )
        return f"Saved as #{row_id}."

    def recall(
        self,
        query: str,
        top_k: int = 5,
        original_only: bool = False,
        diversity: float = 0.0,
    ) -> str:
        """Search memory. Returns ranked results with id, abstract, summary, score."""
        self._track("recall")
        results = self._store.search(
            query, top_k=top_k, original_only=original_only, diversity=diversity
        )
        if not results:
            return "No memories found."
        result_ids = [r.id for r in results]
        stats = self._store.merge_stats(result_ids)
        edge_counts = self._store.edge_counts(result_ids)
        lines = []
        for rank, r in enumerate(results, 1):
            cs = r.chunk_summary
            depth, sources = stats.get(r.id, (0, 1))
            meta = f"score={r.score:.2f}"
            if depth > 0:
                meta += f", depth={depth}, sources={sources}"
            n_edges = edge_counts.get(r.id, 0)
            if n_edges > 0:
                meta += f", +{n_edges} related"
            lines.append(
                f'[{rank}] #{r.id} ({meta}) "{cs.abstract}"\n' f"    {cs.summary}"
            )
        return "\n".join(lines)

    def recall_detail(self, chunk_id: int) -> str:
        """Get full text and merge lineage for a chunk."""
        self._track("recall_detail")
        row = self._store.conn.execute(
            "SELECT text, abstract, summary, active FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return f"Chunk #{chunk_id} not found."

        text, abstract, summary, active = row
        parts = [
            f"#{chunk_id} ({'active' if active else 'inactive'})",
            f"Abstract: {abstract}",
            f"Summary: {summary}",
            f"Text:\n{text}",
        ]

        node = self._store.lineage(chunk_id)
        if node is not None and not node.is_leaf:
            leaves = _collect_leaves(node)
            leaf_lines = [f'  #{lf.id}: "{lf.abstract}"' for lf in leaves]
            parts.append(
                f"Sources ({len(leaves)} originals):\n" + "\n".join(leaf_lines)
            )

        edges = self._store.edges(chunk_id)
        if edges:
            # Look up abstracts for related chunks.
            related_ids = [eid for eid, _, _ in edges]
            abs_rows = self._store.conn.execute(
                f"SELECT id, abstract FROM chunks WHERE id IN "
                f"({','.join('?' for _ in related_ids)})",
                related_ids,
            ).fetchall()
            abs_map = dict(abs_rows)
            edge_lines = [
                f'  #{eid} ({etype}): "{abs_map.get(eid, "?")}"'
                for eid, etype, _ in edges
            ]
            parts.append(f"Related ({len(edges)}):\n" + "\n".join(edge_lines))

        return "\n".join(parts)

    def close(self) -> None:
        """Close the underlying store."""
        self._store.close()
