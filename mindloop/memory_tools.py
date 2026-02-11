"""Agent-facing memory tools: remember, recall, recall_detail."""

from __future__ import annotations

from pathlib import Path

from mindloop.memory import DEFAULT_DB_PATH, LineageNode, MemoryStore
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunk
from mindloop.chunker import Chunk, Turn
from datetime import datetime


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


class MemoryTools:
    """Encapsulates agent memory tool implementations bound to a store."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        model: str = "openrouter/free",
    ) -> None:
        self._store = MemoryStore(db_path=db_path)
        self._model = model

    @property
    def store(self) -> MemoryStore:
        """Expose the underlying store for testing."""
        return self._store

    def remember(self, text: str, abstract: str) -> str:
        """Save a memory. Auto-generates summary via LLM, then merges."""
        chunk = Chunk(turns=[Turn(timestamp=datetime.now(), role="memory", text=text)])
        cs = summarize_chunk(chunk, model=self._model)
        # Use the LLM-generated summary but keep the agent-provided abstract.
        row_id = save_memory(
            self._store,
            text,
            abstract,
            cs.summary,
            model=self._model,
        )
        return f"Saved as #{row_id}."

    def recall(self, query: str, top_k: int = 5) -> str:
        """Search memory. Returns ranked results with id, abstract, summary, score."""
        results = self._store.search(query, top_k=top_k)
        if not results:
            return "No memories found."
        lines = []
        for rank, r in enumerate(results, 1):
            cs = r.chunk_summary
            lines.append(
                f'[{rank}] #{r.id} (score={r.score:.2f}) "{cs.abstract}"\n'
                f"    {cs.summary}"
            )
        return "\n".join(lines)

    def recall_detail(self, chunk_id: int) -> str:
        """Get full text and merge lineage for a chunk."""
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
            parts.append(f"Lineage:\n{_format_lineage(node)}")

        return "\n".join(parts)

    def close(self) -> None:
        """Close the underlying store."""
        self._store.close()
