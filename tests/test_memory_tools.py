"""Tests for mindloop.memory_tools."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.chunker import Chunk, Turn
from mindloop.memory_tools import MemoryTools, _format_lineage
from mindloop.summarizer import ChunkSummary
from mindloop.tools import ToolRegistry, add_memory_tools

_EMB = np.array([1.0, 0.0], dtype=np.float32)


def _mock_embeddings(texts: list[str], **_kw: object) -> np.ndarray:
    return np.tile(_EMB, (len(texts), 1))


def _summary(text: str, abstract: str = "abs", summary: str = "sum") -> ChunkSummary:
    chunk = Chunk(turns=[Turn(timestamp=datetime.min, role="", text=text)])
    return ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)


@pytest.fixture()
def mt(tmp_path: Path) -> MemoryTools:
    return MemoryTools(db_path=tmp_path / "test.db")


# --- remember ---


def test_remember_saves_chunk(mt: MemoryTools) -> None:
    cs = _summary("some fact", summary="auto summary")
    with (
        patch("mindloop.memory_tools.summarize_chunk", return_value=cs),
        patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings),
    ):
        result = mt.remember("some fact", "about facts")

    assert result.startswith("Saved as #")
    assert mt.store.count() == 1


def test_remember_uses_agent_abstract(mt: MemoryTools) -> None:
    """The agent-provided abstract is stored, not the summarizer's."""
    cs = _summary("text", abstract="llm abstract", summary="llm summary")
    with (
        patch("mindloop.memory_tools.summarize_chunk", return_value=cs),
        patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings),
    ):
        result = mt.remember("text", "agent abstract")

    row_id = int(result.split("#")[1].rstrip("."))
    row = mt.store.conn.execute(
        "SELECT abstract, summary FROM chunks WHERE id = ?", (row_id,)
    ).fetchone()
    assert row[0] == "agent abstract"
    # Summary comes from the summarizer.
    assert row[1] == "llm summary"


# --- recall ---


def test_recall_empty(mt: MemoryTools) -> None:
    with patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings):
        result = mt.recall("anything")
    assert result == "No memories found."


def test_recall_returns_results(mt: MemoryTools) -> None:
    mt.store.save(_summary("cats are great", "about cats", "cats summary"))
    with patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings):
        result = mt.recall("cats")

    assert "[1]" in result
    assert "about cats" in result
    assert "cats summary" in result
    assert "score=" in result


def test_recall_respects_top_k(mt: MemoryTools) -> None:
    for i in range(5):
        mt.store.save(_summary(f"fact {i}"))
    with patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings):
        result = mt.recall("fact", top_k=2)

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" not in result


# --- recall_detail ---


def test_recall_detail_not_found(mt: MemoryTools) -> None:
    assert mt.recall_detail(999) == "Chunk #999 not found."


def test_recall_detail_returns_text(mt: MemoryTools) -> None:
    row_id = mt.store.save(_summary("full text here", "the abstract", "the summary"))
    result = mt.recall_detail(row_id)
    assert "full text here" in result
    assert "the abstract" in result
    assert "the summary" in result
    assert "active" in result


def test_recall_detail_shows_lineage(mt: MemoryTools) -> None:
    id_a = mt.store.save(_summary("leaf a", "abs_a"))
    id_b = mt.store.save(_summary("leaf b", "abs_b"))
    merged_id = mt.store.save(_summary("merged", "abs_m"), source_a=id_a, source_b=id_b)
    result = mt.recall_detail(merged_id)
    assert "Lineage:" in result
    assert "abs_a" in result
    assert "abs_b" in result
    assert "abs_m" in result


def test_recall_detail_no_lineage_for_leaf(mt: MemoryTools) -> None:
    row_id = mt.store.save(_summary("leaf"))
    result = mt.recall_detail(row_id)
    assert "Lineage:" not in result


# --- _format_lineage ---


def test_format_lineage_leaf() -> None:
    from mindloop.memory import LineageNode

    node = LineageNode(id=1, text="t", abstract="leaf abs", active=False)
    result = _format_lineage(node)
    assert '#1 (leaf) "leaf abs"' in result


def test_format_lineage_tree() -> None:
    from mindloop.memory import LineageNode

    root = LineageNode(
        id=3,
        text="t",
        abstract="root",
        active=True,
        source_a=LineageNode(id=1, text="t", abstract="left", active=False),
        source_b=LineageNode(id=2, text="t", abstract="right", active=False),
    )
    result = _format_lineage(root)
    assert "#3" in result
    assert "#1" in result
    assert "#2" in result
    assert "├" in result or "└" in result


# --- add_memory_tools ---


def test_add_memory_tools_registers_all(tmp_path: Path) -> None:
    reg = ToolRegistry()
    mt = add_memory_tools(reg, db_path=tmp_path / "test.db")
    names = [d["function"]["name"] for d in reg.definitions()]
    assert "remember" in names
    assert "recall" in names
    assert "recall_detail" in names
    mt.close()


def test_add_memory_tools_execute_recall(tmp_path: Path) -> None:
    """Tools are callable via registry.execute."""
    reg = ToolRegistry()
    mt = add_memory_tools(reg, db_path=tmp_path / "test.db")
    with patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings):
        result = reg.execute("recall", '{"query": "test"}')
    assert "No memories found" in result
    mt.close()


def test_stats_tracking(mt: MemoryTools) -> None:
    """Each tool call increments its counter in stats."""
    mt.store.save(_summary("fact"))
    with patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings):
        mt.recall("test")
        mt.recall("test")
    mt.recall_detail(1)
    assert mt._stats["memory"] == {"recall": 2, "recall_detail": 1}


def test_stats_shared_with_registry(tmp_path: Path) -> None:
    """Stats dict is shared between MemoryTools and ToolRegistry."""
    reg = ToolRegistry()
    mt = add_memory_tools(reg, db_path=tmp_path / "test.db")
    with patch("mindloop.memory.get_embeddings", side_effect=_mock_embeddings):
        reg.execute("recall", '{"query": "test"}')
    assert reg.stats["memory"]["recall"] == 1
    mt.close()
