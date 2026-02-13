"""Tests for mindloop.cli.query."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.chunker import Chunk, Turn
from mindloop.cli.query import (
    _format_sources,
    _node_label,
    main,
    print_result,
    print_tree,
)
from mindloop.memory import LineageNode, MemoryStore, SearchResult
from mindloop.summarizer import ChunkSummary


def _result(
    chunk_id: int = 1,
    text: str = "hello",
    abstract: str = "abs",
    summary: str = "sum",
    score: float = 0.95,
    source_a: int | None = None,
    source_b: int | None = None,
) -> SearchResult:
    chunk = Chunk(
        turns=[Turn(timestamp=__import__("datetime").datetime.min, role="", text=text)]
    )
    cs = ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
    return SearchResult(
        id=chunk_id,
        chunk_summary=cs,
        score=score,
        source_a=source_a,
        source_b=source_b,
    )


def test_format_sources_original() -> None:
    assert _format_sources(_result()) == "original"


def test_format_sources_single() -> None:
    assert _format_sources(_result(source_a=3)) == "merged from #3"


def test_format_sources_both() -> None:
    assert _format_sources(_result(source_a=3, source_b=7)) == "merged from #3, #7"


def test_print_result_default(capsys: pytest.CaptureFixture[str]) -> None:
    print_result(_result(abstract="test abstract", summary="test summary"), rank=1)
    out = capsys.readouterr().out
    assert "score=0.9500" in out
    assert "test abstract" in out
    assert "test summary" in out
    assert "Text:" not in out


def test_print_result_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    print_result(_result(text="full text here"), rank=1, verbose=True)
    out = capsys.readouterr().out
    assert "Text:" in out
    assert "full text here" in out


def test_main_no_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing database prints error."""
    with patch(
        "sys.argv",
        ["mindloop-query", "test query", "--db", str(tmp_path / "missing.db")],
    ):
        main()
    assert "Database not found" in capsys.readouterr().out


def test_main_empty_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Empty database returns no results."""
    db = tmp_path / "test.db"
    store = MemoryStore(db_path=db)
    store.close()

    with (
        patch(
            "sys.argv",
            ["mindloop-query", "anything", "--db", str(db)],
        ),
        patch(
            "mindloop.memory.get_embeddings",
            return_value=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
    ):
        main()
    assert "No results found" in capsys.readouterr().out


def test_main_returns_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Query returns matching results with scores."""
    db = tmp_path / "test.db"
    store = MemoryStore(db_path=db)
    chunk = Chunk(
        turns=[
            Turn(
                timestamp=__import__("datetime").datetime.min,
                role="",
                text="cats are great",
            )
        ]
    )
    cs = ChunkSummary(chunk=chunk, abstract="about cats", summary="cats summary")
    store.save(cs)
    store.close()

    with (
        patch(
            "sys.argv",
            ["mindloop-query", "cats", "--db", str(db), "-k", "1"],
        ),
        patch(
            "mindloop.memory.get_embeddings",
            return_value=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
    ):
        main()

    out = capsys.readouterr().out
    assert "1 result(s)" in out
    assert "about cats" in out
    assert "cats summary" in out


def test_main_verbose_shows_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbose mode shows full chunk text."""
    db = tmp_path / "test.db"
    store = MemoryStore(db_path=db)
    chunk = Chunk(
        turns=[
            Turn(
                timestamp=__import__("datetime").datetime.min,
                role="",
                text="detailed content",
            )
        ]
    )
    cs = ChunkSummary(chunk=chunk, abstract="abs", summary="sum")
    store.save(cs)
    store.close()

    with (
        patch(
            "sys.argv",
            ["mindloop-query", "test", "--db", str(db), "-v"],
        ),
        patch(
            "mindloop.memory.get_embeddings",
            return_value=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
    ):
        main()

    out = capsys.readouterr().out
    assert "detailed content" in out


def test_main_shows_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Results display source lineage."""
    db = tmp_path / "test.db"
    store = MemoryStore(db_path=db)
    chunk = Chunk(
        turns=[
            Turn(timestamp=__import__("datetime").datetime.min, role="", text="merged")
        ]
    )
    cs = ChunkSummary(chunk=chunk, abstract="abs", summary="sum")
    store.save(cs, source_a=10, source_b=20)
    store.close()

    with (
        patch(
            "sys.argv",
            ["mindloop-query", "test", "--db", str(db)],
        ),
        patch(
            "mindloop.memory.get_embeddings",
            return_value=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
    ):
        main()

    out = capsys.readouterr().out
    assert "merged from #10, #20" in out


# --- lineage tree ---


def _leaf(chunk_id: int, abstract: str = "abs") -> LineageNode:
    return LineageNode(id=chunk_id, text="t", abstract=abstract, active=False)


def test_node_label_active() -> None:
    node = LineageNode(id=1, text="t", abstract="about cats", active=True)
    assert _node_label(node) == '#1 (active) "about cats"'


def test_node_label_leaf() -> None:
    node = _leaf(2, "about dogs")
    assert _node_label(node) == '#2 (leaf) "about dogs"'


def test_node_label_intermediate() -> None:
    node = LineageNode(
        id=3,
        text="t",
        abstract="merged",
        active=False,
        source_a=_leaf(1),
        source_b=_leaf(2),
    )
    assert _node_label(node) == '#3 (intermediate) "merged"'


def test_print_tree_single_leaf(capsys: pytest.CaptureFixture[str]) -> None:
    print_tree(_leaf(1, "only leaf"))
    out = capsys.readouterr().out
    assert '#1 (leaf) "only leaf"' in out


def test_print_tree_full(capsys: pytest.CaptureFixture[str]) -> None:
    root = LineageNode(
        id=5,
        text="t",
        abstract="root",
        active=True,
        source_a=LineageNode(
            id=3,
            text="t",
            abstract="mid",
            active=False,
            source_a=_leaf(1, "leaf_a"),
            source_b=_leaf(2, "leaf_b"),
        ),
        source_b=_leaf(4, "leaf_c"),
    )
    print_tree(root)
    out = capsys.readouterr().out
    # Root has no prefix connector.
    assert out.startswith('#5 (active) "root"')
    # All nodes present.
    assert "#3 (intermediate)" in out
    assert "#1 (leaf)" in out
    assert "#2 (leaf)" in out
    assert "#4 (leaf)" in out
    # Box-drawing characters present.
    assert "\u251c" in out or "\u2514" in out


def test_main_tree_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--tree prints the merge lineage tree."""
    db = tmp_path / "test.db"
    store = MemoryStore(db_path=db)
    id_a = store.save(_result_cs("leaf a", "abs_a"))
    id_b = store.save(_result_cs("leaf b", "abs_b"))
    merged_id = store.save(_result_cs("merged", "abs_m"), source_a=id_a, source_b=id_b)
    store.close()

    with patch(
        "sys.argv", ["mindloop-query", "--tree", str(merged_id), "--db", str(db)]
    ):
        main()

    out = capsys.readouterr().out
    assert "abs_m" in out
    assert "abs_a" in out
    assert "abs_b" in out


def test_main_tree_not_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--tree with nonexistent ID prints error."""
    db = tmp_path / "test.db"
    store = MemoryStore(db_path=db)
    store.close()

    with patch("sys.argv", ["mindloop-query", "--tree", "999", "--db", str(db)]):
        main()

    assert "not found" in capsys.readouterr().out


def _result_cs(text: str, abstract: str = "abs", summary: str = "sum") -> ChunkSummary:
    """Create a ChunkSummary for test data."""
    chunk = Chunk(
        turns=[Turn(timestamp=__import__("datetime").datetime.min, role="", text=text)]
    )
    return ChunkSummary(chunk=chunk, abstract=abstract, summary=summary)
