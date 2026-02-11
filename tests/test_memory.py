"""Tests for mindloop.memory."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mindloop.chunker import Chunk, Turn
from mindloop.memory import MemoryStore
from mindloop.summarizer import ChunkSummary


def _chunk(text: str) -> Chunk:
    return Chunk(turns=[Turn(timestamp=datetime.min, role="You", text=text)])


def _summary(text: str, abstract: str = "abs", summary: str = "sum") -> ChunkSummary:
    return ChunkSummary(chunk=_chunk(text), abstract=abstract, summary=summary)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


def test_save_and_count(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    row_id = store.save(_summary("hello"), emb)
    assert row_id > 0
    assert store.count() == 1


def test_save_with_sources(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    id_a = store.save(_summary("a"), emb)
    id_b = store.save(_summary("b"), emb)
    id_merged = store.save(_summary("merged"), emb, source_a=id_a, source_b=id_b)

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("merged", top_k=1)
    # The top result should be the merged chunk with source info.
    assert results[0].id == id_merged
    assert results[0].source_a == id_a
    assert results[0].source_b == id_b


def test_save_without_sources_returns_none(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    store.save(_summary("no sources"), emb)

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("no sources", top_k=1)
    assert results[0].source_a is None
    assert results[0].source_b is None


def test_save_many(store: MemoryStore) -> None:
    summaries = [_summary("a"), _summary("b"), _summary("c")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    ids = store.save_many(summaries, embeddings)
    assert len(ids) == 3
    assert store.count() == 3


def test_search_empty(store: MemoryStore) -> None:
    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("anything")
    assert results == []


def test_search_returns_most_similar(store: MemoryStore) -> None:
    # Store three chunks with distinct embeddings.
    store.save(
        _summary("about cats", abstract="cats"),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    store.save(
        _summary("about dogs", abstract="dogs"),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    store.save(
        _summary("about fish", abstract="fish"),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    # Query embedding is close to "cats".
    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[0.9, 0.1, 0.0]], dtype=np.float32),
    ):
        results = store.search("cats", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_summary.abstract == "cats"
    assert results[0].score > results[1].score


def test_search_top_k_limits_results(store: MemoryStore) -> None:
    for i in range(10):
        store.save(_summary(f"chunk {i}"), np.array([float(i), 0.0], dtype=np.float32))

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[9.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("query", top_k=3)

    assert len(results) == 3


def test_deactivate_excludes_from_search(store: MemoryStore) -> None:
    id1 = store.save(
        _summary("cats", abstract="cats"), np.array([1.0, 0.0], dtype=np.float32)
    )
    store.save(
        _summary("dogs", abstract="dogs"), np.array([0.0, 1.0], dtype=np.float32)
    )

    store.deactivate([id1])
    assert store.count() == 1
    assert store.count(active_only=False) == 2

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("cats")
    # "cats" is deactivated, only "dogs" should appear.
    assert len(results) == 1
    assert results[0].chunk_summary.abstract == "dogs"


def test_specificity_no_chunks(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    assert store.specificity(emb) == 1.0


def test_specificity_no_neighbors(store: MemoryStore) -> None:
    # Store a chunk orthogonal to the query.
    store.save(_summary("a"), np.array([0.0, 1.0], dtype=np.float32))
    emb = np.array([1.0, 0.0], dtype=np.float32)
    assert store.specificity(emb) == 1.0


def test_specificity_all_neighbors(store: MemoryStore) -> None:
    # Store chunks identical to the query.
    for _ in range(5):
        store.save(_summary("a"), np.array([1.0, 0.0], dtype=np.float32))
    emb = np.array([1.0, 0.0], dtype=np.float32)
    assert store.specificity(emb) == 0.0


def test_specificity_partial_neighbors(store: MemoryStore) -> None:
    # 2 similar, 2 orthogonal.
    store.save(_summary("a"), np.array([1.0, 0.0], dtype=np.float32))
    store.save(_summary("b"), np.array([0.95, 0.31], dtype=np.float32))
    store.save(_summary("c"), np.array([0.0, 1.0], dtype=np.float32))
    store.save(_summary("d"), np.array([-1.0, 0.0], dtype=np.float32))
    emb = np.array([1.0, 0.0], dtype=np.float32)
    # 2 out of 4 are above 0.5 threshold.
    assert store.specificity(emb) == 0.5


def test_specificity_ignores_inactive(store: MemoryStore) -> None:
    id1 = store.save(_summary("a"), np.array([1.0, 0.0], dtype=np.float32))
    store.save(_summary("b"), np.array([0.0, 1.0], dtype=np.float32))
    store.deactivate([id1])
    emb = np.array([1.0, 0.0], dtype=np.float32)
    # Only the orthogonal chunk is active.
    assert store.specificity(emb) == 1.0


def test_activate_restores_to_search(store: MemoryStore) -> None:
    id1 = store.save(
        _summary("cats", abstract="cats"), np.array([1.0, 0.0], dtype=np.float32)
    )
    store.deactivate([id1])
    assert store.count() == 0

    store.activate([id1])
    assert store.count() == 1


# --- transaction ---


def test_transaction_commits_on_success(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    with store.transaction():
        store.save(_summary("a"), emb)
        store.save(_summary("b"), emb)
    assert store.count() == 2


def test_transaction_rolls_back_on_error(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    store.save(_summary("existing"), emb)
    try:
        with store.transaction():
            store.deactivate([1])
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    # Deactivation should have been rolled back.
    assert store.count() == 1


def test_transaction_nested(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    with store.transaction():
        store.save(_summary("a"), emb)
        with store.transaction():
            store.save(_summary("b"), emb)
        # Inner transaction doesn't commit yet.
    # Outer transaction commits both.
    assert store.count() == 2


def test_transaction_nested_inner_error_rolls_back_all(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    try:
        with store.transaction():
            store.save(_summary("a"), emb)
            with store.transaction():
                store.save(_summary("b"), emb)
                raise RuntimeError("inner failure")
    except RuntimeError:
        pass
    # Both saves should be rolled back.
    assert store.count() == 0


# --- lineage ---


def test_lineage_leaf(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    leaf_id = store.save(_summary("leaf", abstract="leaf_abs"), emb)
    node = store.lineage(leaf_id)
    assert node is not None
    assert node.id == leaf_id
    assert node.abstract == "leaf_abs"
    assert node.is_leaf
    assert node.source_a is None
    assert node.source_b is None


def test_lineage_single_merge(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    id_a = store.save(_summary("a", abstract="abs_a"), emb)
    id_b = store.save(_summary("b", abstract="abs_b"), emb)
    merged_id = store.save(
        _summary("merged", abstract="abs_m"), emb, source_a=id_a, source_b=id_b
    )

    node = store.lineage(merged_id)
    assert node is not None
    assert node.id == merged_id
    assert not node.is_leaf
    assert node.source_a is not None and node.source_a.id == id_a
    assert node.source_b is not None and node.source_b.id == id_b
    assert node.source_a.is_leaf
    assert node.source_b.is_leaf


def test_lineage_deep_tree(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    id_a = store.save(_summary("a"), emb)
    id_b = store.save(_summary("b"), emb)
    id_ab = store.save(_summary("ab"), emb, source_a=id_a, source_b=id_b)
    id_c = store.save(_summary("c"), emb)
    id_abc = store.save(_summary("abc"), emb, source_a=id_ab, source_b=id_c)

    root = store.lineage(id_abc)
    assert root is not None
    assert root.id == id_abc
    # Left subtree is itself a merge.
    assert root.source_a is not None and root.source_a.id == id_ab
    assert root.source_a.source_a is not None and root.source_a.source_a.id == id_a
    assert root.source_a.source_b is not None and root.source_a.source_b.id == id_b
    # Right is a leaf.
    assert root.source_b is not None and root.source_b.id == id_c
    assert root.source_b.is_leaf


def test_lineage_nonexistent(store: MemoryStore) -> None:
    assert store.lineage(999) is None


def test_lineage_tracks_active_status(store: MemoryStore) -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)
    id_a = store.save(_summary("a"), emb)
    store.deactivate([id_a])
    id_b = store.save(_summary("b"), emb)
    merged_id = store.save(_summary("m"), emb, source_a=id_a, source_b=id_b)

    node = store.lineage(merged_id)
    assert node is not None
    assert node.active
    assert node.source_a is not None and not node.source_a.active
    assert node.source_b is not None and node.source_b.active


# --- hybrid search (BM25 + embeddings) ---


def test_search_bm25_boosts_keyword_match(store: MemoryStore) -> None:
    """BM25 should boost a chunk that matches the query keyword even when
    all embeddings are identical."""
    emb = np.array([1.0, 0.0], dtype=np.float32)
    store.save(
        _summary("the gfx942 architecture requires special flags", abstract="gfx"), emb
    )
    store.save(
        _summary("general information about compilers", abstract="compilers"), emb
    )
    store.save(_summary("how to set up the environment", abstract="env"), emb)

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("gfx942")

    # With identical embeddings, BM25 keyword match should push gfx to the top.
    assert results[0].chunk_summary.abstract == "gfx"


def test_search_hybrid_combines_both_signals(store: MemoryStore) -> None:
    """A chunk that scores well on both embedding and BM25 should rank above
    chunks that only score well on one."""
    # Chunk A: good embedding, good keyword match.
    store.save(
        _summary("cats are wonderful pets", abstract="cats"),
        np.array([0.9, 0.1], dtype=np.float32),
    )
    # Chunk B: good embedding, no keyword match.
    store.save(
        _summary("dogs are loyal animals", abstract="dogs"),
        np.array([0.85, 0.15], dtype=np.float32),
    )
    # Chunk C: bad embedding, good keyword match.
    store.save(
        _summary("cats in ancient history", abstract="ancient cats"),
        np.array([0.1, 0.9], dtype=np.float32),
    )

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("cats", top_k=3)

    # Chunk A should rank first — strong in both signals.
    assert results[0].chunk_summary.abstract == "cats"


def test_search_original_only_with_bm25(store: MemoryStore) -> None:
    """original_only flag should filter correctly with hybrid search."""
    emb = np.array([1.0, 0.0], dtype=np.float32)
    id_a = store.save(_summary("unique keyword xyzzy", abstract="leaf"), emb)
    store.save(
        _summary("merged result with xyzzy", abstract="merged"),
        emb,
        source_a=id_a,
        source_b=id_a,
    )

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        results = store.search("xyzzy", original_only=True)

    # Only the leaf should be returned.
    abstracts = [r.chunk_summary.abstract for r in results]
    assert "leaf" in abstracts
    assert "merged" not in abstracts


def test_search_graceful_without_fts_match(store: MemoryStore) -> None:
    """Search still works when no FTS5 matches exist (falls back to embeddings)."""
    store.save(
        _summary("about something", abstract="something"),
        np.array([1.0, 0.0], dtype=np.float32),
    )

    with patch(
        "mindloop.memory.get_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        # Query with a word not in any chunk text.
        results = store.search("nonexistent_word_zzz")

    # Should still return the embedding match.
    assert len(results) == 1
    assert results[0].chunk_summary.abstract == "something"


def test_find_exact_returns_active_match(store: MemoryStore) -> None:
    """find_exact returns the id of an active chunk with identical text."""
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cs = _summary("hello world")
    row_id = store.save(cs, emb)
    assert store.find_exact(cs.chunk.text) == row_id


def test_find_exact_ignores_inactive(store: MemoryStore) -> None:
    """find_exact does not return deactivated chunks."""
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cs = _summary("hello world")
    row_id = store.save(cs, emb)
    store.deactivate([row_id])
    assert store.find_exact(cs.chunk.text) is None


def test_find_exact_no_match(store: MemoryStore) -> None:
    """find_exact returns None when no chunk matches."""
    assert store.find_exact("nonexistent") is None
